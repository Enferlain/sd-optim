from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import optuna
import optuna.visualization as vis
from hydra.core.hydra_config import HydraConfig
from optuna import Study
from optuna.trial import FrozenTrial, TrialState

if TYPE_CHECKING:
    from sd_optim.optuna_optimizer import OptunaOptimizer

logger = logging.getLogger(__name__)


def record_trial_callback(optimizer: OptunaOptimizer, study: Study, trial: FrozenTrial) -> None:
    elapsed_time = time.time() - optimizer.optimization_start_time if optimizer.optimization_start_time else 0
    trial.set_user_attr("elapsed_seconds", elapsed_time)
    trial.set_user_attr("timestamp", time.time())

    if hasattr(optimizer, "merger") and hasattr(optimizer.merger, "output_file"):
        trial.set_user_attr("model_path", str(optimizer.merger.output_file))
    if hasattr(optimizer, "iteration"):
        trial.set_user_attr("iteration", optimizer.iteration)

    optimizer.logger.log(
        {
            "trial_number": trial.number,
            "target": trial.value,
            "params": trial.params,
            "state": trial.state.name,
            "datetime": {
                "datetime": trial.datetime_start.isoformat() if trial.datetime_start else None,
                "elapsed_seconds": elapsed_time,
            },
            "scorer_results": trial.user_attrs.get("scorer_results", {}),
            "scorer_results_payloads": trial.user_attrs.get("scorer_results_payloads", []),
        }
    )

    if study is not None and trial.number % 10 == 0 and trial.number > 0:
        create_progress_plots(optimizer)


def _get_hydra_output_dir(context_name: str) -> Path | None:
    try:
        return Path(HydraConfig.get().runtime.output_dir)
    except ValueError:
        logger.error("%s: Hydra context unavailable. Cannot determine save path.", context_name)
        return None


def _write_plot_image(fig: Any, save_path: Path, *, label: str) -> None:
    try:
        fig.write_image(str(save_path))
        logger.info("Saved %s to %s", label, save_path)
    except ValueError as error:
        if "kaleido" in str(error).lower():
            logger.error("Failed to save %s: Kaleido missing/broken.", label)
        else:
            logger.error("ValueError saving %s: %s.", label, error)
    except Exception as error:
        logger.error("Error saving %s: %s.", label, error)


def create_progress_plots(optimizer: OptunaOptimizer) -> None:
    if not optimizer.study or not optimizer.study.trials:
        logger.warning("_create_progress_plots: No study or trials available.")
        return

    completed_trials = [trial for trial in optimizer.study.trials if trial.state == TrialState.COMPLETE and trial.value is not None]
    if len(completed_trials) < 1:
        logger.debug("_create_progress_plots: Not enough completed trials yet.")
        return

    try:
        fig = vis.plot_optimization_history(optimizer.study)
        run_dir = _get_hydra_output_dir("_create_progress_plots")
        if run_dir is None:
            return
        output_dir = run_dir / "visualizations" / "periodic"
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"optuna_history_trial_{optimizer.study.trials[-1].number:04d}.png"
        _write_plot_image(fig, save_path, label="periodic progress plot")
    except ImportError:
        logger.error("Optuna visualization module not available for periodic plots.")
    except Exception as error:
        logger.error("Failed to create periodic progress plot: %s", error, exc_info=True)


def analyze_parameter_importance(optimizer: OptunaOptimizer) -> None:
    if not optimizer.study or len(optimizer.study.trials) < 2:
        logger.warning("_analyze_parameter_importance: Not enough trials.")
        return

    try:
        try:
            importance = optuna.importance.get_param_importances(optimizer.study)
            logger.info("\nParameter Importance Analysis:")
            for param_name, score in importance.items():
                logger.info("  %s: %.4f", param_name, score)
        except Exception as error:
            logger.error("Could not calculate parameter importance: %s", error)
            return

        fig = vis.plot_param_importances(optimizer.study)
        run_dir = _get_hydra_output_dir("_analyze_parameter_importance")
        if run_dir is None:
            return
        output_dir = run_dir / "visualizations"
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"optuna_param_importances_{optimizer.study_name}.png"
        _write_plot_image(fig, save_path, label="parameter importance plot")
    except ImportError as error:
        logger.error("Could not generate importance plot due to missing dependency: %s", error)
    except Exception as error:
        logger.error("Failed to analyze/plot parameter importance: %s", error, exc_info=True)


async def postprocess_study(optimizer: OptunaOptimizer) -> None:
    logger.info("\n%s", "=" * 50)
    logger.info("Optimization Results Recap!")
    logger.info("%s", "=" * 50)

    if not optimizer.study or not optimizer.study.trials:
        logger.warning("No trials completed. Nothing to report.")
        return

    total_trials = len(optimizer.study.trials)
    completed_trials = sum(1 for trial in optimizer.study.trials if trial.state.is_finished())
    logger.info("Total Trials: %s", total_trials)
    logger.info("Completed Trials: %s", completed_trials)

    if optimizer.optimization_start_time:
        total_runtime = time.time() - optimizer.optimization_start_time
        hours, remainder = divmod(total_runtime, 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info("Total runtime: %02d:%02d:%02d", int(hours), int(minutes), int(seconds))
        logger.info("Average time per trial: %.2f seconds", total_runtime / max(1, total_trials))

    logger.info("\nTop 5 Trials:")
    successful_trials = [trial for trial in optimizer.study.trials if trial.value is not None]
    if not successful_trials:
        logger.warning("No successful completed trials to summarize yet.")
        return

    sorted_trials = sorted(successful_trials, key=lambda trial: trial.value, reverse=True)
    for index, trial in enumerate(sorted_trials[:5], start=1):
        logger.info("Rank %s (Trial %s):", index, trial.number)
        logger.info("\tValue: %.4f", trial.value)
        logger.info("\tParameters: %s", trial.params)

    best_trial = sorted_trials[0]
    logger.info("\nBest Trial:")
    logger.info("Value: %.4f", best_trial.value)
    logger.info("Parameters: %s", best_trial.params)

    logger.info("Generating Optuna visualizations...")
    if len(optimizer.study.trials) < 2:
        logger.warning("Not enough trials (need >= 2) for most Optuna visualizations.")
        return

    try:
        run_dir = _get_hydra_output_dir("postprocess")
        if run_dir is None:
            return
        output_dir = run_dir / "visualizations"
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Saving Optuna plots to: %s", output_dir)

        plot_functions = {
            "optimization_history": vis.plot_optimization_history,
            "param_importances": vis.plot_param_importances,
            "slice": vis.plot_slice,
            "parallel_coordinate": vis.plot_parallel_coordinate,
        }

        for name, plot_func in plot_functions.items():
            try:
                fig = plot_func(optimizer.study)
                save_path = output_dir / f"optuna_{name}_{optimizer.study_name}.png"
                _write_plot_image(fig, save_path, label=f"Optuna plot '{name}'")
            except (ValueError, TypeError) as error:
                logger.warning("Could not generate Optuna plot '%s': %s. Skipping.", name, error)
            except ImportError as error:
                logger.warning("Could not generate Optuna plot '%s' due to missing dependency: %s. Skipping.", name, error)
            except Exception as error:
                logger.error("Unexpected error generating Optuna plot '%s': %s", name, error, exc_info=True)
    except ImportError:
        logger.error("Optuna library seems to be missing? Cannot generate visualizations.")
    except Exception as error:
        logger.error("An error occurred during Optuna visualization generation: %s", error, exc_info=True)

    logger.info("%s", "=" * 50)
    logger.info("Optuna Postprocessing Finished")
    logger.info("%s", "=" * 50)


def collect_optimization_history(study: Study | None) -> list[dict[str, Any]]:
    if not study:
        return []

    history: list[dict[str, Any]] = []
    for trial in study.trials:
        if trial.value is None:
            continue
        history.append(
            {
                "trial_number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "datetime": trial.datetime_start.isoformat() if trial.datetime_start else None,
            }
        )
    return history


def create_visualization_report(optimizer: OptunaOptimizer, output_dir: str | Path | None = None) -> None:
    if not optimizer.study or len(optimizer.study.trials) < 2:
        logger.warning("Not enough trials for visualization report.")
        return

    if output_dir is None:
        run_dir = _get_hydra_output_dir("create_visualization_report")
        if run_dir is None:
            output_dir = Path("./optuna_visualizations_report")
            logger.warning("Using fallback directory for report: %s", output_dir)
        else:
            output_dir = run_dir / "visualizations_report"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating Optuna visualization report in: %s", output_dir)

    try:
        report_plots = {
            "optimization_history": vis.plot_optimization_history,
            "param_importances": vis.plot_param_importances,
            "slice": vis.plot_slice,
            "parallel_coordinate": vis.plot_parallel_coordinate,
            "contour": vis.plot_contour,
            "edf": vis.plot_edf,
            "rank": vis.plot_rank,
        }

        for name, plot_func in report_plots.items():
            try:
                fig = plot_func(optimizer.study)
                save_path = output_dir / f"optuna_{name}_{optimizer.study_name}.png"
                _write_plot_image(fig, save_path, label=f"report plot '{name}'")
            except (ValueError, TypeError) as error:
                logger.warning("Report: Cannot generate plot '%s': %s. Skipping.", name, error)
            except ImportError as error:
                logger.warning("Report: Cannot generate plot '%s', missing dependency: %s. Skipping.", name, error)
            except Exception as error:
                logger.error("Report: Unexpected error generating plot '%s': %s", name, error, exc_info=True)
        logger.info("Visualization report generation finished.")
    except ImportError:
        logger.error("Optuna library seems missing? Cannot generate report.")
    except Exception as error:
        logger.error("Error generating visualization report: %s", error, exc_info=True)
