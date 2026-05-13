from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import optuna
from hydra.core.hydra_config import HydraConfig
from omegaconf import ListConfig

from sd_optim.guide_runtime import GraphRuntimeBundle
from sd_optim.optimizers.optuna.objective import run_objective
from sd_optim.optimizers.optuna.reporting import analyze_parameter_importance, record_trial_callback
from sd_optim.optimizers.optuna.sampler_factory import configure_sampler
from sd_optim.optimizers.optuna.trial_logger import TrialLogger

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from optuna.pruners import BasePruner
    from optuna.study import Study

    from sd_optim.optimizers.optuna.optimizer import OptunaOptimizer

logger = logging.getLogger(__name__)


def initialize_optuna_state(optimizer: OptunaOptimizer) -> None:
    optimizer.study = None
    optimizer._param_bounds = None
    optimizer.study_name = None
    optimizer.optuna_storage_dir = initialize_storage_dir(optimizer.cfg)
    optimizer.early_stopping = optimizer.cfg.optimizer.optuna_config.get("early_stopping", False)
    optimizer.patience = optimizer.cfg.optimizer.optuna_config.get("patience", 10)
    optimizer.min_improvement = optimizer.cfg.optimizer.optuna_config.get("min_improvement", 0.001)
    optimizer.no_improvement_count = 0
    optimizer.trial_scores = []
    optimizer.logger = TrialLogger()
    dependencies_cfg = optimizer.cfg.optimization_guide.get("dependencies", [])
    guide_runtime = getattr(optimizer, "guide_runtime", None)
    if isinstance(guide_runtime, GraphRuntimeBundle):
        if dependencies_cfg:
            raise ValueError(
                "optimization_guide.dependencies is not supported with graph runtime bundles yet."
            )
        optimizer.child_to_parent = {}
    else:
        optimizer.child_to_parent = optimizer.bounds_initializer.validate_dependencies(
            optimizer.param_info,
            dependencies_cfg,
        )


def initialize_storage_dir(cfg: DictConfig) -> Path:
    project_root = Path(__file__).resolve().parents[3]
    default_storage_path = project_root / "optuna_db"
    optuna_storage_path_str = cfg.optimizer.optuna_config.get("storage_dir", str(default_storage_path))
    storage_dir = Path(optuna_storage_path_str).resolve()
    storage_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Optuna databases will be stored in: %s", storage_dir)
    return storage_dir


def configure_pruner(cfg: DictConfig) -> BasePruner | None:
    if not cfg.optimizer.optuna_config.get("use_pruning", False):
        return None

    pruner_type = cfg.optimizer.optuna_config.get("pruner_type", "median")
    if pruner_type == "median":
        logger.info("Using Median Pruner")
        return optuna.pruners.MedianPruner(n_startup_trials=cfg.optimizer.init_points)
    if pruner_type == "successive_halving":
        logger.info("Using Successive_halving Pruner")
        return optuna.pruners.SuccessiveHalvingPruner()

    logger.warning("Unknown pruner_type '%s'. Continuing without pruning.", pruner_type)
    return None


def flatten_scorers(scorers_obj: Any) -> list[str]:
    flattened: list[str] = []
    if isinstance(scorers_obj, (list, ListConfig)):
        for item in scorers_obj:
            flattened.extend(flatten_scorers(item))
        return flattened

    flattened.append(str(scorers_obj))
    return flattened


def sanitize_name(name: Any) -> str:
    safe_name = str(name).replace("\\", "/").split("/")[-1]
    return "".join(char for char in safe_name if char.isalnum() or char in ("_", "-"))


def build_storage_uri_for_new_study(
    cfg: DictConfig,
    optuna_storage_dir: Path,
    *,
    is_fork: bool = False,
) -> tuple[str, str]:
    try:
        opt_mode = cfg.get("optimization_mode", "unknown_mode")
        merge_method_name = "N/A"
        if opt_mode == "merge":
            merge_method_name = cfg.merge.merge_method
        elif opt_mode == "recipe":
            recipe_path_str = cfg.recipe_optimization.get("recipe_path")
            merge_method_name = f"recipe_{Path(recipe_path_str).stem}" if recipe_path_str else "recipe"
        elif opt_mode == "layer_adjust":
            merge_method_name = "layer_adjust"

        scorers_str_list = sorted(flatten_scorers(cfg.scoring.scorer_method))
        scorer_name_part = "_".join(scorers_str_list)
        fork_prefix = "fork_" if is_fork else ""

        if opt_mode == "recipe":
            db_filename_base = f"optuna_{fork_prefix}{sanitize_name(merge_method_name)}_{sanitize_name(scorer_name_part)}"
        else:
            db_filename_base = (
                f"optuna_{fork_prefix}{sanitize_name(opt_mode)}_"
                f"{sanitize_name(merge_method_name)}_{sanitize_name(scorer_name_part)}"
            )

        db_filename = f"{db_filename_base}.db"
        storage_path = optuna_storage_dir / db_filename
        storage_uri = f"sqlite:///{storage_path.resolve()}"
        logger.info("Determined storage for new study/fork: '%s'", db_filename)
        return storage_uri, db_filename
    except Exception as error:
        logger.error("Failed to determine DB name, falling back to default: %s", error, exc_info=True)
        db_filename = "optuna_fallback.db"
        storage_path = optuna_storage_dir / db_filename
        return f"sqlite:///{storage_path.resolve()}", db_filename


def find_db_for_study(optuna_storage_dir: Path, study_name_to_find: str) -> tuple[str | None, str | None]:
    logger.info("Searching for study '%s' in all .db files...", study_name_to_find)
    for db_file in optuna_storage_dir.glob("*.db"):
        storage_uri = f"sqlite:///{db_file.resolve()}"
        try:
            for summary in optuna.study.get_all_study_summaries(storage=storage_uri):
                if summary.study_name == study_name_to_find:
                    logger.info("Found study '%s' in database: '%s'", study_name_to_find, db_file.name)
                    return storage_uri, db_file.name
        except Exception as error:
            logger.warning("Could not inspect database '%s': %s", db_file.name, error)
    logger.error("Study '%s' was not found in any database in %s", study_name_to_find, optuna_storage_dir)
    return None, None


def set_initial_study_attributes(
    study: Study,
    cfg: DictConfig,
    study_name: str,
    *,
    parent_name: str | None = None,
) -> None:
    try:
        models_str = str([str(path) for path in cfg.merge.model_paths])
        study.set_user_attr("config_input_models", models_str)
        study.set_user_attr("config_base_model_index", cfg.merge.base_model_index)
        study.set_user_attr("config_optimization_mode", cfg.get("optimization_mode", "N/A"))
        study.set_user_attr("config_scorers", list(flatten_scorers(cfg.scoring.scorer_method)))

        if cfg.get("optimization_mode") == "merge":
            study.set_user_attr("config_merge_method", cfg.merge.merge_method)
        if parent_name:
            study.set_user_attr("forked_from", parent_name)

        logger.info("Stored initial configuration as user attributes for new study '%s'.", study_name)
    except Exception as error:
        logger.warning("Could not store config as study attributes: %s", error)


def setup_trial_logger_path(trial_logger: TrialLogger, study_name: str) -> None:
    try:
        hydra_run_path = Path(HydraConfig.get().runtime.output_dir)
        trials_log_filename = f"{study_name}_trials.jsonl"
        trial_logger.set_path(hydra_run_path / trials_log_filename)
    except Exception as error:
        logger.error("Failed to set trial logger path: %s", error)


def restore_trials_from_log(study: Study, trial_logger: TrialLogger) -> None:
    trials_data = trial_logger.load_trials()
    if not trials_data:
        logger.warning("No trials found in log file. Starting fresh optimization.")
        return

    logger.info("Restoring %s trials from log file", len(trials_data))
    for trial_data in trials_data:
        if "params" not in trial_data or "target" not in trial_data:
            continue

        value = trial_data["target"]
        if value is None:
            continue

        study.add_trial(
            optuna.trial.create_trial(
                params=trial_data["params"],
                value=value,
                state=optuna.trial.TrialState.COMPLETE,
            )
        )

    logger.info("Restored %s valid trials", len(study.trials))


async def optimize_study(optimizer: OptunaOptimizer) -> None:
    optimizer.optimization_start_time = time.time()
    logger.debug("Optimizer bounds already prepared for %s parameters.", len(optimizer.optimizer_pbounds))

    sampler = configure_sampler(optimizer.cfg, optimizer_pbounds=optimizer.optimizer_pbounds)
    pruner = configure_pruner(optimizer.cfg)
    optuna_cfg = optimizer.cfg.optimizer.optuna_config
    parent_study_name_to_load = optuna_cfg.get("resume_from_study")
    should_fork_study = optuna_cfg.get("fork_study", False)

    if not parent_study_name_to_load:
        logger.info("No 'resume_from_study' specified. Creating a brand-new study.")
        storage_uri, db_filename = build_storage_uri_for_new_study(
            optimizer.cfg,
            optimizer.optuna_storage_dir,
        )
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        sampler_name = optuna_cfg.get("sampler", {}).get("type", "tpe")
        optimizer.study_name = f"run_{timestamp}_{sampler_name}"
        optimizer.study = optuna.create_study(
            study_name=optimizer.study_name,
            storage=storage_uri,
            sampler=sampler,
            pruner=pruner,
            direction="maximize",
            load_if_exists=False,
        )
        logger.info("Successfully created new study '%s' in '%s'.", optimizer.study_name, db_filename)
        set_initial_study_attributes(optimizer.study, optimizer.cfg, optimizer.study_name)
    else:
        logger.info("Attempting to load parent study '%s'.", parent_study_name_to_load)
        parent_storage_uri, _ = find_db_for_study(optimizer.optuna_storage_dir, parent_study_name_to_load)
        if not parent_storage_uri:
            raise ValueError(f"Could not find study '{parent_study_name_to_load}' to resume/fork.")

        parent_study = optuna.load_study(study_name=parent_study_name_to_load, storage=parent_storage_uri)
        if should_fork_study:
            logger.info("Forking study '%s' into a new study.", parent_study_name_to_load)
            new_storage_uri, new_db_filename = build_storage_uri_for_new_study(
                optimizer.cfg,
                optimizer.optuna_storage_dir,
                is_fork=True,
            )
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            optimizer.study_name = f"fork_{parent_study_name_to_load}_{timestamp}"
            optimizer.study = optuna.create_study(
                study_name=optimizer.study_name,
                storage=new_storage_uri,
                sampler=sampler,
                pruner=pruner,
                direction="maximize",
            )

            completed_trials = [trial for trial in parent_study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
            for trial in completed_trials:
                optimizer.study.enqueue_trial(trial.params)

            logger.info(
                "Created fork '%s' in '%s' and enqueued %s parent trials.",
                optimizer.study_name,
                new_db_filename,
                len(completed_trials),
            )
            set_initial_study_attributes(
                optimizer.study,
                optimizer.cfg,
                optimizer.study_name,
                parent_name=parent_study_name_to_load,
            )
        else:
            logger.info("Resuming study '%s' directly.", parent_study_name_to_load)
            parent_scorers = set(map(str, parent_study.user_attrs.get("config_scorers", [])))
            current_scorers = set(flatten_scorers(optimizer.cfg.scoring.scorer_method))
            if parent_scorers != current_scorers:
                raise ValueError(
                    f"FATAL: Cannot resume study '{parent_study_name_to_load}' because scorers have changed. "
                    f"(Study used {sorted(parent_scorers)}, config has {sorted(current_scorers)}). "
                    "To proceed, set 'fork_study: true'."
                )

            optimizer.study = parent_study
            optimizer.study_name = parent_study_name_to_load
            optimizer.completed_trials = len(parent_study.trials)
            logger.info(
                "Re-using existing sampler of type '%s' from the loaded study.",
                type(optimizer.study.sampler).__name__,
            )
            logger.info("Setting iteration offset to %s to account for existing trials.", optimizer.completed_trials)

    setup_trial_logger_path(optimizer.logger, optimizer.study_name)
    completed_trials = len(optimizer.study.trials)
    if completed_trials > 0:
        logger.info("Writing %s existing trials to the new log file...", len(optimizer.study.trials))
        for existing_trial in optimizer.study.trials:
            record_trial_callback(optimizer, optimizer.study, existing_trial)

    total_trials_planned = optimizer.cfg.optimizer.init_points + optimizer.cfg.optimizer.n_iters
    remaining_trials = max(0, total_trials_planned - completed_trials)

    if completed_trials > 0:
        n_startup_trials = optimizer.cfg.optimizer.init_points
        remaining_startup_trials = max(0, n_startup_trials - completed_trials)
        remaining_exploration_trials = max(0, remaining_trials - remaining_startup_trials)
        logger.info("Study has %s existing trials. %s new trials to run.", completed_trials, remaining_trials)
        if remaining_startup_trials > 0:
            logger.info(
                "  (%s init trials + %s optimization trials)",
                remaining_startup_trials,
                remaining_exploration_trials,
            )
        else:
            logger.info("  (%s optimization trials)", remaining_exploration_trials)

        if optimizer.study.best_trial:
            optimizer.best_rolling_score = max(optimizer.best_rolling_score, optimizer.study.best_value)

    if remaining_trials > 0:
        logger.info("Starting optimization for %s new trials.", remaining_trials)

        def objective(trial: optuna.Trial) -> float:
            return run_objective(optimizer, trial)

        def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            record_trial_callback(optimizer, study, trial)

        try:
            await asyncio.to_thread(
                optimizer.study.optimize,
                func=objective,
                n_trials=remaining_trials,
                n_jobs=optuna_cfg.get("n_jobs", 1),
                show_progress_bar=True,
                callbacks=[callback],
            )
        except (KeyboardInterrupt, Exception) as error:
            logger.error(
                "Optimization loop stopped: %s",
                error,
                exc_info=not isinstance(error, KeyboardInterrupt),
            )
            if not isinstance(error, KeyboardInterrupt):
                raise
    else:
        logger.info("All planned trials already completed. Skipping optimization.")

    analyze_parameter_importance(optimizer)
