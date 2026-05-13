import asyncio
import hydra
import logging
import subprocess
import sys
from pathlib import Path
from typing import Literal

from omegaconf import DictConfig

from sd_optim.config_schema import register_config_schemas
from sd_optim.utils.conversions import (
    load_and_register_custom_configs,
    load_and_register_custom_conversion,
)

# Configure logging level and format early. Can be overridden by Hydra later.
logging.basicConfig(
    level=logging.INFO,  # Default level
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Suppress noisy third-party loggers - ADD THIS HERE
logging.getLogger("choreographer").setLevel(logging.WARNING)
logging.getLogger("kaleido").setLevel(logging.WARNING)
logging.getLogger("plotly").setLevel(logging.WARNING)  # Just in case
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)
logging.getLogger("PIL.Image").setLevel(logging.WARNING)
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("numba.core").setLevel(logging.WARNING)
logging.getLogger("httpcore.http11").setLevel(logging.WARNING)

# Use a logger specific to this main script
logger = logging.getLogger(__name__)  # Hydra often configures this further

register_config_schemas()


def _determine_extension_paths(cfg: DictConfig) -> tuple[Path, Path]:
    """Resolve model-config and conversion search paths."""
    project_root = Path(__file__).resolve().parent
    default_dir = project_root / "sd_optim" / "extensions" / "bundled" / "model_configs"

    configs_dir_str = cfg.get("configs_dir")
    conversion_dir_str = cfg.get("conversion_dir")

    custom_configs_path = Path(configs_dir_str).resolve() if configs_dir_str else default_dir
    custom_conversion_path = Path(conversion_dir_str).resolve() if conversion_dir_str else default_dir
    return custom_configs_path, custom_conversion_path


def _select_optimizer_class(cfg: DictConfig) -> tuple[type, str, Literal["bayes", "optuna"]]:
    """Select the configured optimizer class."""
    if cfg.optimizer.get("bayes", False):
        try:
            from sd_optim.optimizers.bayes.optimizer import BayesOptimizer
        except ModuleNotFoundError as error:
            if getattr(error, "name", "") == "bayes_opt":
                logger.error(
                    "Bayes optimizer selected, but dependency 'bayesian-optimization' is not installed. "
                    "Install the Bayes extra before running with optimizer.bayes=true."
                )
                raise SystemExit(1) from error
            raise
        return BayesOptimizer, "BayesOpt", "bayes"

    if cfg.optimizer.get("optuna", False):
        from sd_optim.optimizers.optuna.optimizer import OptunaOptimizer

        return OptunaOptimizer, "Optuna", "optuna"

    possible_opts = [key for key, value in cfg.optimizer.items() if isinstance(value, bool)]
    logger.error("No optimizer selected! Please set one of %s to True in config.yaml under 'optimizer'.", possible_opts)
    raise SystemExit(1)


def _has_optuna_results(optim_instance: object) -> bool:
    study = getattr(optim_instance, "study", None)
    return bool(study and getattr(study, "trials", None))


def _has_bayes_results(optim_instance: object) -> bool:
    optimizer = getattr(optim_instance, "optimizer", None)
    return bool(optimizer and hasattr(optimizer, "res") and optimizer.res)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for the sd-optim application."""

    # --- Initial Logging & Config Display ---
    logger.info("==================================================")
    logger.info("             Starting sd-optim v1.x             ")
    logger.info("==================================================")
    try:
        # run_dir = Path(os.getcwd()) # Hydra sets CWD to the output directory
        # logger.info(f"Hydra Run Directory: {run_dir}")
        # Log the entire config using OmegaConf for better readability (optional, consider DEBUG level)
        # logger.debug(f"Full configuration:\n{OmegaConf.to_yaml(cfg)}")
        logger.info("Selected WebUI: %s", cfg.get("webui", "N/A"))
        logger.info("Optimization Mode: %s", cfg.get("optimization_mode", "N/A"))
        if cfg.get("optimization_mode") == "merge":
            logger.info("Merge Method: %s", cfg.get("merge_method", "N/A"))
    except Exception as log_cfg_e:
        logger.warning("Could not log initial config details: %s", log_cfg_e)

    # --- Determine Project Root and Custom Extension Paths ---
    try:
        custom_configs_path, custom_conversion_path = _determine_extension_paths(cfg)

        logger.info(
            "Using custom configs directory: %s %s",
            custom_configs_path,
            "(Default)" if not cfg.get("configs_dir") else "(User Specified)",
        )
        logger.info(
            "Using custom conversion directory: %s %s",
            custom_conversion_path,
            "(Default)" if not cfg.get("conversion_dir") else "(User Specified)",
        )

    except Exception as path_e:
        logger.error(
            f"CRITICAL ERROR determining custom extension paths: {path_e}",
            exc_info=True,
        )
        logger.error("Ensure sd_optim.py is in the project root or adjust path logic.")
        sys.exit(1)

    # --- Load Custom Configs FIRST ---
    # This registers the config IDs (like "sdxl-optim_blocks")
    try:
        logger.info("--- Loading Custom ModelConfigs ---")
        load_and_register_custom_configs(custom_configs_path)
    except Exception as config_load_e:
        logger.error(f"CRITICAL ERROR loading custom configs: {config_load_e}", exc_info=True)
        logger.error("Halting execution due to config loading failure.")
        sys.exit(1)

    # --- Load Custom Converters/Methods SECOND ---
    # Importing modules here triggers the @sd_mecha.merge_method decorators inside them,
    # which require the config IDs registered in the previous step to be valid.
    try:
        logger.info("--- Loading Custom Converters/MergeMethods ---")
        load_and_register_custom_conversion(custom_conversion_path)
    except Exception as converter_load_e:
        logger.error(
            f"CRITICAL ERROR loading/registering custom converters: {converter_load_e}",
            exc_info=True,
        )
        logger.error("Halting execution due to converter loading failure.")
        sys.exit(1)

    # --- Select Optimizer Class ---
    logger.info("--- Selecting Optimizer ---")
    optimizer_class, optimizer_name, optimizer_kind = _select_optimizer_class(cfg)
    logger.info(f"Using Optimizer: {optimizer_name}")

    # --- Initialize and Run Optimizer ---
    optim_instance = None
    dashboard_process = None
    try:
        logger.info(f"--- Initializing {optimizer_name} ---")
        optim_instance = optimizer_class(cfg)

        logger.info("Validating optimizer configuration...")
        if not optim_instance.validate_optimizer_config():
            logger.error(f"Invalid configuration for {optimizer_name}.")
            sys.exit(1)
        logger.info("Optimizer configuration validated.")

        # --- Launch Dashboard BEFORE Optimization ---
        if optimizer_kind == "optuna" and cfg.optimizer.optuna_config.get("launch_dashboard", False):
            dashboard_port = cfg.optimizer.optuna_config.get("dashboard_port", 8080)
            logger.info(f"--- Attempting to launch Optuna Dashboard in background (Port: {dashboard_port}) ---")
            dashboard_process = optim_instance.start_dashboard_background(port=dashboard_port)
            if dashboard_process is None:
                logger.warning("Failed to start dashboard process. Continuing without background dashboard.")
            else:
                logger.info("Background dashboard process launch initiated.")

        init_points = cfg.optimizer.get("init_points", 0)
        n_iters = cfg.optimizer.get("n_iters", 0)
        logger.info(f"--- Starting Optimization Loop ({init_points} init + {n_iters} iters = {init_points + n_iters} total) ---")

        # Run the main optimization loop
        asyncio.run(optim_instance.optimize())

        # --- Postprocessing after NORMAL completion ---
        # This is now handled by the finally block to ensure it runs even after interrupts/errors
        # logger.info("--- Optimization Finished: Running Postprocessing ---")
        # asyncio.run(optim_instance.postprocess()) # <<< COMMENTED OUT / REMOVED

    except KeyboardInterrupt:
        logger.info("\n--- Optimization interrupted by user (Ctrl+C) ---")
        # Let finally block handle postprocessing attempt

    except ValueError as val_err:
        logger.error(f"Configuration or Setup Error: {val_err}", exc_info=True)
        logger.error("Halting execution.")
        # Let finally block handle postprocessing attempt (if instance exists)
        # sys.exit(1) # Consider if you truly want to exit *before* finally

    except Exception:
        logger.error("--- An Unexpected Error Occurred During Optimization ---", exc_info=True)
        # Let finally block handle postprocessing attempt

    finally:
        # --- ADDED: Attempt Postprocessing ---
        logger.info("--- Attempting Postprocessing (Finally Block) ---")
        if optim_instance is not None:
            if optimizer_kind in {"optuna", "bayes"}:
                try:
                    should_run_postprocess = False
                    if optimizer_kind == "optuna":
                        should_run_postprocess = _has_optuna_results(optim_instance)
                        if not should_run_postprocess:
                            logger.warning("Optuna study has no trials, skipping postprocessing.")
                    elif optimizer_kind == "bayes":
                        should_run_postprocess = _has_bayes_results(optim_instance)
                        if not should_run_postprocess:
                            logger.warning("Bayes optimizer has no results, skipping postprocessing.")

                    if should_run_postprocess:
                        logger.info("Running postprocess for %s...", optimizer_name)
                        asyncio.run(optim_instance.postprocess())
                        logger.info("Postprocessing for %s finished.", optimizer_name)
                    else:
                        logger.info("No results found for postprocessing.")

                except Exception as e_post:
                    logger.error(
                        f"Error during postprocessing in finally block: {e_post}",
                        exc_info=True,
                    )
            else:
                logger.info("Optimizer type does not require specific postprocessing visuals.")
        else:
            logger.warning("Optimizer instance was not created, cannot run postprocessing.")
        # --- END Added Postprocessing Section ---

        # --- Dashboard Termination (Improved) ---
        if dashboard_process is not None:
            logger.info(f"Attempting to terminate background dashboard process (PID: {dashboard_process.pid}) launched by this run...")
            try:
                # Check if process hasn't already finished using poll()
                if dashboard_process.poll() is None:
                    dashboard_process.terminate()  # SIGTERM first
                    try:
                        dashboard_process.wait(timeout=3)  # Wait briefly
                        logger.info(f"Dashboard process terminated gracefully with code: {dashboard_process.returncode}")
                    except subprocess.TimeoutExpired:
                        logger.warning("Dashboard process did not terminate after 3s, sending kill signal (SIGKILL).")
                        dashboard_process.kill()  # Force kill
                        dashboard_process.wait()  # Wait for kill
                        logger.info("Dashboard process killed.")
                else:
                    # Log if it already finished before finally block reached it
                    logger.info(f"Dashboard process already exited before termination attempt with code: {dashboard_process.returncode}")
            except Exception as e_term:
                # Catch errors during terminate/wait/kill
                logger.error(
                    f"Error during dashboard process termination: {e_term}",
                    exc_info=True,
                )
        else:
            # Log if no dashboard was launched by this specific run
            logger.info("No dashboard process was launched by this run to terminate.")
        # --- End Dashboard Termination ---

        logger.info("==================================================")
        logger.info("              sd-optim run finished.              ")
        logger.info("==================================================")
        logging.shutdown()


if __name__ == "__main__":
    main()
