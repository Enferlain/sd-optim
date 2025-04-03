# sd_optim.py - Version 1.3 - Modular loading & Configurable extension paths

import hydra
import asyncio
import logging
import sys
import os
from pathlib import Path

# Import main config/utility helpers
from omegaconf import DictConfig, OmegaConf # Using OmegaConf for cleaner config logging
from sd_optim import utils # Import utils (needs to exist)
from sd_optim import BayesOptimizer, OptunaOptimizer


# --- Basic Logging Setup ---
# Configure logging level and format early. Can be overridden by Hydra later.
logging.basicConfig(
    level=logging.INFO, # Default level
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# Use a logger specific to this main script
logger = logging.getLogger(__name__) # Hydra often configures this further


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for the sd-optim application."""

    # --- Initial Logging & Config Display ---
    logger.info("==================================================")
    logger.info("             Starting sd-optim v1.x             ")
    logger.info("==================================================")
    try:
        run_dir = Path(os.getcwd()) # Hydra sets CWD to the output directory
        logger.info(f"Hydra Run Directory: {run_dir}")
        # Log the entire config using OmegaConf for better readability (optional, consider DEBUG level)
        # logger.debug(f"Full configuration:\n{OmegaConf.to_yaml(cfg)}")
        logger.info(f"Selected WebUI: {cfg.get('webui', 'N/A')}")
        logger.info(f"Optimization Mode: {cfg.get('optimization_mode', 'N/A')}")
        if cfg.get('optimization_mode') == 'merge':
            logger.info(f"Merge Method: {cfg.get('merge_method', 'N/A')}")
    except Exception as log_cfg_e:
         logger.warning(f"Could not log initial config details: {log_cfg_e}")

    # --- Determine Project Root and Custom Extension Paths ---
    try:
        # Assuming sd_optim.py is located at the root of the project checkout
        project_root = Path(__file__).parent.resolve()
        logger.debug(f"Determined project root: {project_root}")

        # Define default paths relative to the sd_optim package within the project
        # Assumes structure: project_root/sd_optim/custom_configs, etc.
        default_configs_dir = project_root / "sd_optim" / "model_configs"
        default_conversion_dir = project_root / "sd_optim" / "model_configs"

        # Get paths from config, falling back to defaults if null or missing
        configs_dir_str = cfg.get('configs_dir') # Returns None if key is missing/null
        conversion_dir_str = cfg.get('conversion_dir')

        # Resolve paths: Use config path if specified, otherwise use default. Convert to absolute Path.
        custom_configs_path = Path(configs_dir_str).resolve() if configs_dir_str else default_configs_dir
        custom_conversion_path = Path(conversion_dir_str).resolve() if conversion_dir_str else default_conversion_dir

        logger.info(f"Using custom configs directory: {custom_configs_path} {'(Default)' if not configs_dir_str else '(User Specified)'}")
        logger.info(f"Using custom conversion directory: {custom_conversion_path} {'(Default)' if not conversion_dir_str else '(User Specified)'}")

    except Exception as path_e:
         logger.error(f"CRITICAL ERROR determining custom extension paths: {path_e}", exc_info=True)
         logger.error("Ensure sd_optim.py is in the project root or adjust path logic.")
         sys.exit(1)

    # --- Load Custom Configs FIRST ---
    # This registers the config IDs (like "sdxl-optim_blocks")
    try:
        logger.info("--- Loading Custom ModelConfigs ---")
        utils.load_and_register_custom_configs(custom_configs_path)
    except Exception as config_load_e:
        logger.error(f"CRITICAL ERROR loading custom configs: {config_load_e}", exc_info=True)
        logger.error("Halting execution due to config loading failure.")
        sys.exit(1)

    # --- Load Custom Converters/Methods SECOND ---
    # Importing modules here triggers the @sd_mecha.merge_method decorators inside them,
    # which require the config IDs registered in the previous step to be valid.
    try:
        logger.info("--- Loading Custom Converters/MergeMethods ---")
        utils.load_and_register_custom_conversion(custom_conversion_path)
    except Exception as converter_load_e:
        logger.error(f"CRITICAL ERROR loading/registering custom converters: {converter_load_e}", exc_info=True)
        logger.error("Halting execution due to converter loading failure.")
        sys.exit(1)

    # --- Select Optimizer Class ---
    logger.info("--- Selecting Optimizer ---")
    optimizer_class = None
    optimizer_name = "N/A"
    # Access optimizer selection flags safely
    if cfg.optimizer.get("bayes", False):
        optimizer_class = BayesOptimizer
        optimizer_name = "BayesOpt"
    elif cfg.optimizer.get("optuna", False):
        optimizer_class = OptunaOptimizer
        optimizer_name = "Optuna"
    # Add elif for other optimizers if re-implemented (e.g., TPE, ATPE)

    if optimizer_class is None:
        # Try to list available boolean flags under optimizer section
        possible_opts = [k for k, v in cfg.optimizer.items() if isinstance(v, bool)]
        logger.error(f"No optimizer selected! Please set one of {possible_opts} to True in config.yaml under 'optimizer'.")
        sys.exit(1)
    logger.info(f"Using Optimizer: {optimizer_name}")

    # --- Initialize and Run Optimizer ---
    optim_instance = None # Initialize for finally block
    try:
        logger.info(f"--- Initializing {optimizer_name} ---")
        # Initialization order: Optimizer -> ParameterHandler -> (loads configs/method)
        # This is safe now because custom configs/converters were loaded above.
        optim_instance = optimizer_class(cfg)

        logger.info("Validating optimizer configuration...")
        if not optim_instance.validate_optimizer_config():
             # Specific error message should be logged by validate_optimizer_config
             logger.error(f"Invalid configuration for {optimizer_name}. Please check config.yaml.")
             sys.exit(1)
        logger.info("Optimizer configuration validated.")

        # Log planned iterations from config
        init_points = cfg.optimizer.get('init_points', 0)
        n_iters = cfg.optimizer.get('n_iters', 0)
        logger.info(f"--- Starting Optimization Loop ({init_points} init + {n_iters} iters = {init_points + n_iters} total) ---")

        # Run the main async optimization loop using asyncio.run
        asyncio.run(optim_instance.optimize())

        logger.info("--- Optimization Finished: Running Postprocessing ---")
        # Run async postprocessing
        asyncio.run(optim_instance.postprocess())

        # --- Optional: Optuna Dashboard ---
        # Check if optuna is the selected optimizer AND launch flag is True
        if isinstance(optim_instance, OptunaOptimizer) and cfg.optimizer.optuna_config.get("launch_dashboard", False):
            dashboard_port = cfg.optimizer.optuna_config.get("dashboard_port", 8080) # Use default port if not set
            logger.info(f"--- Launching Optuna Dashboard (Access: http://localhost:{dashboard_port}) ---")
            logger.info("Press Ctrl+C in this terminal to stop the dashboard.")
            # This call blocks, run it last.
            optim_instance.launch_dashboard(port=dashboard_port)

    except KeyboardInterrupt:
        logger.info("\n--- Optimization interrupted by user (Ctrl+C) ---")
        if optim_instance and hasattr(optim_instance, 'save_checkpoint'):  # <<< Check for public name
            logger.info("Attempting to save final checkpoint...")
            try:
                optim_instance.save_checkpoint()  # <<< Call public name
                logger.info("Checkpoint state saved.")
            except Exception as chkpt_e:
                logger.error(f"Failed to save checkpoint during interrupt: {chkpt_e}")

    except ValueError as val_err:
         # Catch specific configuration or setup errors we raised intentionally
         logger.error(f"Configuration or Setup Error: {val_err}", exc_info=True) # Show traceback for ValueErrors too
         logger.error("Halting execution.")
         sys.exit(1)
    except Exception as e:
        # Catch any other unexpected errors during the main process
        logger.error("--- An Unexpected Error Occurred During Optimization ---", exc_info=True)
    finally:
         # This block always runs, even if errors occur
         logger.info("==================================================")
         logger.info("              sd-optim run finished.              ")
         logger.info("==================================================")
         logging.shutdown() # Ensure all logs are flushed


if __name__ == "__main__":
    # This block runs when the script is executed directly: python sd_optim.py
    # Hydra takes over from here by parsing command line args and calling main()
    main()