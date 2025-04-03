# sd_optim.py (Formerly bayesian_merger.py) - Version 1.0
import sys

import hydra
import asyncio # Import asyncio
import logging # Import logging

from omegaconf import DictConfig, OmegaConf
from sd_optim import BayesOptimizer, OptunaOptimizer # Import OptunaOptimizer
# Remove TPE/ATPE imports if they are not implemented/deprecated
# from sd_optim import TPEOptimizer, ATPEOptimizer
from sd_optim import utils # Import utils

# Setup logger (consider moving to a dedicated logging config if complex)
# Use basicConfig for simple setup, or configure via Hydra later
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Setup logger for this module

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # --- Initial Logging ---
    # Use Hydra's logger if available, otherwise fallback to root logger
    hydra_logger = logging.getLogger(__name__) # Hydra usually sets up loggers
    hydra_logger.info("Starting sd-optim...")
    # Log the entire config (can be verbose, consider logging specific parts)
    # Use OmegaConf.to_yaml for cleaner logging of the config object
    try:
        hydra_logger.debug(f"Full configuration:\n{OmegaConf.to_yaml(cfg)}")
    except Exception as log_e:
         hydra_logger.warning(f"Could not log full configuration: {log_e}")
    hydra_logger.info(f"Selected WebUI: {cfg.webui}")
    hydra_logger.info(f"Optimization Mode: {cfg.optimization_mode}")
    hydra_logger.info(f"Merge Method (if applicable): {cfg.get('merge_method', 'N/A')}")

    # --- Call Setup Utilities FIRST ---
    # This needs to run before ParameterHandler or other classes that might need
    # the dynamically registered configs/methods.
    try:
        # No need to check if section exists, setup_custom_blocks handles that internally
        hydra_logger.info("Attempting custom block setup (if defined in config)...")
        utils.setup_custom_blocks(cfg) # Call setup utility early
        hydra_logger.info("Custom block setup process completed (or skipped if not defined).")
    except Exception as setup_e:
         # Catch specific expected errors from setup is better if possible
         hydra_logger.error(f"CRITICAL ERROR during custom block setup: {setup_e}", exc_info=True)
         hydra_logger.error("Halting execution due to custom block setup failure.")
         sys.exit(1) # Use sys.exit for cleaner exit

    # --- Select Optimizer Class ---
    optimizer_class = None
    # Use .get() with default False for safety
    if cfg.optimizer.get("bayes", False):
        optimizer_class = BayesOptimizer
        hydra_logger.info("Using BayesOptimizer.")
    elif cfg.optimizer.get("optuna", False):
        optimizer_class = OptunaOptimizer
        hydra_logger.info("Using OptunaOptimizer.")
    # Remove TPE/ATPE or add checks if they are implemented
    # elif cfg.optimizer.get("tpe", False):
    #     optimizer_class = TPEOptimizer
    # elif cfg.optimizer.get("atpe", False):
    #     optimizer_class = ATPEOptimizer
    else:
        # Log error and exit if no valid optimizer is selected
        # Find valid keys dynamically
        valid_optimizers = [k for k in cfg.optimizer.keys() if cfg.optimizer.get(k) is True and k in ['bayes', 'optuna']] # Adjust as needed
        if not valid_optimizers:
             all_opts = [k for k in cfg.optimizer.keys() if k not in ['random_state', 'init_points', 'n_iters', 'load_log_file', 'reset_log_file', 'acquisition_function', 'bounds_transformer', 'sampler', 'use_pruning', 'pruner_type', 'checkpoint_dir', 'checkpoint_interval', 'early_stopping', 'patience', 'min_improvement', 'n_jobs', 'launch_dashboard', 'dashboard_port']] # Filter out common settings
             hydra_logger.error(f"No valid optimizer selected in configuration: {cfg.optimizer}")
             hydra_logger.error(f"Please set one of {all_opts} to True in config.yaml under the 'optimizer' section.")
             sys.exit(1) # Exit gracefully
        # If multiple are True, maybe log a warning and pick the first?
        # For now, assumes only one is True based on current structure.

    # --- Initialize and Run Optimizer ---
    try:
        hydra_logger.info(f"Initializing optimizer: {optimizer_class.__name__}")
        optim_instance = optimizer_class(cfg) # Initialize the chosen optimizer

        # Validate config specific to the chosen optimizer
        if not optim_instance.validate_optimizer_config():
             hydra_logger.error(f"Invalid configuration for {optimizer_class.__name__}. Please check settings in config.yaml under the 'optimizer' section.")
             sys.exit(1)

        hydra_logger.info(f"Running optimization loop with {optimizer_class.__name__}...")
        # Run the main async optimization loop using asyncio.run
        asyncio.run(optim_instance.optimize())

        hydra_logger.info("Optimization finished. Running postprocessing...")
        # Run async postprocessing
        asyncio.run(optim_instance.postprocess())

        # Optional: Launch Optuna dashboard if using Optuna and configured
        if isinstance(optim_instance, OptunaOptimizer) and cfg.optimizer.get("launch_dashboard", False):
            dashboard_port = cfg.optimizer.get("dashboard_port", 8080) # Use default if not set
            hydra_logger.info(f"Launching Optuna dashboard on port {dashboard_port}...")
            # This call might block, consider if it should run async or be optional
            optim_instance.launch_dashboard(port=dashboard_port)

    except KeyboardInterrupt:
         hydra_logger.info("Optimization process interrupted by user (KeyboardInterrupt).")
         # Perform any necessary cleanup here if possible
    except Exception as e:
        hydra_logger.error(f"An unexpected error occurred during the optimization process: {e}", exc_info=True)
        # Potentially add more specific error handling or cleanup here
    finally:
         hydra_logger.info("sd-optim run finished.")
         # Ensure logs are flushed, etc.
         logging.shutdown()


if __name__ == "__main__":
    # This block runs when the script is executed directly
    main()