# sd_optim.py (Formerly bayesian_merger.py) - Version 1.0

import hydra
import asyncio # Import asyncio
import logging # Import logging

from omegaconf import DictConfig
from sd_optim import BayesOptimizer, OptunaOptimizer # Import OptunaOptimizer
# Remove TPE/ATPE imports if they are not implemented/deprecated
# from sd_optim import TPEOptimizer, ATPEOptimizer
from sd_optim import utils # Import utils

logger = logging.getLogger(__name__) # Setup logger

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info("Starting sd-optim...")
    logger.info(f"Loaded configuration: {cfg}") # Log loaded config

    # --- Call Setup Utilities ---
    custom_blocks_defined = cfg.optimization_guide.get("custom_block_configs") is not None # Check if section exists
    if custom_blocks_defined:
        logger.info("Found custom block definitions, attempting setup...")
        try:
            # Assuming setup_custom_blocks is defined in utils and takes cfg
            utils.setup_custom_blocks(cfg) # Call setup utility
            logger.info("Custom block configurations processed and registered successfully.")
        except Exception as e:
             # Catching specific expected errors from setup is better if possible
             logger.error(f"CRITICAL ERROR during custom block setup: {e}", exc_info=True)
             logger.error("Halting execution due to custom block setup failure.")
             exit(1) # Exit only on actual setup errors
    else:
        logger.info("No custom block definitions found in configuration, skipping setup.")

    # --- Select Optimizer Class ---
    optimizer_class = None
    if cfg.optimizer.get("bayes", False): # Use .get for safety
        optimizer_class = BayesOptimizer
        logger.info("Using BayesOptimizer.")
    elif cfg.optimizer.get("optuna", False): # Use .get for safety
        optimizer_class = OptunaOptimizer
        logger.info("Using OptunaOptimizer.")
    # Remove TPE/ATPE or add checks if they are implemented
    # elif cfg.optimizer.get("tpe", False):
    #     optimizer_class = TPEOptimizer
    # elif cfg.optimizer.get("atpe", False):
    #     optimizer_class = ATPEOptimizer
    else:
        # Log error and exit if no valid optimizer is selected
        valid_optimizers = [k for k in cfg.optimizer.keys() if k in ['bayes', 'optuna']] # Adjust as needed
        logger.error(f"No valid optimizer selected in configuration: {cfg.optimizer}")
        logger.error(f"Please set one of {valid_optimizers} to True in config.yaml.")
        exit(1) # Exit gracefully

    # --- Initialize and Run Optimizer ---
    try:
        optim_instance = optimizer_class(cfg)

        # Validate config specific to the chosen optimizer
        if not optim_instance.validate_optimizer_config():
             logger.error(f"Invalid configuration for {optimizer_class.__name__}. Please check settings.")
             exit(1)

        logger.info(f"Running optimization with {optimizer_class.__name__}...")
        # Run the main async optimization loop
        asyncio.run(optim_instance.optimize())

        logger.info("Optimization finished. Running postprocessing...")
        # Run async postprocessing
        asyncio.run(optim_instance.postprocess())

        # Optional: Launch Optuna dashboard if using Optuna
        if isinstance(optim_instance, OptunaOptimizer) and cfg.optimizer.get("launch_dashboard", False):
            optim_instance.launch_dashboard(port=cfg.optimizer.get("dashboard_port", 8080))

    except Exception as e:
        logger.error(f"An error occurred during the optimization process: {e}", exc_info=True)
        # Potentially add more specific error handling or cleanup here
    finally:
         logger.info("sd-optim run finished.")


if __name__ == "__main__":
    main()
