# bayes_optimizer.py - Version 1.2

import os
import random
import logging
import json
import asyncio
import time # Import time
import pickle # Import pickle

from typing import Dict, List, Any, Optional  # Import Any
from pathlib import Path
from bayes_opt import BayesianOptimization, Events, UtilityFunction
from bayes_opt.logger import JSONLogger
from bayes_opt.domain_reduction import SequentialDomainReductionTransformer
from hydra.core.hydra_config import HydraConfig
from scipy.stats import qmc

from sd_optim.artist import Artist
from sd_optim.optimizer import Optimizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class BayesOptimizer(Optimizer):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.artist = Artist(self)
        self.setup_logging()
        self.optimizer: Optional[BayesianOptimization] = None # Initialize optimizer attribute
        self.optimization_start_time: Optional[float] = None # Track start time

        # Checkpoint directory
        self.checkpoint_dir = Path(self.cfg.optimizer.get("checkpoint_dir", os.getcwd())) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_interval = self.cfg.optimizer.get("checkpoint_interval", 10)

    def setup_logging(self) -> None:
        """Initialize Bayesian optimization specific logging"""
        run_name = self.cfg.get("log_name", "default_bayes_run") # Use cfg for log_name
        self.log_name = run_name
        self.log_file_path = Path(HydraConfig.get().runtime.output_dir, f"{self.log_name}.json")

        # Initialize with empty list - will be populated if loading previous data
        self.previous_iterations = []

        # First create a fresh logger
        self.json_logger = JSONLogger(path=str(self.log_file_path), reset=self.cfg.optimizer.reset_log_file) # Renamed to json_logger

        # Load previous data if specified
        load_log_path = self.cfg.optimizer.get("load_log_file")
        if load_log_path:
            load_log_file = Path(load_log_path)
            if load_log_file.is_file():
                try:
                    with open(load_log_file, "r") as f:
                        self.previous_iterations = [json.loads(line) for line in f]

                    # Write previous data to new log file if resetting is false
                    if not self.cfg.optimizer.reset_log_file:
                         with open(self.log_file_path, "w") as f:
                            for iteration_data in self.previous_iterations:
                                f.write(json.dumps(iteration_data) + "\n")
                         logger.info(
                            f"Loaded and transferred {len(self.previous_iterations)} iterations from {load_log_file}")
                    else:
                        logger.info(f"Loaded {len(self.previous_iterations)} iterations from {load_log_file} but resetting log file.")

                except Exception as e:
                    logger.warning(f"Failed to load previous optimization data from {load_log_file}: {e}")
            else:
                logger.info(f"No previous log file found at {load_log_file}")

    async def optimize(self) -> None: # Changed to async
        self.optimization_start_time = time.time()
        logger.debug(f"Initial Parameter Bounds: {self.optimizer_pbounds}")  # Use the attribute directly

        # --- Acquisition Function Configuration ---
        acq_config = self.cfg.optimizer.get("acquisition_function", {})
        acquisition_function = UtilityFunction(
            kind=acq_config.get("kind", "ucb"),
            kappa=acq_config.get("kappa", 3.0),
            xi=acq_config.get("xi", 0.05),
            kappa_decay=acq_config.get("kappa_decay", 0.98),
            kappa_decay_delay=acq_config.get("kappa_decay_delay", self.cfg.optimizer.init_points)
        )

        # --- Bounds Transformer Configuration ---
        bt_config = self.cfg.optimizer.get("bounds_transformer", {})
        bounds_transformer_instance = SequentialDomainReductionTransformer(
            gamma_osc=bt_config.get("gamma_osc", 0.65),
            gamma_pan=bt_config.get("gamma_pan", 0.9),
            eta=bt_config.get("eta", 0.83),
            minimum_window=bt_config.get("minimum_window", 0.15),
        )
        bounds_transformer_enabled = self.cfg.optimizer.get("bounds_transformer_enabled", False) # Use explicit enable flag

        # --- Synchronous Wrapper for Async Target Function ---
        def sync_target_function_wrapper(**params_dict):
            # bayes_opt passes params as keyword arguments, convert to dict
            # Run the async target function in a synchronous context
            try:
                result = asyncio.run(self.sd_target_function(params_dict))
                return result
            except Exception as e:
                logger.error(f"Error in target function execution: {e}", exc_info=True)
                return -float('inf') # Return very negative score on error

        # --- Initialize BayesianOptimization ---
        self.optimizer = BayesianOptimization(
            f=sync_target_function_wrapper, # Use the synchronous wrapper
            pbounds=self.optimizer_pbounds,
            random_state=self.cfg.optimizer.random_state,
            bounds_transformer=bounds_transformer_instance if bounds_transformer_enabled else None,
            # verbose=2 # Optional: set verbosity level
        )

        # --- Load Previous Points ---
        if self.previous_iterations:
            try:
                loaded_count = 0
                for point in self.previous_iterations:
                    # Register points with the optimizer
                    # Check if the parameters are already registered to avoid duplicates
                    # Note: bayes_opt doesn't have a direct way to check, so we rely on its internal handling or skip if reset=True
                    if not self.cfg.optimizer.reset_log_file or not self.optimizer.space.params_registered(point["params"]):
                        self.optimizer.register(
                            params=point["params"],
                            target=point["target"]
                        )
                        loaded_count +=1
                logger.info(f"Registered {loaded_count} unique previous points with the optimizer")
            except Exception as e:
                logger.warning(f"Failed to register previous points with optimizer: {e}")

        # --- Subscribe Logger ---
        # Subscribe json_logger to capture new points
        self.optimizer.subscribe(Events.OPTIMIZATION_STEP, self.json_logger)

        # Subscribe checkpoint callback
        def checkpoint_subscriber(event, instance):
            iteration = len(instance.res)
            if iteration % self.checkpoint_interval == 0 and iteration > 0:
                logger.info(f"Creating periodic checkpoint at iteration {iteration}")
                self.save_checkpoint(instance)
        self.optimizer.subscribe(Events.OPTIMIZATION_STEP, checkpoint_subscriber)

        # --- Initial Sampling (Quasi-Random) ---
        init_points = self.cfg.optimizer.init_points
        completed_trials = len(self.optimizer.res) # Get count of already registered/completed trials
        remaining_init_points = max(0, init_points - completed_trials)

        if remaining_init_points > 0:
            sampler_type = self.cfg.optimizer.get("sampler", "random").lower()
            # Separate continuous and categorical/binary for sampling
            continuous_bounds = {k: v for k, v in pbounds.items() if not (isinstance(v, tuple) and v in [(0.0, 1.0), (1.0, 0.0)])}
            categorical_params = {k: v for k, v in pbounds.items() if (isinstance(v, tuple) and v in [(0.0, 1.0), (1.0, 0.0)])}

            if sampler_type != "random" and continuous_bounds:
                n_samples = remaining_init_points
                d = len(continuous_bounds)
                try:
                    if sampler_type == "latin_hypercube":
                        sampler = qmc.LatinHypercube(d=d, seed=self.cfg.optimizer.random_state)
                    elif sampler_type == "sobol":
                        sampler = qmc.Sobol(d=d, seed=self.cfg.optimizer.random_state)
                    elif sampler_type == "halton":
                        sampler = qmc.Halton(d=d, seed=self.cfg.optimizer.random_state)
                    else:
                        logger.warning(f"Unknown sampler type '{sampler_type}', falling back to random sampling for init points")
                        sampler_type = "random"

                    if sampler_type != "random":
                        continuous_samples = sampler.random(n_samples)
                        l_bounds = [b[0] for b in continuous_bounds.values()]
                        u_bounds = [b[1] for b in continuous_bounds.values()]
                        scaled_continuous = qmc.scale(continuous_samples, l_bounds, u_bounds)
                        continuous_param_names = list(continuous_bounds.keys())

                        logger.info(f"Probing {n_samples} initial points using {sampler_type} sampler...")
                        for i in range(n_samples):
                            params = {}
                            # Add continuous parameters
                            continuous_values = scaled_continuous[i]
                            for name, value in zip(continuous_param_names, continuous_values):
                                params[name] = value
                            # Add categorical/binary parameters randomly
                            for name, bound in categorical_params.items():
                                params[name] = random.choice([bound[0], bound[1]]) # Choose 0 or 1

                            self.optimizer.probe(params=params, lazy=True)
                        remaining_init_points = 0 # All init points are probed

                except Exception as e:
                    logger.error(f"Error during initial sampling: {e}. Falling back to random.", exc_info=True)
                    remaining_init_points = n_samples # Revert if sampling failed


        # --- Run Optimization ---
        total_iterations = self.cfg.optimizer.n_iters
        remaining_iterations = max(0, total_iterations - max(0, completed_trials - init_points))

        if remaining_iterations > 0 or remaining_init_points > 0:
            logger.info(f"Starting optimization: {remaining_init_points} random init points, {remaining_iterations} optimization iterations.")
            try:
                # Run the synchronous maximize method in a separate thread
                await asyncio.to_thread(
                    self.optimizer.maximize,
                    init_points=remaining_init_points, # Use remaining random points
                    n_iter=remaining_iterations,
                    acquisition_function=acquisition_function,
                    # Other maximize parameters if needed
                ) # acq is deprecated
            except KeyboardInterrupt:
                logger.info("Optimization interrupted by user. Saving current state...")
                self.save_checkpoint(self.optimizer) # Pass optimizer instance
            except Exception as e:
                logger.error(f"Optimization failed: {e}", exc_info=True)
                self.save_checkpoint(self.optimizer) # Pass optimizer instance
                raise
        else:
            logger.info("All optimization iterations already completed. Skipping maximize.")

        # Save final checkpoint
        self.save_checkpoint(self.optimizer)


    def save_checkpoint(self, optimizer_instance: BayesianOptimization):
        """Save current optimization state using pickle."""
        if not optimizer_instance:
            logger.warning("No optimizer instance to checkpoint")
            return

        checkpoint_file = self.checkpoint_dir / f"bayesopt_checkpoint_{self.log_name}.pkl"
        try:
            # Save the optimizer's state (includes space, results)
            with open(checkpoint_file, "wb") as f:
                pickle.dump(optimizer_instance, f)
            logger.info(f"Saved Bayesian Optimization checkpoint to {checkpoint_file}")
        except Exception as e:
            logger.error(f"Failed to save Bayesian Optimization checkpoint: {e}", exc_info=True)


    async def postprocess(self) -> None: # Keep async
        logger.info("\nBayesOpt Recap!")

        # --- Step 1: Check if results exist ---
        if not self.optimizer or not hasattr(self.optimizer, 'res') or not self.optimizer.res:
            logger.warning("No Bayesian Optimization results found to process or display.")
            return # Exit early if no results

        # --- Step 2: Log statistics from existing results ---
        results = self.optimizer.res # Get the results list
        total_trials = len(results)
        logger.info(f"Total Trials Run: {total_trials}")

        if self.optimization_start_time:
             total_runtime = time.time() - self.optimization_start_time
             hours, remainder = divmod(total_runtime, 3600)
             minutes, seconds = divmod(remainder, 60)
             logger.info(f"Total runtime: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
             if total_trials > 0:
                 logger.info(f"Average time per trial: {total_runtime/total_trials:.2f} seconds")

        # Log top trials
        logger.info("\nTop 5 BayesOpt Trials:")
        # Sort the results list directly
        sorted_results = sorted(results, key=lambda x: x["target"], reverse=True)
        for i, res in enumerate(sorted_results[:5]):
            logger.info(f"Rank {i + 1}:")
            logger.info(f"\tTarget: {res['target']:.4f}")
            # Ensure params are logged, might need careful formatting if complex
            param_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in res['params'].items())
            logger.info(f"\tParams: {{{param_str}}}")

        # Log best trial
        if hasattr(self.optimizer, 'max') and self.optimizer.max:
            best_trial = self.optimizer.max
            logger.info("\nBest BayesOpt Trial Found:")
            logger.info(f"\tTarget: {best_trial['target']:.4f}")
            param_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in best_trial['params'].items())
            logger.info(f"\tParams: {{{param_str}}}")
        else:
             logger.warning("Could not determine the best trial from optimizer results.")

        # --- Step 3: Prepare data for Artist plot ---
        iterations = list(range(1, total_trials + 1)) # Simple trial numbers
        scores = [res["target"] for res in results]   # Extract scores

        # Calculate rolling best scores from the results
        best_scores = []
        current_best = -float('inf')
        for res in results: # Iterate through results in the order they happened
            current_best = max(current_best, res["target"])
            best_scores.append(current_best)

        # --- Step 4: Populate the Artist instance ---
        # Check if artist was initialized (should be in BayesOpt __post_init__)
        if not hasattr(self, 'artist'):
             logger.error("Artist object not found on BayesOptimizer. Cannot generate plot.")
             return

        self.artist.iterations = iterations
        self.artist.scores = scores
        self.artist.best_scores = best_scores
        # Optionally populate parameters if needed by future artist plots
        # self.artist.parameters = [res["params"] for res in results]

        # --- Step 5: Generate Visualization using Artist ---
        logger.info("Generating convergence plot via Artist...")
        try:
            # Since visualize_optimization now calls plot_convergence which is async
            await self.artist.visualize_optimization()
        except Exception as e:
             logger.error(f"Failed to create visualization via Artist: {e}", exc_info=True)

    def get_best_parameters(self) -> Dict:
        """Return best parameters found during optimization."""
        return self.optimizer.max["params"] if self.optimizer and self.optimizer.max else {}

    def get_optimization_history(self) -> List[Dict]:
        """Return history of optimization attempts."""
        return self.optimizer.res if self.optimizer else []

    def validate_optimizer_config(self) -> bool:
        """Validate optimizer-specific configuration"""
        required_fields = ['n_iters', 'init_points', 'random_state']
        valid = all(hasattr(self.cfg.optimizer, field) for field in required_fields)

        if not valid:
            missing = [field for field in required_fields if not hasattr(self.cfg.optimizer, field)]
            logger.error(f"Missing required configuration fields for BayesOpt: {missing}")

        # Add any BayesOpt specific validation here
        # e.g., check acquisition function kind

        return valid

# Removed parse_scores function as it's not used directly in this class
