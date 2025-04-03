# optuna_optimizer.py - Version 1.3 - more samplers and better validate

import os
import time
import json
import logging
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import optuna

from pathlib import Path
from typing import Dict, List, Optional

from omegaconf import ListConfig
from optuna import Trial, Study
from optuna.trial import TrialState, FrozenTrial
from optuna.samplers import (
    TPESampler, RandomSampler, CmaEsSampler, QMCSampler, GridSampler, NSGAIISampler, GPSampler
)
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
from sd_optim.optimizer import Optimizer


logger = logging.getLogger(__name__)


class OptunaOptimizer(Optimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.study = None
        self._param_bounds = None
        self.log_name = self.cfg.get("log_name", "default")

        # Make checkpoint directory configurable
        self.checkpoint_dir = Path(self.cfg.optimizer.get("checkpoint_dir", os.getcwd())) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Enhanced settings
        self.checkpoint_interval = self.cfg.optimizer.get("checkpoint_interval", 10)

        # Early stopping settings
        self.early_stopping = self.cfg.optimizer.get("early_stopping", False)
        self.patience = self.cfg.optimizer.get("patience", 10)
        self.min_improvement = self.cfg.optimizer.get("min_improvement", 0.001)
        self.no_improvement_count = 0

        # Tracking metrics
        self.trial_scores = []
        self.optimization_start_time = None

        # Initialize logger for trials
        self.logger = self._setup_trial_logger()

    def _setup_trial_logger(self):
        """Setup logging for optimization trials."""

        class TrialLogger:
            def __init__(self, log_path):
                self.log_path = Path(log_path)
                self.log_path.parent.mkdir(parents=True, exist_ok=True)

                # Initialize with empty file if it doesn't exist
                if not self.log_path.exists():
                    with open(self.log_path, 'w', encoding='utf-8') as f:
                        f.write('')

            def log(self, data):
                with open(self.log_path, 'a', encoding='utf-8') as f:
                    json.dump(data, f)
                    f.write('\n')

            def load_trials(self):
                """Load trials from log file."""
                if not self.log_path.exists():
                    return []

                trials = []
                with open(self.log_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            trials.append(json.loads(line))
                return trials

        return TrialLogger(Path(os.getcwd()) / f"trials_{self.log_name}.jsonl")

    def validate_optimizer_config(self) -> bool:
        """Validate optimizer-specific configuration"""
        required_fields = ['n_iters', 'init_points', 'random_state']
        valid = all(hasattr(self.cfg.optimizer, field) for field in required_fields)

        if not valid:
            missing = [field for field in required_fields if not hasattr(self.cfg.optimizer, field)]
            logger.error(f"Missing required configuration fields: {missing}")
            return False  # Return early if basic fields missing

        # Validate sampler config
        sampler_config = self.cfg.optimizer.get("sampler", {})
        sampler_type = sampler_config.get("type", "tpe").lower()
        if sampler_type == "grid" and "search_space" not in sampler_config:
            logger.error("Grid sampler selected but 'search_space' is missing in optimizer.sampler config.")
            valid = False
        # Add more sampler-specific checks if needed

        # Validate pruner config (if pruning enabled)
        if self.cfg.optimizer.get("use_pruning", False):
            pruner_type = self.cfg.optimizer.get("pruner_type", "median").lower()
            if pruner_type not in ["median", "successive_halving"]:
                logger.warning(f"Unknown pruner_type '{pruner_type}'. Optuna might default or error.")
            # Add pruner-specific checks if needed

        return valid

    def _configure_sampler(self):
        """Configure and return the appropriate sampler based on configuration."""
        sampler_config = self.cfg.optimizer.get("sampler", {})
        sampler_type = sampler_config.get("type", "tpe").lower()
        seed = self.cfg.optimizer.random_state

        # Common sampler parameters
        sampler_kwargs = {
            "seed": seed
        }

        logger.info(f"Configuring sampler: type='{sampler_type}', seed={seed}")

        if sampler_type == "random":
            sampler = RandomSampler(**sampler_kwargs)
            logger.info("Using Random Sampler")

        elif sampler_type == "tpe":
            tpe_kwargs = {
                "n_startup_trials": self.cfg.optimizer.init_points,
                "multivariate": sampler_config.get("multivariate", True),
                "group": sampler_config.get("group", False), # Group parameters
                "warn_independent_sampling": sampler_config.get("warn_independent_sampling", True),
                "constant_liar": sampler_config.get("constant_liar", False),
                **sampler_kwargs
            }
            sampler = TPESampler(**tpe_kwargs)
            logger.info(f"Using TPE Sampler with options: {tpe_kwargs}")

        elif sampler_type == "cmaes":
            # CMA-ES is good for continuous, non-linear problems
            cmaes_kwargs = {
                "n_startup_trials": self.cfg.optimizer.init_points,
                "restart_strategy": sampler_config.get("restart_strategy", None), # 'ipop' or 'bipop'
                "sigma0": sampler_config.get("sigma0", None), # Initial step size
                "warn_independent_sampling": sampler_config.get("warn_independent_sampling", True),
                **sampler_kwargs
            }
            sampler = CmaEsSampler(**cmaes_kwargs)
            logger.info(f"Using CMA-ES Sampler with options: {cmaes_kwargs}")

        elif sampler_type == "gp":
            # Gaussian Process - can be powerful but potentially slower
            gp_kwargs = {
                "n_startup_trials": self.cfg.optimizer.init_points,
                # Add other GP-specific params from Optuna docs if needed
                **sampler_kwargs
            }
            sampler = GPSampler(**gp_kwargs)
            logger.info(f"Using Gaussian Process (GP) Sampler with options: {gp_kwargs}")

        elif sampler_type == "qmc":
            # Quasi-Monte Carlo - good for exploring high-dimensional spaces evenly
            qmc_kwargs = {
                "qmc_type": sampler_config.get("qmc_type", "sobol"), # 'sobol', 'halton', 'lhs'
                "scramble": sampler_config.get("scramble", True),
                "warn_independent_sampling": sampler_config.get("warn_independent_sampling", True),
                "warn_asyncronous_seeding": sampler_config.get("warn_asyncronous_seeding", True),
                **sampler_kwargs
            }
            sampler = QMCSampler(**qmc_kwargs)
            logger.info(f"Using QMC Sampler with options: {qmc_kwargs}")

        elif sampler_type == "grid":
            # Grid search - systematic but only practical for few parameters
            if "search_space" not in sampler_config:
                raise ValueError("Grid sampler requires a 'search_space' configuration in optimizer.sampler")
            # Convert search space values if they are lists (required by GridSampler)
            search_space = {k: list(v) if isinstance(v, (list, ListConfig)) else v
                            for k, v in sampler_config["search_space"].items()}
            sampler = GridSampler(search_space)
            logger.info(f"Using Grid Sampler with search space: {search_space}")

        elif sampler_type == "nsgaii":
            # NSGA-II - for multi-objective optimization (not directly applicable here, but included for completeness)
            # Note: Requires defining multiple objectives, which our current setup doesn't do.
            nsgaii_kwargs = {
                "population_size": sampler_config.get("population_size", 50),
                "mutation_prob": sampler_config.get("mutation_prob", None),
                "crossover_prob": sampler_config.get("crossover_prob", 0.9),
                "swapping_prob": sampler_config.get("swapping_prob", 0.5),
                "warn_independent_sampling": sampler_config.get("warn_independent_sampling", True),
                **sampler_kwargs
            }
            sampler = NSGAIISampler(**nsgaii_kwargs)
            logger.warning("Using NSGA-II Sampler - primarily for multi-objective optimization.")
            logger.info(f"NSGA-II options: {nsgaii_kwargs}")

        # elif sampler_type == "botorch":
        #     # Requires `pip install optuna[integration]` or `pip install botorch gpytorch`
        #     try:
        #         from optuna.integration import BoTorchSampler
        #         botorch_kwargs = {
        #             "n_startup_trials": self.cfg.optimizer.init_points,
        #             # Add other BoTorch specific params if needed
        #             **sampler_kwargs
        #         }
        #         sampler = BoTorchSampler(**botorch_kwargs)
        #         logger.info(f"Using BoTorch Sampler with options: {botorch_kwargs}")
        #     except ImportError:
        #         logger.error("BoTorchSampler requires 'botorch' and 'gpytorch'. Install with 'pip install botorch gpytorch'. Falling back to TPE.")
        #         sampler_type = "tpe" # Fallback
        #         # Fall through to TPE below

        else:
            if sampler_type != "tpe": # Avoid duplicate warning if default is used
                 logger.warning(f"Unknown sampler type: '{sampler_type}', falling back to TPE.")
            sampler = TPESampler(
                n_startup_trials=self.cfg.optimizer.init_points,
                multivariate=True,
                **sampler_kwargs
            )
            logger.info("Using TPE Sampler (Fallback)")

        return sampler

    async def optimize(self) -> None:
        """Run Optuna optimization process."""
        self.optimization_start_time = time.time()
        logger.debug(f"Initial Parameter Bounds: {self.optimizer_pbounds}")  # Use the attribute directly

        # Configure sampler
        sampler = self._configure_sampler() # Call the new function

        # Configure pruner if enabled
        pruner = None
        if self.cfg.optimizer.get("use_pruning", False):
            pruner_type = self.cfg.optimizer.get("pruner_type", "median")
            if pruner_type == "median":
                pruner = MedianPruner(n_startup_trials=self.cfg.optimizer.init_points)
                logger.info("Using Median Pruner")
            elif pruner_type == "successive_halving":
                pruner = SuccessiveHalvingPruner()
                logger.info("Using Successive Halving Pruner")

        study_name = f"optimization_{self.log_name}"
        # Use checkpoint dir for storage DB (ensure self.checkpoint_dir is Path)
        storage_path = self.checkpoint_dir / f'{study_name}.db'
        storage = f"sqlite:///{storage_path}"
        logger.info(f"Using Optuna storage: {storage}")

        # --- Determine if loading or creating ---
        # Use the common load_log_file flag to indicate resuming
        # Optuna's load_if_exists=True handles both creating and loading
        should_load_study = bool(self.cfg.optimizer.get("load_log_file", False))
        logger.info(f"Attempting to {'load' if should_load_study else 'create'} study '{study_name}'.")

        try:
            self.study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                sampler=sampler,
                pruner=pruner,
                direction="maximize",
                load_if_exists=True # <<< ALWAYS TRUE: Creates if not exist, loads if exists
            )
            # Log whether it was loaded or created
            if should_load_study and len(self.study.trials) > 0:
                 logger.info(f"Successfully loaded existing study '{study_name}' with {len(self.study.trials)} trials.")
            elif not should_load_study and len(self.study.trials) == 0:
                 logger.info(f"Successfully created new study '{study_name}'.")
            elif should_load_study and len(self.study.trials) == 0:
                 logger.warning(f"Load specified, but study '{study_name}' was created new or is empty.")
            # else: !should_load_study and len > 0 means created new study but somehow got trials? Unlikely.

        except Exception as e_study:
             # Fallback to in-memory might lose history unless we load from JSONL logs
             logger.error(f"Failed to create/load study using DB: {e_study}. Check DB path/permissions.")
             logger.info("Attempting fallback to in-memory storage.")
             self.study = optuna.create_study(study_name=study_name, sampler=sampler, pruner=pruner, direction="maximize")
             self._restore_from_trials_log() # Attempt restore if using in-memory

        # --- Calculate remaining trials ---
        completed_trials = len(self.study.trials) # Count trials loaded/existing
        total_trials_planned = self.cfg.optimizer.init_points + self.cfg.optimizer.n_iters
        remaining_trials = max(0, total_trials_planned - completed_trials)

        if completed_trials > 0:
            logger.info(f"Found {completed_trials} completed trials. {remaining_trials} trials remaining.")

            # Update best_rolling_score from existing trials
            if self.study.best_trial and self.study.best_trial.value is not None:
                self.best_rolling_score = max(self.best_rolling_score, self.study.best_value)
                logger.info(f"Best score from previous runs: {self.best_rolling_score}")

        # Run optimization if there are trials remaining
        if remaining_trials > 0:
            try:
                # Run Optuna's synchronous optimize in a separate thread
                import asyncio
                await asyncio.to_thread(
                    self.study.optimize,
                    func=self._objective,
                    n_trials=remaining_trials,
                    n_jobs=self.cfg.optimizer.get("n_jobs", 1),
                    show_progress_bar=True,
                    callbacks=[self._trial_callback, self._checkpoint_callback]
                )

            except KeyboardInterrupt:
                logger.info("Optimization interrupted by user. Saving current state...")
                self.save_checkpoint()
            except Exception as e:
                logger.error(f"Optimization failed: {e}")
                self.save_checkpoint()
                raise
        else:
            logger.info("All trials already completed. Skipping optimization.")

        # Save final checkpoint
        self.save_checkpoint()

        # Analyze parameter importance
        self._analyze_parameter_importance()

    def _restore_from_trials_log(self):
        """Restore trials from the JSON log file if database is unavailable."""
        trials_data = self.logger.load_trials()
        if not trials_data:
            logger.warning("No trials found in log file. Starting fresh optimization.")
            return

        logger.info(f"Restoring {len(trials_data)} trials from log file")

        for trial_data in trials_data:
            if "params" in trial_data and "target" in trial_data:
                params = trial_data["params"]
                value = trial_data["target"]

                # Add trial to study
                if value is not None:  # Skip failed trials
                    self.study.add_trial(
                        optuna.trial.create_trial(
                            params=params,
                            value=value,
                            state=optuna.trial.TrialState.COMPLETE
                        )
                    )

        logger.info(f"Restored {len(self.study.trials)} valid trials")

    def save_checkpoint(self):
        """Save current optimization state."""
        if not self.study:
            logger.warning("No study to checkpoint")
            return

        try:
            # Create checkpoint directory if it doesn't exist
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Save additional state that isn't captured in the database
            checkpoint_data = {
                'best_rolling_score': self.best_rolling_score,
                'iteration': self.iteration,
                'timestamp': time.time()
            }

            checkpoint_file = self.checkpoint_dir / f"optimizer_state_{self.log_name}.pkl"
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)

            logger.info(f"Saved optimizer state to {checkpoint_file}")

            # The main Optuna study state is already saved in the SQLite database
            # No need to manually save it if using a persistent storage

            # Also create visualization at checkpoint
            self._create_progress_plots()
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def _checkpoint_callback(self, study: Study, trial: Trial):
        """Callback to periodically save checkpoints."""
        if trial.number % self.checkpoint_interval == 0 and trial.number > 0:
            logger.info(f"Creating periodic checkpoint at trial {trial.number}")
            self.save_checkpoint()

        # Perform garbage collection periodically
        if trial.number % 5 == 0 and trial.number > 0:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _objective(self, trial: Trial) -> float:
        """Objective function for Optuna optimization."""
        # Convert param bounds to Optuna parameter suggestions
        params = {}
        # Use self.optimizer_pbounds (or self.param_info) which was set during __post_init__
        if not self.optimizer_pbounds:
            logger.error("Optimizer bounds not initialized in _objective. Cannot suggest parameters.")
            raise optuna.exceptions.TrialPruned("Bounds not available")  # Prune trial if bounds missing

        # V1.1 - Corrected bound handling for float ranges
        for param_name, bounds_value in self.optimizer_pbounds.items():
            if isinstance(bounds_value, tuple) and len(bounds_value) == 2:
                # REMOVED check for exact (0.0, 1.0) tuple

                # Check for integer range first
                if all(isinstance(v, int) or (isinstance(v, float) and v.is_integer()) for v in bounds_value):
                    # Ensure bounds are actually different before suggesting int range
                    low, high = int(bounds_value[0]), int(bounds_value[1])
                    if low == high:
                        params[param_name] = low  # Treat as fixed if bounds are same
                        trial.set_user_attr(f"fixed_{param_name}", True)
                    else:
                        params[param_name] = trial.suggest_int(param_name, low, high)
                # Default to float range (handles (0.0, 1.0) correctly now)
                else:
                    low, high = float(bounds_value[0]), float(bounds_value[1])
                    # Ensure bounds are different before suggesting range
                    if abs(low - high) < 1e-9:  # Check for near equality for floats
                        params[param_name] = low  # Treat as fixed
                        trial.set_user_attr(f"fixed_{param_name}", True)
                    else:
                        params[param_name] = trial.suggest_float(param_name, low, high)

            elif isinstance(bounds_value, list):
                # Categorical list - check if only one option (fixed)
                if len(bounds_value) == 1:
                    params[param_name] = bounds_value[0]
                    trial.set_user_attr(f"fixed_{param_name}", True)
                else:
                    params[param_name] = trial.suggest_categorical(param_name, bounds_value)
            elif isinstance(bounds_value, (int, float)):
                # Fixed value from custom_bounds
                params[param_name] = bounds_value
                trial.set_user_attr(f"fixed_{param_name}", True)
            else:
                logger.warning(
                    f"Unsupported bound type for Optuna suggestion: {type(bounds_value)} for '{param_name}'. Skipping.")
                raise optuna.exceptions.TrialPruned(f"Unsupported bounds for {param_name}")

        try:
            # Run the async function in a synchronous context
            import asyncio
            result = asyncio.run(self.sd_target_function(params))

            # Update metrics
            self.trial_scores.append(result)

            # Implement early stopping if enabled
            if self.early_stopping:
                if self.trial_scores and result > (max(self.trial_scores[:-1] or [0]) + self.min_improvement):
                    self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1

                if self.no_improvement_count >= self.patience:
                    logger.info(f"Early stopping triggered after {len(self.trial_scores)} trials")
                    raise optuna.exceptions.TrialPruned()

            return result

        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            logger.error(f"Error in objective function: {e}", exc_info=True)
            return float('-inf')  # Return very negative score on error

    def _trial_callback(self, study: Study, trial: FrozenTrial) -> None:
        """Callback to log trial information."""
        elapsed_time = time.time() - self.optimization_start_time if self.optimization_start_time else 0

        # Store custom attributes in the trial itself
        trial.set_user_attr("elapsed_seconds", elapsed_time)
        trial.set_user_attr("timestamp", time.time())

        # For complex data structures, convert to JSON string
        if hasattr(self, 'merger') and hasattr(self.merger, 'output_file'):
            trial.set_user_attr("model_path", str(self.merger.output_file))

        # We can also store additional performance metrics here
        if hasattr(self, 'iteration'):
            trial.set_user_attr("iteration", self.iteration)

        # Still keep our custom logging for backward compatibility
        log_data = {
            "trial_number": trial.number,
            "target": trial.value,
            "params": trial.params,
            "state": trial.state.name,
            "datetime": {
                "datetime": trial.datetime_start.isoformat() if trial.datetime_start else None,
                "elapsed_seconds": elapsed_time
            }
        }

        # Write to the JSON logger
        self.logger.log(log_data)

        # Create visualizations periodically
        if trial.number % 10 == 0 and trial.number > 0:
            self._create_progress_plots()

    def _create_progress_plots(self):
        """Create plots showing optimization progress."""
        try:
            if not self.study or not self.study.trials:
                return

            # Create optimization history plot
            plt.figure(figsize=(12, 6))

            # Extract data
            values = [t.value for t in self.study.trials if t.value is not None]
            if not values:
                return

            # Create running best curve
            x = list(range(1, len(values) + 1))
            running_best = [max(values[:i + 1]) for i in range(len(values))]

            # Plot results
            plt.plot(x, values, 'o-', alpha=0.5, label='Trial Value')
            plt.plot(x, running_best, 'r-', linewidth=2, label='Best Value')

            plt.title('Optimization Progress')
            plt.xlabel('Trial Number')
            plt.ylabel('Score')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Save plot
            output_dir = Path(os.getcwd())
            plt.savefig(output_dir / f"optimization_progress_{self.log_name}.png")
            plt.close()

        except Exception as e:
            logger.error(f"Failed to create progress plots: {e}")

    def _analyze_parameter_importance(self):
        """Analyze importance of each parameter in the optimization."""
        try:
            if not self.study or len(self.study.trials) < 5:
                logger.warning("Not enough trials to analyze parameter importance")
                return

            # Use Optuna's built-in importance analyzer
            importance = optuna.importance.get_param_importances(self.study)

            logger.info("\nParameter Importance:")
            for param_name, score in importance.items():
                logger.info(f"{param_name}: {score:.4f}")

            # Create visualization
            plt.figure(figsize=(12, 8))
            param_names = list(importance.keys())
            scores = list(importance.values())

            # Sort by importance
            sorted_indices = np.argsort(scores)
            param_names = [param_names[i] for i in sorted_indices]
            scores = [scores[i] for i in sorted_indices]

            plt.barh(param_names, scores)
            plt.xlabel('Importance Score')
            plt.title('Parameter Importance')
            plt.tight_layout()

            # Save visualization
            output_dir = Path(os.getcwd())
            plt.savefig(output_dir / f"parameter_importance_{self.log_name}.png")
            plt.close()

        except Exception as e:
            logger.error(f"Failed to analyze parameter importance: {e}")

    async def postprocess(self) -> None:
        """Perform post-optimization analysis and reporting."""
        logger.info("\n" + "=" * 50)
        logger.info("Optimization Results Recap!")
        logger.info("=" * 50)

        if not self.study or not self.study.trials:
            logger.warning("No trials completed. Nothing to report.")
            return

        # Log optimization statistics
        total_trials = len(self.study.trials)
        completed_trials = sum(1 for t in self.study.trials if t.state.is_finished())

        logger.info(f"Total Trials: {total_trials}")
        logger.info(f"Completed Trials: {completed_trials}")

        # Calculate runtime statistics
        if self.optimization_start_time:
            total_runtime = time.time() - self.optimization_start_time
            hours, remainder = divmod(total_runtime, 3600)
            minutes, seconds = divmod(remainder, 60)
            logger.info(f"Total runtime: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
            logger.info(f"Average time per trial: {total_runtime / max(1, total_trials):.2f} seconds")

        # Log best trials
        logger.info("\nTop 5 Trials:")
        completed_trials = [t for t in self.study.trials if t.value is not None]
        sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)

        for i, trial in enumerate(sorted_trials[:5]):
            logger.info(f"Rank {i + 1} (Trial {trial.number}):")
            logger.info(f"\tValue: {trial.value:.4f}")
            logger.info(f"\tParameters: {trial.params}")

        # Log best trial details
        logger.info("\nBest Trial:")
        logger.info(f"Value: {self.study.best_value:.4f}")
        logger.info(f"Parameters: {self.study.best_params}")

        # Create visualization
        try:
            await self.artist.visualize_optimization()
        except Exception as e:
            logger.error(f"Failed to create visualization: {e}")

    def get_best_parameters(self) -> Dict:
        """Return best parameters found during optimization."""
        return self.study.best_params if self.study and hasattr(self.study, 'best_params') else {}

    def get_optimization_history(self) -> List[Dict]:
        """Return history of optimization attempts."""
        if not self.study:
            return []

        history = []
        for trial in self.study.trials:
            if trial.value is not None:
                history.append({
                    "trial_number": trial.number,
                    "value": trial.value,
                    "params": trial.params,
                    "datetime": trial.datetime_start.isoformat() if trial.datetime_start else None
                })
        return history

    def launch_dashboard(self, port=8080):
        """Launch the Optuna Dashboard for interactive visualization."""
        storage_path = os.path.join(os.getcwd(), f'optimization_{self.log_name}.db')

        print(f"\n{'=' * 80}")
        print(f"LAUNCHING OPTUNA DASHBOARD")
        print(f"{'=' * 80}")
        print(f"Access the dashboard at: http://localhost:{port}")
        print(f"Press Ctrl+C in this terminal to stop the dashboard when finished")
        print(f"{'=' * 80}\n")

        # This will block until Ctrl+C
        os.system(f"optuna-dashboard sqlite:///{storage_path} --port {port}")

    def create_visualization_report(self, output_dir=None):
        """Generate comprehensive Optuna visualization report."""
        if not self.study or len(self.study.trials) < 2:
            logger.warning("Not enough trials for visualization")
            return

        if output_dir is None:
            output_dir = Path(os.getcwd()) / "optuna_visualizations"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            import optuna.visualization as vis
            from optuna.visualization import matplotlib as vis_mpl

            # 1. Optimization history plot
            fig = vis_mpl.plot_optimization_history(self.study)
            fig.savefig(output_dir / "optimization_history.png")
            plt.close(fig)

            # 2. Parameter importance plot
            fig = vis_mpl.plot_param_importances(self.study)
            fig.savefig(output_dir / "param_importances.png")
            plt.close(fig)

            # 3. Slice plot for each parameter
            fig = vis_mpl.plot_slice(self.study)
            fig.savefig(output_dir / "param_slices.png")
            plt.close(fig)

            # 4. Contour plot for top parameters
            try:
                fig = vis_mpl.plot_contour(self.study)
                fig.savefig(output_dir / "param_contour.png")
                plt.close(fig)
            except:
                logger.info("Could not generate contour plot (requires at least 2 parameters)")

            # 5. Parallel coordinate plot
            fig = vis_mpl.plot_parallel_coordinate(self.study)
            fig.savefig(output_dir / "parallel_coordinate.png")
            plt.close(fig)

            # 6. EDF (Empirical Distribution Function) plot
            try:
                fig = vis_mpl.plot_edf(self.study)
                fig.savefig(output_dir / "edf.png")
                plt.close(fig)
            except:
                logger.info("Could not generate EDF plot")

            logger.info(f"Visualization report generated at {output_dir}")

        except ImportError:
            logger.error("Optuna visualization modules not available. Install with pip install optuna[visualization]")
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
