# optuna_optimizer.py - Version 1.5 - directory correction

import os
import subprocess
import time
import json
import logging
import pickle
import numpy as np
import optuna.visualization as vis
import torch
import optuna

from pathlib import Path
from typing import Dict, List, Optional

from hydra.core.hydra_config import HydraConfig
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
#        self.log_name = self.cfg.get("log_name", "default")
        self.study_name = None # Will be set in optimize()

        # --- Determine Optuna Storage Directory ---
        try:
            # 1. Determine Project Root (assuming this file is in sd_optim/)
            project_root = Path(__file__).parent.parent.resolve() # sd_optim/ -> sd-optim/

            # 2. Define Default Path (relative to project root)
            default_storage_path = project_root / "optuna_db" # e.g., D:\...\sd-optim\optuna_db

            # 3. Read Config or Use Default
            # Use '.get' to safely access nested keys, provide default path as string
            optuna_storage_path_str = self.cfg.optimizer.optuna_config.get("storage_dir", str(default_storage_path))

            # 4. Resolve Final Path and Create Directory
            self.optuna_storage_dir = Path(optuna_storage_path_str).resolve()
            self.optuna_storage_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Optuna databases will be stored in: {self.optuna_storage_dir}")

        except NameError:
             logger.error("__file__ not defined. Cannot reliably determine project root for default Optuna storage.")
             # Handle error - maybe fallback to CWD or raise
             self.optuna_storage_dir = Path("./optuna_db_fallback").resolve()
             self.optuna_storage_dir.mkdir(parents=True, exist_ok=True)
             logger.warning(f"Falling back to Optuna storage directory: {self.optuna_storage_dir}")
        except Exception as e:
             logger.error(f"CRITICAL ERROR setting up Optuna storage directory: {e}", exc_info=True)
             raise # Re-raise critical errors

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

        # Define the logger class internally
        class TrialLogger:
            def __init__(self):
                self.log_path = None  # <<< Initialize path as None

            def set_path(self, log_path: Path):  # <<< Add method to set path later
                self.log_path = log_path
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                # Initialize with empty file if it doesn't exist AFTER path is set
                if not self.log_path.exists():
                    with open(self.log_path, 'w', encoding='utf-8') as f: f.write('')
                logger.info(f"Trial log (.jsonl) will be saved to: {self.log_path}")

            def log(self, data):
                if not self.log_path:  # <<< Check if path is set before logging
                    logger.error("Trial logger path not set. Cannot log trial.")
                    return
                try:  # Add error handling for writing
                    with open(self.log_path, 'a', encoding='utf-8') as f:
                        json.dump(data, f) # Type: ignore
                        f.write('\n')
                except Exception as e:
                    logger.error(f"Failed to write to trial log {self.log_path}: {e}")

            def load_trials(self):
                """Load trials from log file."""
                if not self.log_path or not self.log_path.exists():  # <<< Check path
                    return []

                trials = []
                try:
                    with open(self.log_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    trials.append(json.loads(line))
                                except json.JSONDecodeError as e_json:
                                    logger.warning(f"Skipping invalid line in trial log: {e_json}")
                    return trials
                except Exception as e:
                    logger.error(f"Failed to load trials log {self.log_path}: {e}")
                    return []

        # Return an instance of the logger class, path will be set later
        return TrialLogger()

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

        # --- Determine Shared DB Name and Path ---
        storage_path: Path = None
        storage: str = None
        db_filename: str = "optuna_fallback.db"
        try:
            opt_mode = self.cfg.get("optimization_mode", "unknown_mode")
            merge_method_name = "N/A" # Default for non-merge modes
            if opt_mode == "merge":
                 merge_method_name = self.cfg.get("merge_method", "unknown_method")
            elif opt_mode == "recipe":
                 # Maybe use recipe filename base?
                 recipe_path = self.cfg.recipe_optimization.get("recipe_path")
                 merge_method_name = f"recipe_{Path(recipe_path).stem}" if recipe_path else "recipe"
            elif opt_mode == "layer_adjust":
                 merge_method_name = "layer_adjust"

            scorers = self.cfg.get("scorer_method", ["unknown_scorer"])
            # Ensure scorers is a list even if single string is given
            if isinstance(scorers, str): scorers = [scorers]
            # Sort scorers for consistent naming if list order changes
            scorer_name_part = "_".join(sorted(scorers))

            # 2. Sanitize names (simple example, might need more robust)
            def sanitize(name): return "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in str(name))

            db_filename_base = f"optuna_{sanitize(opt_mode)}_{sanitize(merge_method_name)}_{sanitize(scorer_name_part)}"
            db_filename = f"{db_filename_base}.db"

            # 3. Define storage path using the derived filename
            if not hasattr(self, 'optuna_storage_dir') or not self.optuna_storage_dir.is_dir():
                 raise ValueError("Optuna storage directory not initialized correctly.")

            storage_path = self.optuna_storage_dir / db_filename  # <<< Use correct storage dir

            storage = f"sqlite:///{storage_path.resolve()}"
            logger.info(f"Using Optuna storage file: {storage_path.name}")
            logger.info(f"Full storage URI: {storage}")

        except Exception as e_name:
            logger.error(f"Failed to determine Optuna DB name from config: {e_name}. Falling back to default.")
            # Fallback to a simple default if naming fails
            db_filename = "optuna_fallback.db"
            storage_path = self.optuna_storage_dir / db_filename
            storage = f"sqlite:///{storage_path.resolve()}"

        # --- Check if resuming a specific study ---
        study_to_resume = self.cfg.optimizer.get("resume_study_name", None) # Get name from config

        try:
            if study_to_resume:
                logger.info(f"Attempting to load and resume study '{study_to_resume}' from DB '{storage_path.name}'...")
                self.study = optuna.load_study(
                    study_name=study_to_resume,
                    storage=storage,
                    sampler=sampler, # Pass sampler/pruner in case they need setup? Or load study first? Check Optuna docs.
                    pruner=pruner
                )
                logger.info(f"Successfully loaded study '{study_to_resume}' with {len(self.study.trials)} existing trials.")
                # Set self.study_name to the resumed name for consistency elsewhere (e.g., dashboard launch)
                self.study_name = study_to_resume
            else:
                # --- Create a NEW, unique study for this run ---
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                sampler_type = self.cfg.optimizer.get("sampler", {}).get("type", "tpe").lower()
                self.study_name = f"run_{timestamp}_{sampler_type}" # Generate NEW name

                logger.info(f"Creating new study '{self.study_name}' in DB '{storage_path.name}'.")
                self.study = optuna.create_study(
                    study_name=self.study_name, # Use the NEW unique name
                    storage=storage,
                    sampler=sampler,
                    pruner=pruner,
                    direction="maximize",
                    load_if_exists=False # <<< Explicitly FALSE: Do not load even if name conflicts (unlikely)
                )
                logger.info(f"Successfully created new study '{self.study_name}'.")

                # --- V-- ADDITION START: Set user attributes only for NEW studies --V ---
                # Only set these when the study is definitely new to avoid overwriting
                # attributes from a potentially loaded study if resuming logic changes later.
                try:
                    # Get model paths safely, default to empty list
                    model_paths_list = self.cfg.get("model_paths", [])
                    # Convert Path objects to strings if they exist
                    input_model_paths_str = [str(p) for p in model_paths_list]

                    self.study.set_user_attr('config_input_models', str(input_model_paths_str))
                    self.study.set_user_attr('config_base_model_index', self.cfg.get('base_model_index', -1))
                    # Store the merge method used ONLY if mode is 'merge'
                    if self.cfg.get("optimization_mode") == "merge":
                         self.study.set_user_attr('config_merge_method', self.cfg.get('merge_method', 'N/A'))
                    else:
                         self.study.set_user_attr('config_merge_method', 'N/A') # Indicate not applicable
                    # Add other relevant top-level config for context?
                    self.study.set_user_attr('config_optimization_mode', self.cfg.get('optimization_mode', 'N/A'))
                    self.study.set_user_attr('config_scorers', str(self.cfg.get('scorer_method', [])))

                    logger.info(f"Stored initial configuration as user attributes for new study '{self.study_name}'.")
                except Exception as e_attr:
                    logger.warning(f"Could not store initial config as study attribute: {e_attr}")
                # --- V-- ADDITION END --V ---

            # --- V-- MOVED Logger Path Setup Here --V ---
            # Now self.study_name is guaranteed to be set (either loaded or created)
            # And os.getcwd() is the correct Hydra run directory
            try:
                hydra_run_path = Path(HydraConfig.get().runtime.output_dir) # Get correct run dir
                trials_log_filename = f"{self.study_name}_trials.jsonl" # Use the unique study name
                trials_log_path = hydra_run_path / trials_log_filename
                self.logger.set_path(trials_log_path) # Set the path on the logger instance
            except Exception as e_log_path:
                 logger.error(f"Failed to set trial logger path: {e_log_path}")
                 # Continue without jsonl logging if path fails?
            # --- V-- END Logger Path Setup --V ---

        except KeyError: # Optuna raises KeyError if study_name not found during load_study
             logger.error(f"Study name '{study_to_resume}' not found in the database '{storage_path.name}'. Cannot resume.")
             # Decide behaviour: stop execution or create a new study anyway? Stopping is safer.
             print(f"ERROR: Could not find the study '{study_to_resume}' to resume in {storage_path.name}.")
             raise ValueError(f"Failed to resume study '{study_to_resume}'.") # Stop execution
        except Exception as e_study:
             # Handle other DB connection/creation errors
             logger.error(f"Failed to create/load study using DB: {e_study}. Check DB path/permissions.")
             logger.info("Attempting fallback to in-memory storage.")
             self.study = optuna.create_study(study_name=self.study_name, sampler=sampler, pruner=pruner, direction="maximize")
             self._restore_from_trials_log() # Attempt restore if using in-memory

        # --- Calculate remaining trials ---
        completed_trials = len(self.study.trials) # Count trials loaded/existing
        total_trials_planned = self.cfg.optimizer.init_points + self.cfg.optimizer.n_iters
        remaining_trials = max(0, total_trials_planned - completed_trials)

        if self.study and completed_trials > 0: # Check if study is loaded and has trials
            logger.info(f"Found {completed_trials} completed trials. {remaining_trials} trials remaining.")

            # Update best_rolling_score from existing trials
            if self.study.best_trial and self.study.best_trial.value is not None:
                self.best_rolling_score = max(self.best_rolling_score, self.study.best_value)
                logger.info(f"Best score from previous runs: {self.best_rolling_score}")

        # Run optimization if there are trials remaining
        if remaining_trials > 0:
            try:
                import asyncio
                await asyncio.to_thread(
                    self.study.optimize,
                    func=self._objective,
                    n_trials=remaining_trials,
                    n_jobs=self.cfg.optimizer.get("n_jobs", 1),
                    show_progress_bar=True,
                    # --- VVV REMOVED _checkpoint_callback VVV ---
                    callbacks=[self._trial_callback] # Only log trial info
                )

            except KeyboardInterrupt:
                logger.info("Optimization interrupted by user.")
                # --- VVV REMOVED self.save_checkpoint() VVV ---
            except Exception as e:
                logger.error(f"Optimization failed: {e}", exc_info=True) # Log full traceback
                # --- VVV REMOVED self.save_checkpoint() VVV ---
                raise # Re-raise the exception
        else:
            logger.info("All trials already completed. Skipping optimization.")

        # --- VVV REMOVED final self.save_checkpoint() VVV ---

        # Analyze parameter importance (unchanged)
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

    def _objective(self, trial: Trial) -> float:
        """Objective function for Optuna optimization."""
        # Convert param bounds to Optuna parameter suggestions
        params = {}
        # Use self.optimizer_pbounds (or self.param_info) which was set during __post_init__
        if not self.optimizer_pbounds:
            logger.error("Optimizer bounds not initialized in _objective. Cannot suggest parameters.")
            raise optuna.exceptions.TrialPruned("Bounds not available")  # Prune trial if bounds missing

        # V1.2 - Corrected int vs float range detection
        for param_name, bounds_value in self.optimizer_pbounds.items():
            suggested_value = None
            try:
                logger.debug(f"Suggesting for: '{param_name}', Bounds: {bounds_value} (Type: {type(bounds_value)})")
                if isinstance(bounds_value, tuple) and len(bounds_value) == 2:
                    # --- CORRECTED LOGIC ---
                    # Check if BOTH bounds are actual Python integers
                    if all(isinstance(v, int) for v in bounds_value):
                        # Explicit Integer Range
                        low, high = int(bounds_value[0]), int(bounds_value[1])
                        if low == high:
                            suggested_value = low
                            trial.set_user_attr(f"fixed_{param_name}", True)
                            logger.debug(" -> Fixed Int")
                        else:
                            suggested_value = trial.suggest_int(param_name, low, high)
                            logger.debug(" -> suggest_int (Explicit Int Range)")
                    # Otherwise (at least one is float, or they are floats like 0.0, 1.0), treat as float
                    else:
                        low, high = float(bounds_value[0]), float(bounds_value[1])
                        if abs(low - high) < 1e-9:  # Check for fixed float
                            suggested_value = low
                            trial.set_user_attr(f"fixed_{param_name}", True)
                            logger.debug(" -> Fixed Float")
                        else:
                            # This now correctly handles (0.0, 1.0)
                            suggested_value = trial.suggest_float(param_name, low, high)
                            logger.debug(" -> suggest_float (Float Range or Mixed)")
                    # --- END OF CORRECTION ---

                elif isinstance(bounds_value, list):
                    # Categorical logic (remains the same)
                    if len(bounds_value) == 1:
                        suggested_value = bounds_value[0]
                        trial.set_user_attr(f"fixed_{param_name}", True)
                        logger.debug(" -> Fixed Categorical")
                    else:
                        suggested_value = trial.suggest_categorical(param_name, bounds_value)
                        logger.debug(" -> suggest_categorical")
                elif isinstance(bounds_value, (int, float)):
                    # Fixed value logic (remains the same)
                    suggested_value = bounds_value
                    trial.set_user_attr(f"fixed_{param_name}", True)
                    logger.debug(" -> Fixed Value")
                else:
                    logger.warning(f"Unsupported bound type: {type(bounds_value)} for '{param_name}'.")
                    raise optuna.exceptions.TrialPruned(f"Unsupported bounds for {param_name}")

                params[param_name] = suggested_value
            except Exception as e_suggest:
                logger.error(f"Error during suggestion for '{param_name}' with bounds {bounds_value}: {e_suggest}",
                             exc_info=True)
                raise  # Re-raise error to stop trial

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

    # --- MODIFIED: _create_progress_plots ---
    def _create_progress_plots(self):
        """Create plot showing optimization progress using Optuna's Plotly backend."""
        if not self.study or not self.study.trials:
            logger.warning("_create_progress_plots: No study or trials available.")
            return

        # Check if enough trials exist for the plot
        completed_trials = [t for t in self.study.trials if t.state == TrialState.COMPLETE and t.value is not None]
        if len(completed_trials) < 1: # History plot needs at least one point
            logger.debug("_create_progress_plots: Not enough completed trials yet.")
            return

        try:
            # Generate the plot using Optuna's function
            fig = vis.plot_optimization_history(self.study)

            # Determine output directory using Hydra
            try:
                 run_dir = Path(HydraConfig.get().runtime.output_dir)
            except ValueError:
                 logger.error("_create_progress_plots: Hydra context unavailable. Cannot determine save path.")
                 return
            # Maybe save periodic plots to a sub-folder?
            output_dir = run_dir / "visualizations" / "periodic"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save plot using write_image
            save_path = output_dir / f"optuna_history_trial_{self.study.trials[-1].number:04d}.png"
            try:
                fig.write_image(str(save_path))
                logger.info(f"Saved periodic progress plot to {save_path}")
            except ValueError as ve:
                if "kaleido" in str(ve).lower(): logger.error(f"Failed to save periodic plot: Kaleido missing/broken.")
                else: logger.error(f"ValueError saving periodic plot: {ve}.")
            except Exception as e_write: logger.error(f"Error saving periodic plot: {e_write}.")

        except ImportError: logger.error("Optuna visualization module not available for periodic plots.")
        except Exception as e: logger.error(f"Failed to create periodic progress plot: {e}", exc_info=True)

    # --- MODIFIED: _analyze_parameter_importance ---
    def _analyze_parameter_importance(self):
        """Analyze and plot importance of parameters using Optuna's functions."""
        # This method might be redundant if postprocess already generates the importance plot.
        # Keeping it updated for consistency or potential separate use.
        if not self.study or len(self.study.trials) < 2: # Importance usually needs >= 2 trials
            logger.warning("_analyze_parameter_importance: Not enough trials.")
            return

        try:
            # --- Get importance data (remains the same) ---
            try:
                 # Calculate importance (might raise error if no completed trials or issue with metric)
                 importance = optuna.importance.get_param_importances(self.study)
                 logger.info("\nParameter Importance Analysis:")
                 for param_name, score in importance.items():
                     logger.info(f"  {param_name}: {score:.4f}")
            except Exception as e_imp:
                 logger.error(f"Could not calculate parameter importance: {e_imp}")
                 return # Cannot plot if calculation fails

            # --- Generate plot using Optuna's function ---
            fig = vis.plot_param_importances(self.study)

            # Determine output directory using Hydra
            try:
                 run_dir = Path(HydraConfig.get().runtime.output_dir)
            except ValueError:
                 logger.error("_analyze_parameter_importance: Hydra context unavailable. Cannot determine save path.")
                 return
            output_dir = run_dir / "visualizations" # Save with other postprocess plots
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save plot using write_image
            save_path = output_dir / f"optuna_param_importances_{self.study_name}.png"
            try:
                fig.write_image(str(save_path))
                logger.info(f"Saved parameter importance plot to {save_path}")
            except ValueError as ve:
                if "kaleido" in str(ve).lower(): logger.error(f"Failed to save importance plot: Kaleido missing/broken.")
                else: logger.error(f"ValueError saving importance plot: {ve}.")
            except Exception as e_write: logger.error(f"Error saving importance plot: {e_write}.")

        except ImportError as imp_err:
             # Handle missing optional dependencies like scikit-learn
             logger.error(f"Could not generate importance plot due to missing dependency: {imp_err}")
        except Exception as e:
            logger.error(f"Failed to analyze/plot parameter importance: {e}", exc_info=True)

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

        # --- ADD Optuna Plot Generation ---
        logger.info("Generating Optuna visualizations...")
        if len(self.study.trials) < 2:
            logger.warning("Not enough trials (need >= 2) for most Optuna visualizations.")
            return

        try:
            # Define output directory using Hydra's run directory
            try:
                run_dir = Path(HydraConfig.get().runtime.output_dir)
            except ValueError:
                 logger.error("Hydra context not available. Cannot determine output directory for plots.")
                 return
            output_dir = run_dir / "visualizations"
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving Optuna plots to: {output_dir}")

            # Dictionary of plot functions to call {name: function}
            # Using optuna.visualization (defaults to Plotly)
            plot_functions = {
                "optimization_history": vis.plot_optimization_history,
                "param_importances": vis.plot_param_importances,
                "slice": vis.plot_slice,
                "parallel_coordinate": vis.plot_parallel_coordinate,
                # Add others if desired and handle potential errors (like contour needing >1 param)
                # "contour": vis.plot_contour,
                # "edf": vis.plot_edf,
                # "rank": vis.plot_rank,
            }

            for name, plot_func in plot_functions.items():
                try:
                    fig = plot_func(self.study) # Generate Plotly figure
                    save_path = output_dir / f"optuna_{name}_{self.study_name}.png"

                    # Save using write_image (requires kaleido)
                    try:
                        fig.write_image(str(save_path))
                        logger.info(f"  Successfully saved Optuna plot: {name}")
                    except ValueError as ve:
                        if "kaleido" in str(ve).lower():
                             logger.error(f"  Failed to save Plotly plot '{name}': Kaleido engine not found or not functional. Please install with 'pip install -U kaleido'. Skipping save.")
                        else:
                             logger.error(f"  ValueError saving Plotly plot '{name}': {ve}. Skipping save.")
                    except Exception as e_write:
                         logger.error(f"  Unexpected error saving Plotly plot '{name}': {e_write}. Skipping save.")

                except (ValueError, TypeError) as plot_err:
                     # Handle errors like not enough parameters for contour, etc.
                     logger.warning(f"  Could not generate Optuna plot '{name}': {plot_err}. Skipping.")
                except ImportError as imp_err:
                     # Handle missing optional dependencies like scikit-learn for importance
                     logger.warning(f"  Could not generate Optuna plot '{name}' due to missing dependency: {imp_err}. Skipping.")
                except Exception as e_gen:
                     logger.error(f"  Unexpected error generating Optuna plot '{name}': {e_gen}", exc_info=True)

        except ImportError:
            # Should not happen if Optuna is installed, but good practice
            logger.error("Optuna library seems to be missing? Cannot generate visualizations.")
        except Exception as e_vis:
            logger.error(f"An error occurred during Optuna visualization generation: {e_vis}", exc_info=True)

        logger.info("=" * 50)
        logger.info("Optuna Postprocessing Finished")
        logger.info("=" * 50)
        # --- END Optuna Plot Generation ---


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

    # --- MODIFIED: create_visualization_report ---
    def create_visualization_report(self, output_dir=None):
        """Generate comprehensive Optuna visualization report using Plotly."""
        if not self.study or len(self.study.trials) < 2:
            logger.warning("Not enough trials for visualization report.")
            return

        # Determine output directory
        if output_dir is None:
            try:
                run_dir = Path(HydraConfig.get().runtime.output_dir)
            except ValueError:
                 logger.error("Hydra context not available. Cannot determine default output directory for report.")
                 # Fallback or raise error
                 output_dir = Path("./optuna_visualizations_report") # Example fallback
                 logger.warning(f"Using fallback directory for report: {output_dir}")
            else:
                 output_dir = run_dir / "visualizations_report" # Use a separate folder? Or same as postprocess?

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Generating Optuna visualization report in: {output_dir}")

        try:
            # Use default Plotly backend
            import optuna.visualization as vis

            # List of all desired plots (add/remove as needed)
            report_plots = {
                "optimization_history": vis.plot_optimization_history,
                "param_importances": vis.plot_param_importances,
                "slice": vis.plot_slice,
                "parallel_coordinate": vis.plot_parallel_coordinate,
                "contour": vis.plot_contour,
                "edf": vis.plot_edf,
                "rank": vis.plot_rank,
                # Add timeline, intermediate_values etc. if useful
                # "timeline": vis.plot_timeline,
                # "intermediate_values": vis.plot_intermediate_values,
            }

            for name, plot_func in report_plots.items():
                try:
                    fig = plot_func(self.study)
                    save_path = output_dir / f"optuna_{name}_{self.study_name}.png" # Use study_name
                    # Save using write_image
                    try:
                        fig.write_image(str(save_path))
                        logger.info(f"  Report: Saved plot '{name}'")
                    except ValueError as ve:
                        if "kaleido" in str(ve).lower(): logger.error(f"  Report: Failed to save '{name}': Kaleido missing/broken. Skipping.")
                        else: logger.error(f"  Report: ValueError saving '{name}': {ve}. Skipping.")
                    except Exception as e_write: logger.error(f"  Report: Error saving '{name}': {e_write}. Skipping.")

                except (ValueError, TypeError) as plot_err: logger.warning(f"  Report: Cannot generate plot '{name}': {plot_err}. Skipping.")
                except ImportError as imp_err: logger.warning(f"  Report: Cannot generate plot '{name}', missing dependency: {imp_err}. Skipping.")
                except Exception as e_gen: logger.error(f"  Report: Unexpected error generating plot '{name}': {e_gen}", exc_info=True)

            logger.info(f"Visualization report generation finished.")

        except ImportError: logger.error("Optuna library seems missing? Cannot generate report.")
        except Exception as e: logger.error(f"Error generating visualization report: {e}", exc_info=True)

    # --- New Method ---
    def start_dashboard_background(self, port=8080):
        """Determines DB path and starts Optuna Dashboard in background."""
        logger.info("Preparing to launch Optuna Dashboard in background...")
        storage_uri = None
        storage_path = None

        try:
            # --- Determine database filename based on CURRENT config ---
            opt_mode = self.cfg.get("optimization_mode", "unknown_mode")
            merge_method_name = "N/A"
            if opt_mode == "merge": merge_method_name = self.cfg.get("merge_method", "unknown_method")
            elif opt_mode == "recipe":
                recipe_path_str = self.cfg.recipe_optimization.get("recipe_path")
                merge_method_name = f"recipe_{Path(recipe_path_str).stem}" if recipe_path_str else "recipe"
            elif opt_mode == "layer_adjust": merge_method_name = "layer_adjust"

            scorers_list = self.cfg.get("scorer_method", ["unknown_scorer"])
            if isinstance(scorers_list, str): scorers_list = [scorers_list]
            scorer_name_part = "_".join(sorted(scorers_list))
            def sanitize(name): return "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in str(name))
            db_filename_base = f"optuna_{sanitize(opt_mode)}_{sanitize(merge_method_name)}_{sanitize(scorer_name_part)}"
            db_filename = f"{db_filename_base}.db"
            # --- End db filename logic ---

            if not hasattr(self, 'optuna_storage_dir') or not isinstance(self.optuna_storage_dir, Path):
                 raise ValueError("Optuna storage directory not initialized correctly.")
            if not self.optuna_storage_dir.is_dir():
                logger.warning(f"Optuna storage directory {self.optuna_storage_dir} does not exist yet. Creating.")
                self.optuna_storage_dir.mkdir(parents=True, exist_ok=True)

            storage_path = self.optuna_storage_dir / db_filename
            storage_uri = f"sqlite:///{storage_path.resolve()}"
            # <<< ADDED: Explicit logging of target DB >>>
            logger.info(f"Determined database URI for dashboard: {storage_uri}")
            if not storage_path.exists():
                logger.warning(f"Target Optuna DB file {storage_path} doesn't exist yet. Dashboard might show empty study initially.")

        except Exception as e_name:
            logger.error(f"Failed to determine Optuna DB path for background dashboard: {e_name}")
            return None

        # Launch dashboard using the determined URI via the helper
        return run_dashboard_in_background(storage_uri, port)


# --- Helper Function (can be outside the class) ---
def run_dashboard_in_background(storage_uri, port):
    """Run the Optuna dashboard as a separate process that won't block."""
    print(f"\n{'=' * 80}")
    print(f"LAUNCHING OPTUNA DASHBOARD")
    print(f"{'=' * 80}")
    print(f"Access the dashboard at: http://localhost:{port}")
    print(f"The dashboard will run in the background.")
    print(f"(Check console running sd_optim.py for dashboard process status/errors on exit)")
    print(f"{'=' * 80}\n")
    # Small delay to potentially let prints appear before subprocess output might start
    time.sleep(0.5)

    # Command list for Popen
    cmd = ["optuna-dashboard", storage_uri, "--port", str(port)]

    try:
        # Launch without waiting, pipe output to avoid cluttering main console
        # Use creationflags on Windows to prevent a new console window flashing
        creationflags = 0
        if os.name == 'nt': # Windows
             creationflags = subprocess.CREATE_NO_WINDOW

        dashboard_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=creationflags # Prevent console window on Windows
        )
        logger.info(f"Launched background dashboard process (PID: {dashboard_process.pid})")
        return dashboard_process
    except FileNotFoundError:
        logger.error(f"Could not find '{cmd[0]}' command. Is optuna-dashboard installed and in PATH?")
        print(f"ERROR: Failed to launch dashboard - command '{cmd[0]}' not found.")
        return None
    except Exception as e:
         logger.error(f"Failed to launch dashboard process: {e}", exc_info=True)
         print(f"ERROR: Failed to launch dashboard process: {e}")
         return None