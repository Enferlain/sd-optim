# optuna_optimizer.py - Version 1.6 - objective fixes

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
import asyncio

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

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
        self.study_name = None  # Will be set in optimize()

        # --- Determine Optuna Storage Directory ---
        try:
            # 1. Determine Project Root (assuming this file is in sd_optim/)
            project_root = Path(__file__).parent.parent.resolve()  # sd_optim/ -> sd-optim/

            # 2. Define Default Path (relative to project root)
            default_storage_path = project_root / "optuna_db"  # e.g., D:\...\sd-optim\optuna_db

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
            raise  # Re-raise critical errors

        # Early stopping settings
        self.early_stopping = self.cfg.optimizer.optuna_config.get("early_stopping", False)
        self.patience = self.cfg.optimizer.optuna_config.get("patience", 10)
        self.min_improvement = self.cfg.optimizer.optuna_config.get("min_improvement", 0.001)
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
                        json.dump(data, f)  # Type: ignore
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
        sampler_config = self.cfg.optimizer.optuna_config.get("sampler", {})
        sampler_type = sampler_config.get("type", "tpe").lower()
        if sampler_type == "grid" and "search_space" not in sampler_config:
            logger.error("Grid sampler selected but 'search_space' is missing in optimizer.sampler config.")
            valid = False
        # Add more sampler-specific checks if needed

        # Validate pruner config (if pruning enabled)
        if self.cfg.optimizer.get("use_pruning", False):
            pruner_type = self.cfg.optimizer.optuna_config.get("pruner_type", "median").lower()
            if pruner_type not in ["median", "successive_halving"]:
                logger.warning(f"Unknown pruner_type '{pruner_type}'. Optuna might default or error.")
            # Add pruner-specific checks if needed

        return valid

    def _configure_sampler(self):
        """Configure and return the appropriate sampler based on configuration."""
        sampler_config = self.cfg.optimizer.optuna_config.get("sampler", {})
        sampler_type = sampler_config.get("type", "tpe").lower()
        seed = self.cfg.optimizer.random_state

        # Optuna samplers expect None for a random seed, not -1, which can cause errors with numpy.
        if seed == -1:
            seed = None
        
        # If seed is None, generate a random one for reproducibility logging
        if seed is None:
            # Use os.urandom to generate a cryptographically secure 32-bit integer.
            # This avoids the np.random.randint overflow on some systems for 2**32-1.
            seed = int.from_bytes(os.urandom(4), 'big')
            logger.info(f"Random seed was not provided (-1 or null). Generated a new seed: {seed}")

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
                "group": sampler_config.get("group", False),
                "warn_independent_sampling": sampler_config.get("warn_independent_sampling", True),
                "constant_liar": sampler_config.get("constant_liar", False),
                # --- ADDED PARAMETERS ---
                "n_ei_candidates": sampler_config.get("n_ei_candidates", 24),
                "prior_weight": sampler_config.get("prior_weight", 1.0),
                "consider_magic_clip": sampler_config.get("consider_magic_clip", True),
                "consider_endpoints": sampler_config.get("consider_endpoints", False),
                **sampler_kwargs
            }

            # Only add gamma if it's not None
            gamma = sampler_config.get("gamma", None)
            if gamma is not None:
                tpe_kwargs["gamma"] = gamma

            sampler = TPESampler(**tpe_kwargs)
            logger.info(f"Using TPE Sampler with options: {tpe_kwargs}")

        elif sampler_type == "cmaes":
            # CMA-ES is good for continuous, non-linear problems
            cmaes_kwargs = {
                "n_startup_trials": self.cfg.optimizer.init_points,
                "restart_strategy": sampler_config.get("restart_strategy", None),
                "sigma0": sampler_config.get("sigma0", None),
                "warn_independent_sampling": sampler_config.get("warn_independent_sampling", True),
                # --- ADDED PARAMETERS ---
                "popsize": sampler_config.get("popsize", None),  # Population size
                "inc_popsize": sampler_config.get("inc_popsize", 2),  # Population increase factor for restarts
                "use_separable_cma": sampler_config.get("use_separable_cma", False),  # For high dimensions
                "lr_adapt": sampler_config.get("lr_adapt", False),  # Learning rate adaptation
                # --- NEWLY ADDED PARAMETERS from Optuna documentation ---
                "x0": sampler_config.get("x0", None),  # Initial parameter values
                "consider_pruned_trials": sampler_config.get("consider_pruned_trials", False),  # Consider pruned trials
                "with_margin": sampler_config.get("with_margin", False),  # Use CMA-ES with margin
                # Note: source_trials is not configurable from YAML as it requires actual FrozenTrial objects
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
                "qmc_type": sampler_config.get("qmc_type", "sobol"),  # 'sobol', 'halton', 'lhs'
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
                # --- NEWLY ADDED PARAMETERS from Optuna documentation ---
                "constraints_func": sampler_config.get("constraints_func", None),  # Constraint function for optimization
                "elite_population_selection_strategy": sampler_config.get("elite_population_selection_strategy", None),  # Elite selection strategy
                "child_generation_strategy": sampler_config.get("child_generation_strategy", None),  # Child generation strategy
                "after_trial_strategy": sampler_config.get("after_trial_strategy", None),  # After trial strategy
                "seed": sampler_kwargs.get("seed")  # Use only the seed from sampler_kwargs
            }
            # Handle crossover separately as it requires specific import
            crossover_config = sampler_config.get("crossover", None)
            if crossover_config is not None:
                try:
                    # Import the crossover class dynamically
                    from optuna.samplers.nsgaii import BaseCrossover
                    if isinstance(crossover_config, str):
                        # Try to get the crossover class by name
                        import optuna.samplers.nsgaii as nsgaii_module
                        crossover_class = getattr(nsgaii_module, crossover_config)
                        nsgaii_kwargs["crossover"] = crossover_class()
                    elif isinstance(crossover_config, BaseCrossover):
                        nsgaii_kwargs["crossover"] = crossover_config
                except (ImportError, AttributeError) as e:
                    logger.warning(f"Could not set crossover '{crossover_config}': {e}")
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
            if sampler_type != "tpe":  # Avoid duplicate warning if default is used
                logger.warning(f"Unknown sampler type: '{sampler_type}', falling back to TPE.")
            sampler = TPESampler(
                n_startup_trials=self.cfg.optimizer.init_points,
                multivariate=True,
                **sampler_kwargs
            )
            logger.info("Using TPE Sampler (Fallback)")

        return sampler

    async def optimize(self) -> None:
        """Run Optuna optimization process with explicit resume/fork controls."""
        self.optimization_start_time = time.time()
        logger.debug(f"Initial Parameter Bounds: {self.optimizer_pbounds}")

        # Configure sampler and pruner
        sampler = self._configure_sampler()
        pruner = None  # Define pruner here, inside the if block
        if self.cfg.optimizer.optuna_config.get("use_pruning", False):
            pruner_type = self.cfg.optimizer.optuna_config.get("pruner_type", "median")
            if pruner_type == "median":
                pruner = optuna.pruners.MedianPruner(n_startup_trials=self.cfg.optimizer.init_points)
            elif pruner_type == "successive_halving":
                pruner = optuna.pruners.SuccessiveHalvingPruner()
            if pruner: logger.info(f"Using {pruner_type.capitalize()} Pruner")

        # --- Start of New Logic Integration ---
        optuna_cfg = self.cfg.optimizer.optuna_config
        parent_study_name_to_load = optuna_cfg.get("resume_from_study")
        should_fork_study = optuna_cfg.get("fork_study", False)

        # PATH A: Starting a brand-new study
        if not parent_study_name_to_load:
            logger.info("No 'resume_from_study' specified. Creating a brand-new study.")
            storage_uri, db_filename = self._get_storage_uri_for_new_study()
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            sampler_name = self.cfg.optimizer.optuna_config.get("sampler", {}).get("type", "tpe")
            self.study_name = f"run_{timestamp}_{sampler_name}"

            self.study = optuna.create_study(
                study_name=self.study_name, storage=storage_uri, sampler=sampler, pruner=pruner, direction="maximize",
                load_if_exists=False
            )
            logger.info(f"Successfully created new study '{self.study_name}' in '{db_filename}'.")
            self._set_initial_study_attributes()

        # PATH B: Resuming or Forking
        else:
            logger.info(f"Attempting to load parent study '{parent_study_name_to_load}'.")
            parent_storage_uri, parent_db_filename = self._find_db_for_study(parent_study_name_to_load)
            if not parent_storage_uri:
                raise ValueError(f"Could not find study '{parent_study_name_to_load}' to resume/fork.")

            parent_study = optuna.load_study(study_name=parent_study_name_to_load, storage=parent_storage_uri)

            # Sub-path B1: FORKING
            if should_fork_study:
                logger.info(f"Forking study '{parent_study_name_to_load}' into a new study.")
                new_storage_uri, new_db_filename = self._get_storage_uri_for_new_study(is_fork=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                self.study_name = f"fork_{parent_study_name_to_load}_{timestamp}"

                self.study = optuna.create_study(
                    study_name=self.study_name, storage=new_storage_uri, sampler=sampler, pruner=pruner,
                    direction="maximize"
                )

                completed_trials = [t for t in parent_study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                for trial in completed_trials: self.study.enqueue_trial(trial.params)

                logger.info(
                    f"Created fork '{self.study_name}' in '{new_db_filename}' and enqueued {len(completed_trials)} parent trials.")
                self._set_initial_study_attributes(parent_name=parent_study_name_to_load)

            # Sub-path B2: RESUMING
            else:
                logger.info(f"Resuming study '{parent_study_name_to_load}' directly.")

                # 1. Get parent scorers, ensuring they are strings.
                parent_scorers_raw = parent_study.user_attrs.get("config_scorers", [])
                parent_scorers = set(map(str, parent_scorers_raw))

                # 2. Get current scorers and flatten them.
                current_scorers_raw = self.cfg.get("scorer_method", [])
                
                def flatten_scorers(scorers_obj):
                    if isinstance(scorers_obj, (list, ListConfig)):
                        for item in scorers_obj:
                            yield from flatten_scorers(item)
                    else:
                        yield str(scorers_obj)

                current_scorers = set(flatten_scorers(current_scorers_raw))

                if parent_scorers != current_scorers:
                    # Convert current_scorers_raw to a more readable format for the error message
                    current_scorers_for_error = list(flatten_scorers(current_scorers_raw))
                    raise ValueError(
                        f"FATAL: Cannot resume study '{parent_study_name_to_load}' because scorers have changed. "
                        f"(Study used {sorted(list(parent_scorers))}, config has {sorted(current_scorers_for_error)}). "
                        "To proceed, set 'fork_study: true'."
                    )

                self.study = parent_study
                # --- FIX: Do NOT replace the sampler on resume ---
                # The existing sampler in the study already contains the history. Replacing it resets the state.
                # self.study.sampler = sampler
                logger.info(f"Re-using existing sampler of type '{type(self.study.sampler).__name__}' from the loaded study.")
                # The pruner is part of the study's permanent configuration and should not be changed on resume.
                # self.study.pruner = pruner
                self.study_name = parent_study_name_to_load
                # --- NEW: Set the completed trials count on the base class ---
                self.completed_trials = len(parent_study.trials)
                logger.info(f"Setting iteration offset to {self.completed_trials} to account for existing trials.")

        # --- Common logic from here ---
        self._setup_logger_path()
        completed_trials = len(self.study.trials)

        # --- NEW: Pre-populate logger with existing trials on resume ---
        if completed_trials > 0:
            logger.info(f"Writing {len(self.study.trials)} existing trials to the new log file...")
            for existing_trial in self.study.trials:
                # Use the callback to ensure consistent logging format
                self._trial_callback(self.study, existing_trial)
        # --- END NEW ---
        total_trials_planned = self.cfg.optimizer.init_points + self.cfg.optimizer.n_iters
        remaining_trials = max(0, total_trials_planned - completed_trials)

        if completed_trials > 0:
            # --- NEW: Detailed trial breakdown ---
            n_startup_trials = self.cfg.optimizer.init_points
            remaining_startup_trials = max(0, n_startup_trials - completed_trials)
            remaining_exploration_trials = max(0, remaining_trials - remaining_startup_trials)

            logger.info(f"Study has {completed_trials} existing trials. {remaining_trials} new trials to run.")
            if remaining_startup_trials > 0:
                logger.info(f"  ({remaining_startup_trials} init trials + {remaining_exploration_trials} optimization trials)")
            else:
                logger.info(f"  ({remaining_exploration_trials} optimization trials)")
            # --- END NEW ---
            if self.study.best_trial: self.best_rolling_score = max(self.best_rolling_score, self.study.best_value)

        if remaining_trials > 0:
            logger.info(f"Starting optimization for {remaining_trials} new trials.")
            try:
                await asyncio.to_thread(
                    self.study.optimize, func=self._objective, n_trials=remaining_trials,
                    n_jobs=optuna_cfg.get("n_jobs", 1), show_progress_bar=True, callbacks=[self._trial_callback]
                )
            except (KeyboardInterrupt, Exception) as e:
                logger.error(f"Optimization loop stopped: {e}",
                             exc_info=True if not isinstance(e, KeyboardInterrupt) else False)
                if not isinstance(e, KeyboardInterrupt):
                    raise
        else:
            logger.info("All planned trials already completed. Skipping optimization.")

        self._analyze_parameter_importance()

    def _restore_from_trials_log(self):
        """Restore trials from the JSON log file if database is unavailable."""
        if not self.study:
            logger.error("Cannot restore trials because study is not initialized.")
            return
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
        """
        Objective function for Optuna, supporting advanced parameter suggestions like
        log scale, step, categorical, and continuous ranges.
        """
        params: Dict[str, Any] = {}
        if not self.optimizer_pbounds:
            logger.error("Optimizer bounds not initialized in _objective. Cannot suggest parameters.")
            raise optuna.exceptions.TrialPruned("Bounds not available")

        for param_name, bound_config in self.optimizer_pbounds.items():
            try:
                # Case 1: Rich dictionary format (e.g., {"range": (0, 1), "log": True})
                # This is for our new advanced settings.
                if isinstance(bound_config, dict) and "range" in bound_config:
                    low, high = bound_config["range"]
                    log = bound_config.get("log", False)
                    step = bound_config.get("step", None)

                    # Check if it should be an integer suggestion
                    is_integer_range = isinstance(low, int) and isinstance(high, int) and (
                                step is None or isinstance(step, int))

                    if is_integer_range:
                        params[param_name] = trial.suggest_int(param_name, low, high, step=step or 1, log=log)
                        logger.debug(
                            f"Suggesting for '{param_name}': Int range [{low}-{high}], Step={step or 1}, Log={log}")
                    else:  # Otherwise, it's a float
                        params[param_name] = trial.suggest_float(param_name, float(low), float(high), step=step,
                                                                 log=log)
                        logger.debug(
                            f"Suggesting for '{param_name}': Float range [{low}-{high}], Step={step}, Log={log}")

                # Case 2: List format (always for categorical choices)
                # In YAML: `param: [value1, value2, value3]`
                elif isinstance(bound_config, list):
                    params[param_name] = trial.suggest_categorical(param_name, bound_config)
                    logger.debug(f"Suggesting for '{param_name}': Categorical {bound_config}")

                # Case 3: Tuple format (for simple continuous ranges, backward compatibility)
                # In YAML: `param: (0.0, 1.0)`
                elif isinstance(bound_config, tuple):
                    if len(bound_config) != 2: raise ValueError("Range tuple must have 2 values.")
                    low, high = bound_config
                    if isinstance(low, int) and isinstance(high, int):
                        params[param_name] = trial.suggest_int(param_name, low, high)
                        logger.debug(f"Suggesting for '{param_name}': Simple Int range [{low}-{high}]")
                    else:
                        params[param_name] = trial.suggest_float(param_name, float(low), float(high))
                        logger.debug(f"Suggesting for '{param_name}': Simple Float range [{low}-{high}]")

                # Case 4: Single number (fixed parameter, not optimized)
                elif isinstance(bound_config, (int, float)):
                    params[param_name] = bound_config
                    logger.debug(f"Using fixed value for '{param_name}': {bound_config}")

                else:
                    raise ValueError(f"Unsupported bounds format for '{param_name}': {bound_config}")

            except Exception as e_suggest:
                logger.error(
                    f"Error during parameter suggestion for '{param_name}' with config {bound_config}: {e_suggest}",
                    exc_info=True)
                raise  # Re-raise to stop the trial, as it's a config error.

        try:
            # Run the async function in a synchronous context
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

    def _get_storage_uri_for_new_study(self, is_fork: bool = False) -> Tuple[str, str]:
        """
        Determines the database filename and URI based on the current
        run's configuration. This is used for creating NEW studies or FORKS.
        """
        try:
            opt_mode = self.cfg.get("optimization_mode", "unknown_mode")
            merge_method_name = "N/A"
            if opt_mode == "merge":
                merge_method_name = self.cfg.get("merge_method", "unknown_method")
            elif opt_mode == "recipe":
                recipe_path_str = self.cfg.recipe_optimization.get("recipe_path")
                if recipe_path_str:
                    merge_method_name = f"recipe_{Path(recipe_path_str).stem}"
                else:
                    merge_method_name = "recipe"
            elif opt_mode == "layer_adjust":
                merge_method_name = "layer_adjust"

            # --- THIS IS THE FIX ---
            scorers_raw = self.cfg.get("scorer_method", [])
            # 1. Ensure it's a list-like object (handles single string case)
            scorers_list_like = scorers_raw if isinstance(scorers_raw, (list, ListConfig)) else [scorers_raw]
            # 2. Convert every item to a string and then sort
            scorers_str_list = sorted([str(s) for s in scorers_list_like])
            # 3. Now, join the list of strings. This is safe!
            scorer_name_part = "_".join(scorers_str_list)

            # --- END OF FIX ---

            def sanitize(name):
                # This sanitize function can be improved to handle paths better
                safe_name = str(name).replace('\\', '/').split('/')[-1]  # Get filename from path
                return "".join(c for c in safe_name if c.isalnum() or c in ('_', '-'))

            fork_prefix = "fork_" if is_fork else ""

            # NEW, CORRECTED LINE:
            if opt_mode == 'recipe':
                # In recipe mode, merge_method_name already contains "recipe_filename", so we don't need opt_mode.
                db_filename_base = f"optuna_{fork_prefix}{sanitize(merge_method_name)}_{sanitize(scorer_name_part)}"
            else:
                # For all other modes (like 'merge'), we use the original logic.
                db_filename_base = f"optuna_{fork_prefix}{sanitize(opt_mode)}_{sanitize(merge_method_name)}_{sanitize(scorer_name_part)}"

            db_filename = f"{db_filename_base}.db"

            storage_path = self.optuna_storage_dir / db_filename
            storage_uri = f"sqlite:///{storage_path.resolve()}"

            logger.info(f"Determined storage for new study/fork: '{db_filename}'")
            return storage_uri, db_filename

        except Exception as e:
            logger.error(f"Failed to determine DB name, falling back to default: {e}", exc_info=True)
            db_filename = "optuna_fallback.db"
            storage_path = self.optuna_storage_dir / db_filename
            return f"sqlite:///{storage_path.resolve()}", db_filename

    def _find_db_for_study(self, study_name_to_find: str) -> Tuple[Optional[str], Optional[str]]:
        """Scans all .db files in the storage directory to find which one contains the target study."""
        logger.info(f"Searching for study '{study_name_to_find}' in all .db files...")
        for db_file in self.optuna_storage_dir.glob("*.db"):
            storage_uri = f"sqlite:///{db_file.resolve()}"
            try:
                # Get all study summaries from this DB without loading the whole study
                all_summaries = optuna.study.get_all_study_summaries(storage=storage_uri)
                for summary in all_summaries:
                    if summary.study_name == study_name_to_find:
                        logger.info(f"Found study '{study_name_to_find}' in database: '{db_file.name}'")
                        return storage_uri, db_file.name
            except Exception as e:
                logger.warning(f"Could not inspect database '{db_file.name}': {e}")
                continue
        logger.error(f"Study '{study_name_to_find}' was not found in any database in {self.optuna_storage_dir}")
        return None, None

    def _set_initial_study_attributes(self, parent_name: Optional[str] = None):
        """Sets user attributes for a newly created study or fork."""
        if not self.study:
            logger.warning("Cannot set initial attributes because study is not initialized.")
            return
        try:
            models_str = str([str(p) for p in self.cfg.get("model_paths", [])])
            self.study.set_user_attr('config_input_models', models_str)
            self.study.set_user_attr('config_base_model_index', self.cfg.get('base_model_index', -1))
            self.study.set_user_attr('config_optimization_mode', self.cfg.get('optimization_mode', 'N/A'))
            self.study.set_user_attr('config_scorers', list(self.cfg.get("scorer_method", [])))  # Store as list

            if self.cfg.get("optimization_mode") == "merge":
                self.study.set_user_attr('config_merge_method', self.cfg.get('merge_method', 'N/A'))

            if parent_name:
                self.study.set_user_attr('forked_from', parent_name)

            logger.info(f"Stored initial configuration as user attributes for new study '{self.study_name}'.")
        except Exception as e:
            logger.warning(f"Could not store config as study attributes: {e}")

    def _setup_logger_path(self):
        """Sets up the path for the .jsonl trial logger."""
        try:
            hydra_run_path = Path(HydraConfig.get().runtime.output_dir)
            trials_log_filename = f"{self.study_name}_trials.jsonl"
            trials_log_path = hydra_run_path / trials_log_filename
            self.logger.set_path(trials_log_path)
        except Exception as e:
            logger.error(f"Failed to set trial logger path: {e}")

    # --- MODIFIED: _create_progress_plots ---
    def _create_progress_plots(self):
        """Create plot showing optimization progress using Optuna's Plotly backend."""
        if not self.study or not self.study.trials:
            logger.warning("_create_progress_plots: No study or trials available.")
            return

        # Check if enough trials exist for the plot
        completed_trials = [t for t in self.study.trials if t.state == TrialState.COMPLETE and t.value is not None]
        if len(completed_trials) < 1:  # History plot needs at least one point
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
                if "kaleido" in str(ve).lower():
                    logger.error(f"Failed to save periodic plot: Kaleido missing/broken.")
                else:
                    logger.error(f"ValueError saving periodic plot: {ve}.")
            except Exception as e_write:
                logger.error(f"Error saving periodic plot: {e_write}.")

        except ImportError:
            logger.error("Optuna visualization module not available for periodic plots.")
        except Exception as e:
            logger.error(f"Failed to create periodic progress plot: {e}", exc_info=True)

    # --- MODIFIED: _analyze_parameter_importance ---
    def _analyze_parameter_importance(self):
        """Analyze and plot importance of parameters using Optuna's functions."""
        # This method might be redundant if postprocess already generates the importance plot.
        # Keeping it updated for consistency or potential separate use.
        if not self.study or len(self.study.trials) < 2:  # Importance usually needs >= 2 trials
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
                return  # Cannot plot if calculation fails

            # --- Generate plot using Optuna's function ---
            fig = vis.plot_param_importances(self.study)

            # Determine output directory using Hydra
            try:
                run_dir = Path(HydraConfig.get().runtime.output_dir)
            except ValueError:
                logger.error("_analyze_parameter_importance: Hydra context unavailable. Cannot determine save path.")
                return
            output_dir = run_dir / "visualizations"  # Save with other postprocess plots
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save plot using write_image
            save_path = output_dir / f"optuna_param_importances_{self.study_name}.png"
            try:
                fig.write_image(str(save_path))
                logger.info(f"Saved parameter importance plot to {save_path}")
            except ValueError as ve:
                if "kaleido" in str(ve).lower():
                    logger.error(f"Failed to save importance plot: Kaleido missing/broken.")
                else:
                    logger.error(f"ValueError saving importance plot: {ve}.")
            except Exception as e_write:
                logger.error(f"Error saving importance plot: {e_write}.")

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
                    fig = plot_func(self.study)  # Generate Plotly figure
                    save_path = output_dir / f"optuna_{name}_{self.study_name}.png"

                    # Save using write_image (requires kaleido)
                    try:
                        fig.write_image(str(save_path))
                        logger.info(f"  Successfully saved Optuna plot: {name}")
                    except ValueError as ve:
                        if "kaleido" in str(ve).lower():
                            logger.error(
                                f"  Failed to save Plotly plot '{name}': Kaleido engine not found or not functional. Please install with 'pip install -U kaleido'. Skipping save.")
                        else:
                            logger.error(f"  ValueError saving Plotly plot '{name}': {ve}. Skipping save.")
                    except Exception as e_write:
                        logger.error(f"  Unexpected error saving Plotly plot '{name}': {e_write}. Skipping save.")

                except (ValueError, TypeError) as plot_err:
                    # Handle errors like not enough parameters for contour, etc.
                    logger.warning(f"  Could not generate Optuna plot '{name}': {plot_err}. Skipping.")
                except ImportError as imp_err:
                    # Handle missing optional dependencies like scikit-learn for importance
                    logger.warning(
                        f"  Could not generate Optuna plot '{name}' due to missing dependency: {imp_err}. Skipping.")
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
                output_dir = Path("./optuna_visualizations_report")  # Example fallback
                logger.warning(f"Using fallback directory for report: {output_dir}")
            else:
                output_dir = run_dir / "visualizations_report"  # Use a separate folder? Or same as postprocess?

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
                    save_path = output_dir / f"optuna_{name}_{self.study_name}.png"  # Use study_name
                    # Save using write_image
                    try:
                        fig.write_image(str(save_path))
                        logger.info(f"  Report: Saved plot '{name}'")
                    except ValueError as ve:
                        if "kaleido" in str(ve).lower():
                            logger.error(f"  Report: Failed to save '{name}': Kaleido missing/broken. Skipping.")
                        else:
                            logger.error(f"  Report: ValueError saving '{name}': {ve}. Skipping.")
                    except Exception as e_write:
                        logger.error(f"  Report: Error saving '{name}': {e_write}. Skipping.")

                except (ValueError, TypeError) as plot_err:
                    logger.warning(f"  Report: Cannot generate plot '{name}': {plot_err}. Skipping.")
                except ImportError as imp_err:
                    logger.warning(f"  Report: Cannot generate plot '{name}', missing dependency: {imp_err}. Skipping.")
                except Exception as e_gen:
                    logger.error(f"  Report: Unexpected error generating plot '{name}': {e_gen}", exc_info=True)

            logger.info(f"Visualization report generation finished.")

        except ImportError:
            logger.error("Optuna library seems missing? Cannot generate report.")
        except Exception as e:
            logger.error(f"Error generating visualization report: {e}", exc_info=True)

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
            if opt_mode == "merge":
                merge_method_name = self.cfg.get("merge_method", "unknown_method")
            elif opt_mode == "recipe":
                # We get the filename stem, or use a default if the path is missing.
                recipe_path_str = self.cfg.recipe_optimization.get("recipe_path")
                merge_method_name = Path(recipe_path_str).stem if recipe_path_str else "unknown-recipe"
                # We NO LONGER add the "recipe_" prefix here!
            elif opt_mode == "layer_adjust":
                merge_method_name = "layer_adjust"

            scorers_list = self.cfg.get("scorer_method", ["unknown_scorer"])
            if isinstance(scorers_list, str):
                scorers_list = [scorers_list]
            scorer_name_part = "_".join(sorted(scorers_list))

            def sanitize(name):
                return "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in str(name))

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
                logger.warning(
                    f"Target Optuna DB file {storage_path} doesn't exist yet. Dashboard might show empty study initially.")

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
        if os.name == 'nt':  # Windows
            creationflags = subprocess.CREATE_NO_WINDOW

        dashboard_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=creationflags  # Prevent console window on Windows
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
