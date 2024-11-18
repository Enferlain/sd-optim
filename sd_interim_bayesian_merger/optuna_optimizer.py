import os
import logging
from typing import Dict, List
import optuna
from optuna.samplers import TPESampler, RandomSampler
from optuna.trial import Trial
from optuna.study import Study
import json

import sd_mecha
from sd_interim_bayesian_merger.bounds import Bounds
from sd_interim_bayesian_merger.optimizer import Optimizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class OptunaOptimizer(Optimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.study = None
        self._param_bounds = None
        self.log_name = self.cfg.get("log_name", "default")  # Add default log name

        # Initialize logger for trials
        self.logger = self._setup_trial_logger()

    def _setup_trial_logger(self):
        """Setup logging for optimization trials."""
        import json
        from pathlib import Path

        class TrialLogger:
            def __init__(self, log_path):
                self.log_path = Path(log_path)
                self.log_path.parent.mkdir(parents=True, exist_ok=True)

            def log(self, data):
                mode = 'a' if self.log_path.exists() else 'w'
                with open(self.log_path, mode, encoding='utf-8') as f:
                    json.dump(data, f)
                    f.write('\n')

        return TrialLogger(Path(os.getcwd()) / f"trials_{self.log_name}.jsonl")

    def validate_optimizer_config(self) -> bool:
        required_fields = ['n_iters', 'init_points', 'random_state']
        return all(hasattr(self.cfg.optimizer, field) for field in required_fields)

    def optimize(self) -> None:
        self._param_bounds = self.init_params()
        logger.debug(f"Initial Parameter Bounds: {self._param_bounds}")

        # Configure sampler based on configuration
        sampler_config = self.cfg.optimizer.get("sampler", {})
        sampler_type = sampler_config.get("type", "tpe").lower()
        if sampler_type == "random":
            sampler = RandomSampler(seed=self.cfg.optimizer.random_state)
        else:  # default to TPE
            sampler = TPESampler(
                seed=self.cfg.optimizer.random_state,
                n_startup_trials=self.cfg.optimizer.init_points,
                multivariate=True
            )

        # Create or load study with error handling
        study_name = f"optimization_{self.log_name}"
        storage_path = os.path.join(os.getcwd(), f'{study_name}.db')
        storage = f"sqlite:///{storage_path}"

        try:
            load_if_exists = bool(self.cfg.optimizer.get("load_log_file", False))
            self.study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                sampler=sampler,
                direction="maximize",
                load_if_exists=load_if_exists
            )
        except Exception as e:
            logger.error(f"Failed to create/load study: {e}")
            # Fallback to in-memory storage if database fails
            logger.info("Falling back to in-memory storage")
            self.study = optuna.create_study(
                study_name=study_name,
                sampler=sampler,
                direction="maximize"
            )

        # Register callback for logging
        self.study.add_trial_callback(self._trial_callback)

        # Run optimization
        try:
            self.study.optimize(
                func=self._objective,
                n_trials=self.cfg.optimizer.n_iters + self.cfg.optimizer.init_points,
                show_progress_bar=True
            )
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise

    def _objective(self, trial: Trial) -> float:
        """Objective function for Optuna optimization."""
        # Convert param bounds to Optuna parameter suggestions
        params = {}
        for param_name, bounds in self._param_bounds.items():
            if isinstance(bounds, (list, tuple)):
                if all(isinstance(v, int) and v in [0, 1] for v in bounds):
                    # Binary parameter
                    params[param_name] = trial.suggest_categorical(param_name, [0.0, 1.0])
                else:
                    # Continuous parameter
                    params[param_name] = trial.suggest_float(param_name, bounds[0], bounds[1])
            else:
                # Fixed value
                params[param_name] = bounds

        return self.sd_target_function(**params)

    def _trial_callback(self, study: Study, trial: Trial) -> None:
        """Callback to log trial information."""
        log_data = {
            "target": trial.value,
            "params": trial.params,
            "datetime": {
                "datetime": trial.datetime_start.isoformat(),
            }
        }

        # Write to the JSON logger
        self.logger.log(log_data)

    def postprocess(self) -> None:
        logger.info("\nOptimization Results Recap!")

        # Log all trials
        for i, trial in enumerate(self.study.trials):
            logger.info(f"Trial {i + 1}:")
            logger.info(f"\tValue: {trial.value}")
            logger.info(f"\tParameters: {trial.params}")

        # Log best trial
        logger.info("\nBest Trial:")
        logger.info(f"Value: {self.study.best_value}")
        logger.info(f"Parameters: {self.study.best_params}")

        # Create visualization
        self.artist.visualize_optimization()


def parse_scores(iterations: List[Dict]) -> List[float]:
    return [r["target"] for r in iterations]