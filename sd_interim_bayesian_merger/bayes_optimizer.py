import os
import random
import logging
import json

from typing import Dict, List
from pathlib import Path
from bayes_opt import BayesianOptimization, Events, UtilityFunction
from bayes_opt.logger import JSONLogger
from bayes_opt.domain_reduction import SequentialDomainReductionTransformer
from hydra.core.hydra_config import HydraConfig
from scipy.stats import qmc

from sd_interim_bayesian_merger.optimizer import Optimizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class BayesOptimizer(Optimizer):
    bounds_transformer = SequentialDomainReductionTransformer()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.setup_logging()

    def setup_logging(self) -> None:
        """Initialize Bayesian optimization specific logging"""
        run_name = "-".join(self.merger.output_file.stem.split("-")[:-1])
        self.log_name = run_name
        self.log_file_path = Path(HydraConfig.get().runtime.output_dir, f"{self.log_name}.json")

        # Initialize with empty list - will be populated if loading previous data
        self.previous_iterations = []

        # First create a fresh logger
        self.logger = JSONLogger(path=str(self.log_file_path), reset=self.cfg.optimizer.reset_log_file)

        # Then load previous data if specified
        if self.cfg.optimizer.get("load_log_file"):
            try:
                if os.path.isfile(self.cfg.optimizer.load_log_file):
                    # Read previous log data
                    with open(self.cfg.optimizer.load_log_file, "r") as f:
                        self.previous_iterations = [json.loads(line) for line in f]

                    # Write previous data to new log file
                    with open(self.log_file_path, "w") as f:
                        for iteration_data in self.previous_iterations:
                            f.write(json.dumps(iteration_data) + "\n")

                    logger.info(
                        f"Loaded and transferred {len(self.previous_iterations)} iterations from {self.cfg.optimizer.load_log_file}")
                else:
                    logger.info(f"No previous log file found at {self.cfg.optimizer.load_log_file}")
            except Exception as e:
                logger.warning(f"Failed to load previous optimization data: {e}")

    def optimize(self) -> None:
        pbounds = self.init_params()
        logger.debug(f"Initial Parameter Bounds: {pbounds}")

        # Separate categorical and continuous bounds
        categorical_bounds = {}
        continuous_bounds = {}

        for param_name, bound in pbounds.items():
            if isinstance(bound, (list, tuple)) and len(bound) == 2:
                if all(isinstance(v, int) and v in [0, 1] for v in bound):
                    # Binary parameters become categorical
                    categorical_bounds[param_name] = ('0', '1')
                    pbounds[param_name] = ('0', '1')
                else:
                    # Other numeric bounds are continuous
                    continuous_bounds[param_name] = bound

        # Acquisition Function Configuration with Defaults
        acq_config = self.cfg.optimizer.get("acquisition_function", {})
        acquisition_function = UtilityFunction(
            kind=acq_config.get("kind", "ucb"),
            kappa=acq_config.get("kappa", 3.0),
            xi=acq_config.get("xi", 0.05),
            kappa_decay=acq_config.get("kappa_decay", 0.98),
            kappa_decay_delay=acq_config.get("kappa_decay_delay", self.cfg.optimizer.init_points)
        )

        self.optimizer = BayesianOptimization(
            f=self.sd_target_function,
            pbounds=pbounds,
            random_state=self.cfg.optimizer.random_state,
            bounds_transformer=self.bounds_transformer if self.cfg.optimizer.bounds_transformer else None,
        )

        # Load previous points into the optimizer if they exist
        if self.previous_iterations:
            try:
                for point in self.previous_iterations:
                    # Register points with the optimizer
                    self.optimizer.register(
                        params=point["params"],
                        target=point["target"]
                    )
                logger.info(f"Registered {len(self.previous_iterations)} previous points with the optimizer")
            except Exception as e:
                logger.warning(f"Failed to register previous points with optimizer: {e}")

        # Subscribe logger to capture new points
        self.optimizer.subscribe(Events.OPTIMIZATION_STEP, self.logger)
        init_points = self.cfg.optimizer.init_points

        # Skip sampling if init_points is 0
        if init_points > 0:
            sampler_type = self.cfg.optimizer.get("sampler", "random").lower()

            if sampler_type != "random" and continuous_bounds:
                n_samples = init_points

                # Select the appropriate sampler
                if continuous_bounds:
                    d = len(continuous_bounds)
                    if sampler_type == "latin_hypercube":
                        sampler = qmc.LatinHypercube(d=d, seed=self.cfg.optimizer.random_state)
                    elif sampler_type == "sobol":
                        sampler = qmc.Sobol(d=d, seed=self.cfg.optimizer.random_state)
                    elif sampler_type == "halton":
                        sampler = qmc.Halton(d=d, seed=self.cfg.optimizer.random_state)
                    else:
                        logger.warning(f"Unknown sampler type '{sampler_type}', falling back to random")
                        sampler_type = "random"

                    if sampler_type != "random":
                        continuous_samples = sampler.random(n_samples)
                        l_bounds = [b[0] for b in continuous_bounds.values()]
                        u_bounds = [b[1] for b in continuous_bounds.values()]
                        scaled_continuous = qmc.scale(continuous_samples, l_bounds, u_bounds)
                        continuous_param_names = list(continuous_bounds.keys())

                        # Generate random samples for categorical parameters
                        categorical_param_names = list(categorical_bounds.keys())

                        # Combine the samples
                        for i in range(n_samples):
                            params = {}

                            # Add continuous parameters from quasi-random sampler
                            if continuous_bounds:
                                continuous_values = scaled_continuous[i]
                                for name, value in zip(continuous_param_names, continuous_values):
                                    params[name] = value

                            # Add categorical parameters randomly
                            for name in categorical_param_names:
                                params[name] = float(random.choice(categorical_bounds[name]))

                            self.optimizer.probe(params=params, lazy=True)

                        init_points = 0

        self.optimizer.maximize(
            init_points=init_points,
            n_iter=self.cfg.optimizer.n_iters,
        )

    def postprocess(self) -> None:
        logger.info("\nRecap!")
        for i, res in enumerate(self.optimizer.res):
            logger.info(f"Iteration {i + 1}: \n\t{res}")

        self.artist.visualize_optimization()


def parse_scores(iterations: List[Dict]) -> List[float]:
    return [r["target"] for r in iterations]
