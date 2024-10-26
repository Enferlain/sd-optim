import os
import random
from typing import Dict, List
import logging

import sd_mecha
from bayes_opt import BayesianOptimization, Events, UtilityFunction
from bayes_opt.util import load_logs
from bayes_opt.domain_reduction import SequentialDomainReductionTransformer
from scipy.stats import qmc

from sd_interim_bayesian_merger.bounds import Bounds
from sd_interim_bayesian_merger.optimizer import Optimizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class BayesOptimizer(Optimizer):
    bounds_transformer = SequentialDomainReductionTransformer()

    def optimize(self) -> None:
        pbounds = self.init_params()
        logger.debug(f"Initial Parameter Bounds: {pbounds}")

        # Separate categorical and continuous bounds
        categorical_bounds = {}
        continuous_bounds = {}

        for param_name, bound in pbounds.items():
            if isinstance(bound, (list, tuple)) and len(bound) == 2:
                if all(isinstance(v, int) and v in [0, 1] for v in bound):
                    # Binary  parameters become  categorical
                    categorical_bounds[param_name] = ('0', '1')
                    pbounds[param_name] = ('0', '1')
                else:
                    # Other numeric  bounds are  continuous
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

        # Load logs if specified
        log_file_path = self.cfg.optimizer.get("load_log_file", None)
        if log_file_path and os.path.isfile(log_file_path):
            try:
                load_logs(self.optimizer, logs=self.cfg.optimizer.load_log_file)
                logger.info(f"Loaded previous optimization data from {self.cfg.optimizer.load_log_file}")
            except Exception as e:
                logger.warning(f"Failed to load optimization logs: {e}")

        self.optimizer.subscribe(Events.OPTIMIZATION_STEP, self.logger)
        init_points = self.cfg.optimizer.init_points

        sampler_type = self.cfg.optimizer.get("sampler", "random").lower()  # "random" is now the default

        if sampler_type != "random" and continuous_bounds:
            n_samples = self.cfg.optimizer.init_points

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
                    sampler_type = "random"  # Explicitly set to random on unknown type

                if sampler_type != "random":  # Only perform LHS if not "random"
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
