import os
from typing import Dict, List
import logging

from bayes_opt import BayesianOptimization, Events, UtilityFunction
from bayes_opt.util import load_logs
from bayes_opt.domain_reduction import SequentialDomainReductionTransformer
from omegaconf import OmegaConf
from scipy.stats import qmc

from sd_interim_bayesian_merger.bounds import Bounds
from sd_interim_bayesian_merger.optimizer import Optimizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class BayesOptimizer(Optimizer):
    bounds_transformer = SequentialDomainReductionTransformer()

    def optimize(self) -> None:
        pbounds = self.init_params()
        logger.info(f"Initial Parameter Bounds: {pbounds}")

        # Load and validate custom bounds
        custom_bounds = Bounds.validate_custom_bounds(self.cfg.optimization_guide.custom_bounds)

        # Apply custom bounds to the existing pbounds dictionary
        for param_name, bound in custom_bounds.items():
            for key in pbounds:
                if param_name in key:
                    pbounds[key] = bound

        # Acquisition Function Configuration with Defaults
        acq_config = self.cfg.optimizer.get("acquisition_function", {})  # Access acquisition function settings, defaulting to empty dict
        acquisition_function = UtilityFunction(
            kind=acq_config.get("kind", "ucb"),  # Default to UCB
            kappa=acq_config.get("kappa", 3.0),  # Default kappa for UCB
            xi=acq_config.get("xi", 0.05),  # Default xi for EI and PI
            kappa_decay=acq_config.get("kappa_decay", 0.98),
            kappa_decay_delay=acq_config.get("kappa_decay_delay", self.cfg.optimizer.init_points)
        )

        # TODO: fork bayesian-optimisation and add LHS
        self.optimizer = BayesianOptimization(
            f=self.sd_target_function,
            pbounds=pbounds,
            random_state=self.cfg.optimizer.random_state,
            bounds_transformer=self.bounds_transformer
            if self.cfg.optimizer.bounds_transformer
            else None,
        )

        # Load logs if a valid log file is specified in the configuration
        log_file_path = self.cfg.optimizer.get("load_log_file", None)
        if log_file_path and os.path.isfile(log_file_path):  # Check if the file exists
            try:
                load_logs(self.optimizer, logs=self.cfg.optimizer.load_log_file)
                logger.info(f"Loaded previous optimization data from {self.cfg.optimizer.load_log_file}")
            except Exception as e:
                logger.warning(f"Failed to load optimization logs: {e}")

        self.optimizer.subscribe(Events.OPTIMIZATION_STEP, self.logger)

        init_points = self.cfg.optimizer.init_points
        if self.cfg.optimizer.latin_hypercube_sampling:
            sampler = qmc.LatinHypercube(d=len(pbounds))
            samples = sampler.random(self.cfg.optimizer.init_points)
            l_bounds = [b[0] for b in pbounds.values()]
            u_bounds = [b[1] for b in pbounds.values()]
            scaled_samples = qmc.scale(samples, l_bounds, u_bounds)

            for sample in scaled_samples.tolist():
                params = dict(zip(pbounds, sample))
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
