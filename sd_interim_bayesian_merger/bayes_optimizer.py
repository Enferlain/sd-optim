from typing import Dict, List
import logging

from bayes_opt import BayesianOptimization, Events
from bayes_opt.domain_reduction import SequentialDomainReductionTransformer
from scipy.stats import qmc

from sd_interim_bayesian_merger.optimizer import Optimizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class BayesOptimizer(Optimizer):
    bounds_transformer = SequentialDomainReductionTransformer()

    def optimize(self) -> None:
        pbounds = self.init_params()
        logger.info(f"Initial Parameter Bounds: {pbounds}")

        # TODO: fork bayesian-optimisation and add LHS
        self.optimizer = BayesianOptimization(
            f=self.sd_target_function,
            pbounds=pbounds,
            random_state=1,
            bounds_transformer=self.bounds_transformer
            if self.cfg.bounds_transformer
            else None,
        )

        self.optimizer.subscribe(Events.OPTIMIZATION_STEP, self.logger)

        init_points = self.cfg.init_points
        if self.cfg.latin_hypercube_sampling:
            sampler = qmc.LatinHypercube(d=len(pbounds))
            samples = sampler.random(self.cfg.init_points)
            l_bounds = [b[0] for b in pbounds.values()]
            u_bounds = [b[1] for b in pbounds.values()]
            scaled_samples = qmc.scale(samples, l_bounds, u_bounds)

            for sample in scaled_samples.tolist():
                params = dict(zip(pbounds, sample))
                self.optimizer.probe(params=params, lazy=True)

            init_points = 0

        self.optimizer.maximize(
            init_points=init_points,
            n_iter=self.cfg.n_iters,
        )

    def postprocess(self) -> None:
        logger.info("\nRecap!")
        for i, res in enumerate(self.optimizer.res):
            logger.info(f"Iteration {i + 1}: \n\t{res}")  # Add 1 to the iteration number

        # No need to assign scores, best_weights, or best_bases here

        self.artist.visualize_optimization()  # Call the Artist's visualize_optimization method


def parse_scores(iterations: List[Dict]) -> List[float]:
    return [r["target"] for r in iterations]
