from __future__ import annotations

import logging
import random
from typing import Any

from bayes_opt import UtilityFunction
from bayes_opt.domain_reduction import SequentialDomainReductionTransformer
from omegaconf import DictConfig
from scipy.stats import qmc

logger = logging.getLogger(__name__)


def build_acquisition_function(cfg: DictConfig) -> UtilityFunction:
    """Build the BayesOpt acquisition function from config."""
    acq_config = cfg.optimizer.bayes_config.get("acquisition_function", {})
    return UtilityFunction(
        kind=acq_config.get("kind", "ucb"),
        kappa=acq_config.get("kappa", 3.0),
        xi=acq_config.get("xi", 0.05),
        kappa_decay=acq_config.get("kappa_decay", 0.98),
        kappa_decay_delay=acq_config.get("kappa_decay_delay", cfg.optimizer.init_points),
    )


def build_bounds_transformer(cfg: DictConfig) -> SequentialDomainReductionTransformer | None:
    """Build the optional bounds transformer when enabled."""
    bt_config = cfg.optimizer.bayes_config.get("bounds_transformer", {})
    if not bt_config.get("enabled", False):
        return None

    return SequentialDomainReductionTransformer(
        gamma_osc=bt_config.get("gamma_osc", 0.65),
        gamma_pan=bt_config.get("gamma_pan", 0.9),
        eta=bt_config.get("eta", 0.83),
        minimum_window=bt_config.get("minimum_window", 0.15),
    )


def split_parameter_bounds(
    optimizer_pbounds: dict[str, tuple[float, float] | float | int | list[Any]],
) -> tuple[dict[str, tuple[float, float] | float | int | list[Any]], dict[str, tuple[float, float]]]:
    """Split continuous bounds from binary-style categorical tuples used by the legacy Bayes path."""
    continuous_bounds: dict[str, tuple[float, float] | float | int | list[Any]] = {}
    categorical_params: dict[str, tuple[float, float]] = {}

    for name, bound in optimizer_pbounds.items():
        if isinstance(bound, tuple) and bound in {(0.0, 1.0), (1.0, 0.0)}:
            categorical_params[name] = bound
        else:
            continuous_bounds[name] = bound

    return continuous_bounds, categorical_params


def probe_initial_points(
    optimizer: Any,
    *,
    optimizer_pbounds: dict[str, tuple[float, float] | float | int | list[Any]],
    remaining_init_points: int,
    sampler_type: str,
    random_state: int,
) -> int:
    """Probe quasi-random initial points and return how many random init points remain."""
    continuous_bounds, categorical_params = split_parameter_bounds(optimizer_pbounds)
    if sampler_type == "random" or not continuous_bounds or remaining_init_points <= 0:
        return remaining_init_points

    n_samples = remaining_init_points
    dimension = len(continuous_bounds)
    try:
        if sampler_type == "latin_hypercube":
            sampler = qmc.LatinHypercube(d=dimension, seed=random_state)
        elif sampler_type == "sobol":
            sampler = qmc.Sobol(d=dimension, seed=random_state)
        elif sampler_type == "halton":
            sampler = qmc.Halton(d=dimension, seed=random_state)
        else:
            logger.warning("Unknown sampler type '%s', falling back to random sampling for init points", sampler_type)
            return remaining_init_points

        continuous_samples = sampler.random(n_samples)
        lower_bounds = [bound[0] for bound in continuous_bounds.values()]
        upper_bounds = [bound[1] for bound in continuous_bounds.values()]
        scaled_continuous = qmc.scale(continuous_samples, lower_bounds, upper_bounds)
        continuous_param_names = list(continuous_bounds.keys())

        logger.info("Probing %s initial points using %s sampler...", n_samples, sampler_type)
        for sample in scaled_continuous:
            params = dict(zip(continuous_param_names, sample, strict=False))
            for name, bound in categorical_params.items():
                params[name] = random.choice([bound[0], bound[1]])
            optimizer.probe(params=params, lazy=True)
        return 0
    except Exception as error:
        logger.error("Error during initial sampling: %s. Falling back to random.", error, exc_info=True)
        return n_samples
