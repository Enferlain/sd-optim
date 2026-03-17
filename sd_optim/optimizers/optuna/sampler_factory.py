from __future__ import annotations

import logging
import os
import warnings
from typing import Any

from omegaconf import DictConfig, ListConfig
from optuna.samplers import (
    CmaEsSampler,
    GPSampler,
    GridSampler,
    NSGAIISampler,
    QMCSampler,
    RandomSampler,
    TPESampler,
)

logger = logging.getLogger(__name__)


def validate_optimizer_config(cfg: DictConfig) -> bool:
    """Validate optimizer-specific configuration."""
    required_fields = ["n_iters", "init_points", "random_state"]
    valid = all(hasattr(cfg.optimizer, field) for field in required_fields)

    if not valid:
        missing = [field for field in required_fields if not hasattr(cfg.optimizer, field)]
        logger.error("Missing required configuration fields: %s", missing)
        return False

    sampler_config = cfg.optimizer.optuna_config.get("sampler", {})
    sampler_type = sampler_config.get("type", "tpe").lower()
    if sampler_type == "grid" and "search_space" not in sampler_config:
        logger.error("Grid sampler selected but 'search_space' is missing in optimizer.sampler config.")
        valid = False

    if cfg.optimizer.optuna_config.get("use_pruning", False):
        pruner_type = cfg.optimizer.optuna_config.get("pruner_type", "median").lower()
        if pruner_type not in ["median", "successive_halving"]:
            logger.warning("Unknown pruner_type '%s'. Optuna might default or error.", pruner_type)

    return valid


def configure_sampler(cfg: DictConfig) -> Any:
    """Configure and return the Optuna sampler based on the current config."""
    sampler_config = cfg.optimizer.optuna_config.get("sampler", {})
    sampler_type = sampler_config.get("type", "tpe").lower()
    seed = cfg.optimizer.random_state

    if seed == -1:
        seed = None

    if seed is None:
        seed = int.from_bytes(os.urandom(4), "big")
        logger.info("Random seed was not provided (-1 or null). Generated a new seed: %s", seed)

    sampler_kwargs = {"seed": seed}
    logger.info("Configuring sampler: type='%s', seed=%s", sampler_type, seed)

    if sampler_type == "random":
        sampler = RandomSampler(**sampler_kwargs)
        logger.info("Using Random Sampler")

    elif sampler_type == "tpe":
        tpe_kwargs = {
            "n_startup_trials": cfg.optimizer.init_points,
            "multivariate": sampler_config.get("multivariate", True),
            "group": sampler_config.get("group", False),
            "warn_independent_sampling": sampler_config.get("warn_independent_sampling", True),
            "constant_liar": sampler_config.get("constant_liar", False),
            "n_ei_candidates": sampler_config.get("n_ei_candidates", 24),
            "prior_weight": sampler_config.get("prior_weight", 1.0),
            "consider_magic_clip": sampler_config.get("consider_magic_clip", True),
            "consider_endpoints": sampler_config.get("consider_endpoints", False),
            **sampler_kwargs,
        }
        gamma = sampler_config.get("gamma", None)
        if gamma is not None:
            tpe_kwargs["gamma"] = gamma

        sampler = TPESampler(**tpe_kwargs)
        logger.info("Using TPE Sampler with options: %s", tpe_kwargs)

    elif sampler_type == "cmaes":
        requested_restart_strategy = sampler_config.get("restart_strategy", None)
        requested_inc_popsize = sampler_config.get("inc_popsize", None)

        if requested_restart_strategy is not None or (
            requested_inc_popsize is not None and requested_inc_popsize != -1
        ):
            logger.warning(
                "CMA-ES restarts are not supported by core Optuna (deprecated since 4.4.0; removal scheduled for 6.0.0). "
                "Ignoring sampler.restart_strategy=%r and sampler.inc_popsize=%r. "
                "If you need restart strategies, use OptunaHub's RestartCmaEsSampler.",
                requested_restart_strategy,
                requested_inc_popsize,
            )

        cmaes_kwargs: dict[str, Any] = {
            "n_startup_trials": cfg.optimizer.init_points,
            "sigma0": sampler_config.get("sigma0", None),
            "warn_independent_sampling": sampler_config.get("warn_independent_sampling", True),
            "popsize": sampler_config.get("popsize", None),
            "use_separable_cma": sampler_config.get("use_separable_cma", False),
            "lr_adapt": sampler_config.get("lr_adapt", False),
            "x0": sampler_config.get("x0", None),
            "consider_pruned_trials": sampler_config.get("consider_pruned_trials", False),
            "with_margin": sampler_config.get("with_margin", False),
            **sampler_kwargs,
        }

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*use_separable_cma.*", category=Warning)
            sampler = CmaEsSampler(**cmaes_kwargs)
        logger.info("Using CMA-ES Sampler with options: %s", cmaes_kwargs)

    elif sampler_type == "gp":
        gp_kwargs = {
            "n_startup_trials": cfg.optimizer.init_points,
            **sampler_kwargs,
        }
        sampler = GPSampler(**gp_kwargs)
        logger.info("Using Gaussian Process (GP) Sampler with options: %s", gp_kwargs)

    elif sampler_type == "qmc":
        qmc_kwargs = {
            "qmc_type": sampler_config.get("qmc_type", "sobol"),
            "scramble": sampler_config.get("scramble", True),
            "warn_independent_sampling": sampler_config.get("warn_independent_sampling", True),
            "warn_asynchronous_seeding": sampler_config.get("warn_asynchronous_seeding", True),
            **sampler_kwargs,
        }
        sampler = QMCSampler(**qmc_kwargs)
        logger.info("Using QMC Sampler with options: %s", qmc_kwargs)

    elif sampler_type == "grid":
        if "search_space" not in sampler_config:
            raise ValueError("Grid sampler requires a 'search_space' configuration in optimizer.sampler")
        search_space = {
            key: list(value) if isinstance(value, (list, ListConfig)) else value
            for key, value in sampler_config["search_space"].items()
        }
        sampler = GridSampler(search_space)
        logger.info("Using Grid Sampler with search space: %s", search_space)

    elif sampler_type == "nsgaii":
        nsgaii_kwargs = {
            "population_size": sampler_config.get("population_size", 50),
            "mutation_prob": sampler_config.get("mutation_prob", None),
            "crossover_prob": sampler_config.get("crossover_prob", 0.9),
            "swapping_prob": sampler_config.get("swapping_prob", 0.5),
            "warn_independent_sampling": sampler_config.get("warn_independent_sampling", True),
            "constraints_func": sampler_config.get("constraints_func", None),
            "elite_population_selection_strategy": sampler_config.get("elite_population_selection_strategy", None),
            "child_generation_strategy": sampler_config.get("child_generation_strategy", None),
            "after_trial_strategy": sampler_config.get("after_trial_strategy", None),
            "seed": sampler_kwargs.get("seed"),
        }
        crossover_config = sampler_config.get("crossover", None)
        if crossover_config is not None:
            try:
                from optuna.samplers.nsgaii import BaseCrossover

                if isinstance(crossover_config, str):
                    import optuna.samplers.nsgaii as nsgaii_module

                    crossover_class = getattr(nsgaii_module, crossover_config)
                    nsgaii_kwargs["crossover"] = crossover_class()
                elif isinstance(crossover_config, BaseCrossover):
                    nsgaii_kwargs["crossover"] = crossover_config
            except (ImportError, AttributeError) as error:
                logger.warning("Could not set crossover '%s': %s", crossover_config, error)
        sampler = NSGAIISampler(**nsgaii_kwargs)
        logger.warning("Using NSGA-II Sampler - primarily for multi-objective optimization.")
        logger.info("NSGA-II options: %s", nsgaii_kwargs)

    else:
        if sampler_type != "tpe":
            logger.warning("Unknown sampler type: '%s', falling back to TPE.", sampler_type)
        sampler = TPESampler(
            n_startup_trials=cfg.optimizer.init_points,
            multivariate=True,
            **sampler_kwargs,
        )
        logger.info("Using TPE Sampler (Fallback)")

    return sampler
