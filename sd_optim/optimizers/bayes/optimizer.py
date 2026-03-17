from __future__ import annotations

import asyncio
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any

from bayes_opt import BayesianOptimization, Events
from bayes_opt.logger import JSONLogger
from hydra.core.hydra_config import HydraConfig

from sd_optim.artist import Artist
from sd_optim.core.optimizer_base import Optimizer
from sd_optim.optimizers.bayes.history import load_previous_iterations, register_previous_iterations
from sd_optim.optimizers.bayes.reporting import build_artist_series, log_postprocess_summary
from sd_optim.optimizers.bayes.sampling import (
    build_acquisition_function,
    build_bounds_transformer,
    probe_initial_points,
)

logger = logging.getLogger(__name__)


class BayesOptimizer(Optimizer):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.artist = Artist(self)
        self.optimizer: BayesianOptimization | None = None
        self.optimization_start_time: float | None = None
        self.previous_iterations: list[dict[str, Any]] = []

        checkpoint_root = self.cfg.optimizer.get("checkpoint_dir", os.getcwd())
        self.checkpoint_dir = Path(checkpoint_root) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_interval = self.cfg.optimizer.get("checkpoint_interval", 10)
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Initialize BayesOpt JSONL logging and any optional resume state."""
        self.log_name = self.cfg.get("log_name", "default_bayes_run")
        self.log_file_path = Path(HydraConfig.get().runtime.output_dir, f"{self.log_name}.json")

        bayes_config = self.cfg.optimizer.get("bayes_config", {})
        self.json_logger = JSONLogger(
            path=str(self.log_file_path),
            reset=bayes_config.get("reset_log_file", False),
        )
        self.previous_iterations = load_previous_iterations(self.cfg, target_log_path=self.log_file_path)

    async def optimize(self) -> None:
        self.optimization_start_time = time.time()
        logger.debug("Initial Parameter Bounds: %s", self.optimizer_pbounds)

        acquisition_function = build_acquisition_function(self.cfg)
        bounds_transformer = build_bounds_transformer(self.cfg)

        def sync_target_function_wrapper(**params_dict: Any) -> float:
            try:
                result = asyncio.run(self.sd_target_function(params_dict))
            except Exception as error:
                logger.error("Error in target function execution: %s", error, exc_info=True)
                return -float("inf")
            return -float("inf") if result is None else float(result)

        self.optimizer = BayesianOptimization(
            f=sync_target_function_wrapper,
            pbounds=self.optimizer_pbounds,
            random_state=self.cfg.optimizer.random_state,
            bounds_transformer=bounds_transformer,
        )

        if self.previous_iterations:
            try:
                loaded_count = register_previous_iterations(self.optimizer, self.previous_iterations)
                logger.info("Registered %s unique previous points with the optimizer", loaded_count)
            except Exception as error:
                logger.warning("Failed to register previous points with optimizer: %s", error)

        self.optimizer.subscribe(Events.OPTIMIZATION_STEP, self.json_logger)

        def checkpoint_subscriber(event: Any, instance: BayesianOptimization) -> None:
            del event
            iteration = len(instance.res)
            if iteration % self.checkpoint_interval == 0 and iteration > 0:
                logger.info("Creating periodic checkpoint at iteration %s", iteration)
                self.save_checkpoint(instance)

        self.optimizer.subscribe(Events.OPTIMIZATION_STEP, checkpoint_subscriber)

        init_points = self.cfg.optimizer.init_points
        completed_trials = len(self.optimizer.res)
        remaining_init_points = max(0, init_points - completed_trials)

        sampler_type = self.cfg.optimizer.bayes_config.get("sampler", "random").lower()
        remaining_init_points = probe_initial_points(
            self.optimizer,
            optimizer_pbounds=self.optimizer_pbounds,
            remaining_init_points=remaining_init_points,
            sampler_type=sampler_type,
            random_state=self.cfg.optimizer.random_state,
        )

        total_iterations = self.cfg.optimizer.n_iters
        remaining_iterations = max(0, total_iterations - max(0, completed_trials - init_points))

        if remaining_iterations > 0 or remaining_init_points > 0:
            logger.info(
                "Starting optimization: %s random init points, %s optimization iterations.",
                remaining_init_points,
                remaining_iterations,
            )
            try:
                await asyncio.to_thread(
                    self.optimizer.maximize,
                    init_points=remaining_init_points,
                    n_iter=remaining_iterations,
                    acquisition_function=acquisition_function,
                )
            except KeyboardInterrupt:
                logger.info("Optimization interrupted by user. Saving current state...")
                self.save_checkpoint(self.optimizer)
            except Exception:
                self.save_checkpoint(self.optimizer)
                raise
        else:
            logger.info("All optimization iterations already completed. Skipping maximize.")

        self.save_checkpoint(self.optimizer)

    def save_checkpoint(self, optimizer_instance: BayesianOptimization | None) -> None:
        """Save current optimization state using pickle."""
        if optimizer_instance is None:
            logger.warning("No optimizer instance to checkpoint")
            return

        checkpoint_file = self.checkpoint_dir / f"bayesopt_checkpoint_{self.log_name}.pkl"
        try:
            with open(checkpoint_file, "wb") as file:
                pickle.dump(optimizer_instance, file)
            logger.info("Saved Bayesian Optimization checkpoint to %s", checkpoint_file)
        except Exception as error:
            logger.error("Failed to save Bayesian Optimization checkpoint: %s", error, exc_info=True)

    async def postprocess(self) -> None:
        logger.info("\nBayesOpt Recap!")

        if not self.optimizer or not hasattr(self.optimizer, "res") or not self.optimizer.res:
            logger.warning("No Bayesian Optimization results found to process or display.")
            return

        results = self.optimizer.res
        log_postprocess_summary(results, self.optimization_start_time)

        if hasattr(self.optimizer, "max") and self.optimizer.max:
            best_trial = self.optimizer.max
            logger.info("\nBest BayesOpt Trial Found:")
            logger.info("\tTarget: %.4f", best_trial["target"])
            param_str = ", ".join(
                f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
                for key, value in best_trial["params"].items()
            )
            logger.info("\tParams: {%s}", param_str)
        else:
            logger.warning("Could not determine the best trial from optimizer results.")

        if not hasattr(self, "artist"):
            logger.error("Artist object not found on BayesOptimizer. Cannot generate plot.")
            return

        iterations, scores, best_scores = build_artist_series(results)
        self.artist.iterations = iterations
        self.artist.scores = scores
        self.artist.best_scores = best_scores

        logger.info("Generating convergence plot via Artist...")
        try:
            await self.artist.visualize_optimization()
        except Exception as error:
            logger.error("Failed to create visualization via Artist: %s", error, exc_info=True)

    def get_best_parameters(self) -> dict[str, Any]:
        """Return best parameters found during optimization."""
        return self.optimizer.max["params"] if self.optimizer and self.optimizer.max else {}

    def get_optimization_history(self) -> list[dict[str, Any]]:
        """Return history of optimization attempts."""
        return self.optimizer.res if self.optimizer else []

    def validate_optimizer_config(self) -> bool:
        """Validate optimizer-specific configuration."""
        required_fields = ["n_iters", "init_points", "random_state"]
        valid = all(hasattr(self.cfg.optimizer, field) for field in required_fields)

        if not valid:
            missing = [field for field in required_fields if not hasattr(self.cfg.optimizer, field)]
            logger.error("Missing required configuration fields for BayesOpt: %s", missing)

        return valid
