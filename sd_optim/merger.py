import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sd_mecha

from omegaconf import DictConfig
from sd_mecha.recipe_nodes import ModelRecipeNode, RecipeNode

from sd_optim.bounds import BoundsInfo
from sd_optim.guide_runtime import GraphRuntimeBundle
from sd_optim.merge.artifacts import create_model_output_name as build_model_output_name
from sd_optim.merge.fallback import fallback_debug_logged
from sd_optim.merge.layer_adjust import layer_adjust as run_layer_adjust
from sd_optim.merge.model_nodes import create_model_nodes
from sd_optim.merge.recipe_optimization import recipe_optimization as run_recipe_optimization
from sd_optim.merge.runtime import run_merge

logger = logging.getLogger(__name__)
__all__ = ["Merger", "fallback_debug_logged"]
GuideRuntimeInput = BoundsInfo | GraphRuntimeBundle

@dataclass
class Merger:
    def __init__(
        self,
        cfg: DictConfig,
        base_model_config: sd_mecha.extensions.model_configs.ModelConfig,
        custom_block_config: sd_mecha.extensions.model_configs.ModelConfig | None,
        models_dir: Path,
    ):
        self.cfg = cfg
        self.base_model_config = base_model_config
        self.custom_block_config = custom_block_config
        self.models_dir = models_dir

        self.output_file: Path | None = None
        self.best_output_file: Path | None = None
        self._model_config_candidates_cache: dict[tuple[str, str], tuple[sd_mecha.extensions.model_configs.ModelConfig, ...]] = {}

        self.models: list[ModelRecipeNode] = []
        if self.cfg.optimization_mode == "merge":
            logger.info("Creating model nodes for 'merge' mode.")
            self.models = create_model_nodes(self)

        logger.info("Merger initialized successfully with pre-loaded configs.")

    def create_model_output_name(
        self,
        iteration: int,
        best: bool = False,
        recipe_node: RecipeNode | None = None,
    ) -> Path:
        return build_model_output_name(self, iteration, best=best, recipe_node=recipe_node)

    def merge(
        self,
        params: dict[str, Any],
        param_info: GuideRuntimeInput,
        cache: dict | None,
        iteration: int = 0,
    ) -> Path:
        """Build and execute an sd-mecha merge recipe."""
        return run_merge(self, params, param_info, cache, iteration)

    def recipe_optimization(
        self,
        params: dict[str, Any],
        param_info: GuideRuntimeInput,
        cache: dict | None,
        iteration: int,
    ) -> Path:
        return run_recipe_optimization(self, params, param_info, cache, iteration)

    def layer_adjust(self, params: dict, cfg: DictConfig) -> Path:  # Takes params
        """Loads a model, applies layer adjustments, and saves the modified model."""
        return run_layer_adjust(self, params, cfg)
