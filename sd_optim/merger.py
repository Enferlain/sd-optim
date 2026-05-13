# merger.py - Version 1.1 - initial changes
import logging
import sd_mecha
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig
from sd_mecha.recipe_nodes import ModelRecipeNode, RecipeNode

from sd_optim.bounds import BoundsInfo
from sd_optim.merge.artifacts import create_model_output_name as build_model_output_name
from sd_optim.merge.artifacts import save_recipe_artifacts as save_merge_artifacts
from sd_optim.merge.execution import execute_recipe
from sd_optim.merge.fallback import fallback_debug_logged
from sd_optim.merge.layer_adjust import layer_adjust as run_layer_adjust
from sd_optim.merge.model_nodes import create_model_nodes
from sd_optim.merge.model_selection import select_base_model
from sd_optim.merge.recipe_builder import (
    handle_delta_output,
    prepare_model_recipe_args,
    prepare_param_recipe_args,
    slice_models,
)
from sd_optim.merge.recipe_optimization import recipe_optimization as run_recipe_optimization
from sd_optim.utils.methods import resolve_merge_method
from sd_optim.utils.recipes import build_recipe_cache_map

logger = logging.getLogger(__name__)
__all__ = ["Merger", "fallback_debug_logged"]

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

    # V1.1 - Accepts param_info metadata
    def merge(
        self,
        params: dict[str, Any],  # Flat params from optimizer
        param_info: BoundsInfo,  # <<< ADDED: Full metadata from ParameterHandler
        cache: dict | None,
        iteration: int = 0,  # <<< ADD iteration parameter
    ) -> Path:
        """Builds and executes sd-mecha recipe, using param_info for expansion."""
        cfg = self.cfg
        cache = cache if cache is not None else {}
        logger.info(f"Starting merge process for iteration {iteration}")  # <<< USE iteration

        # 1. Determine output path (using instance property self.output_file)
        model_path = self.output_file
        if not model_path:  # Safety check
            logger.error("Output file path not set in Merger before merge call.")
            # Define a default path or raise error
            model_path = self.models_dir / f"merge_output_default_{cfg.merge.merge_method}.safetensors"
            logger.warning(f"Using default output path: {model_path}")
            self.output_file = model_path  # Attempt to set it

        # --- Recipe Building ---
        logger.debug(f"Building merge recipe for method: {cfg.merge.merge_method}")

        # 2. Resolve merge method
        merge_func = resolve_merge_method(cfg.merge.merge_method)

        # 3. Select base model (for delta subtraction, conversion context)
        base_model_node = select_base_model(self)

        logger.info(f"Selected base model node: {base_model_node.path if base_model_node else 'None'}")

        # 4. Prepare model input nodes (handles LoRA conversion, delta subtraction)
        prepared_model_nodes = prepare_model_recipe_args(self, self.models, base_model_node, merge_func)

        # 5. Slice model nodes if merge method has fixed arity
        sliced_model_nodes = slice_models(self, prepared_model_nodes, merge_func)

        # 6. Prepare parameter nodes using param_info for expansion
        param_nodes = prepare_param_recipe_args(
            self,
            params,
            param_info,
            merge_func,  # Pass metadata here
        )

        # 7. Build the core merge recipe node
        logger.info(f"Calling '{merge_func.identifier}' with {len(sliced_model_nodes)} model args, {len(param_nodes)} param nodes.")
        core_recipe_node = merge_func(*sliced_model_nodes, **param_nodes)

        # 8. Handle potential delta output (wrap with add_difference)
        final_recipe_node = handle_delta_output(self, core_recipe_node, base_model_node, merge_func)
        cache_map = build_recipe_cache_map(final_recipe_node, cache)
        # --- End Recipe Building ---

        # 9. Optional steps (save recipe, code, add keys)
        save_merge_artifacts(self, final_recipe_node, model_path, iteration)

        # 10. Execute the final recipe (includes fallback logic)
        execute_recipe(self, final_recipe_node, model_path, cache_map=cache_map)

        logger.info(f"Merge process completed. Output: {model_path}")
        return model_path

    def recipe_optimization(
        self,
        params: dict[str, Any],
        param_info: BoundsInfo,
        cache: dict | None,
        iteration: int,
    ) -> Path:
        return run_recipe_optimization(self, params, param_info, cache, iteration)

    def layer_adjust(self, params: dict, cfg: DictConfig) -> Path:  # Takes params
        """Loads a model, applies layer adjustments, and saves the modified model."""
        return run_layer_adjust(self, params, cfg)
