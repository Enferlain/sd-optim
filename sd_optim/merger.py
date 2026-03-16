# merger.py - Version 1.1 - initial changes
import logging
import sd_mecha
import torch
import safetensors
import safetensors.torch

from hydra.core.hydra_config import HydraConfig
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig
from sd_mecha import recipe_nodes
from sd_mecha.extensions.merge_methods import MergeMethod, RecipeNodeOrValue
from sd_mecha.recipe_nodes import ModelRecipeNode, RecipeNode, MergeRecipeNode
from sd_mecha.extensions import merge_methods  # Import model_configs

from sd_optim import utils
from sd_optim.bounds import BoundsInfo
from sd_optim.merge.artifacts import create_model_output_name as build_model_output_name
from sd_optim.merge.artifacts import save_recipe_artifacts as save_merge_artifacts
from sd_optim.merge.artifacts import serialize_and_save_recipe as serialize_merge_recipe
from sd_optim.merge.execution import execute_recipe
from sd_optim.merge.fallback import (
    build_recipe_for_artifacts,
    build_recipe_to_merge,
    get_models_for_fallback_lookup,
    resolve_fallback_node,
)
from sd_optim.merge.model_nodes import create_model_nodes
from sd_optim.merge.model_selection import (
    get_adapter_candidate_ids,
    get_conversion_context_node,
    get_model_config_cache_key,
    get_model_config_candidates_cached,
    is_adapter_model_config,
    select_base_model,
    validate_node_is_not_lora,
)
from sd_optim.merge.fallback import fallback_debug_logged as fallback_debug_logged
from sd_optim.merge.recipe_builder import (
    handle_delta_output,
    prepare_model_recipe_args,
    prepare_param_recipe_args,
    slice_models,
)

logger = logging.getLogger(__name__)
__all__ = ["Merger", "fallback_debug_logged"]

def _sanitize_recipe_text_for_deserialize(recipe_text: str) -> str:
    """
    Normalize recipe text before sd_mecha deserialization.

    sd_mecha's parser raises on empty lines, so we drop blank/whitespace-only lines.
    """
    return "\n".join(line for line in recipe_text.splitlines() if line.strip())

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
            self.models = self._create_model_nodes()

        logger.info("Merger initialized successfully with pre-loaded configs.")

    def _create_model_nodes(self) -> list[ModelRecipeNode]:
        return create_model_nodes(self)

    def create_model_output_name(
        self,
        iteration: int,
        best: bool = False,
        recipe_node: RecipeNode | None = None,
    ) -> Path:
        return build_model_output_name(self, iteration, best=best, recipe_node=recipe_node)

    def _get_model_config_cache_key(self, node: recipe_nodes.ModelRecipeNode) -> tuple[str, str]:
        return get_model_config_cache_key(self, node)

    def _get_model_config_candidates_cached(
        self,
        node: recipe_nodes.ModelRecipeNode,
    ) -> tuple[sd_mecha.extensions.model_configs.ModelConfig, ...]:
        return get_model_config_candidates_cached(self, node)

    def _is_adapter_model_config(self, config: sd_mecha.extensions.model_configs.ModelConfig) -> bool:
        return is_adapter_model_config(self, config)

    def _get_adapter_candidate_ids(self, node: recipe_nodes.ModelRecipeNode) -> tuple[str, ...]:
        return get_adapter_candidate_ids(self, node)

    def _select_base_model(self) -> recipe_nodes.ModelRecipeNode | None:
        return select_base_model(self)

    def _get_conversion_context_node(self) -> recipe_nodes.ModelRecipeNode:
        return get_conversion_context_node(self)

    # We also need a small, reusable validator for the LoRA check.
    def _validate_node_is_not_lora(self, node: recipe_nodes.ModelRecipeNode, raise_error: bool = True):
        validate_node_is_not_lora(self, node, raise_error=raise_error)

    def _slice_models(self, prepared_model_nodes: list[RecipeNodeOrValue], merge_method: MergeMethod) -> list[RecipeNodeOrValue]:
        return slice_models(self, prepared_model_nodes, merge_method)

    def _handle_delta_output(
        self,
        core_recipe_node: recipe_nodes.MergeRecipeNode,  # FIXED: Changed type hint
        base_model_node: ModelRecipeNode | None,
        merge_method: MergeMethod,
    ) -> recipe_nodes.RecipeNode:
        return handle_delta_output(self, core_recipe_node, base_model_node, merge_method)

    def _serialize_and_save_recipe(self, final_recipe_node: recipe_nodes.RecipeNode, model_path: Path):
        serialize_merge_recipe(self, final_recipe_node, model_path)

    def _get_models_for_fallback_lookup(self, final_recipe_node: recipe_nodes.RecipeNode) -> list[ModelRecipeNode]:
        return get_models_for_fallback_lookup(self, final_recipe_node)

    def _resolve_fallback_node(
        self,
        final_recipe_node: recipe_nodes.RecipeNode,
        *,
        log_resolution: bool = True,
    ) -> ModelRecipeNode | None:
        return resolve_fallback_node(self, final_recipe_node, log_resolution=log_resolution)

    def _build_recipe_to_merge(
        self,
        final_recipe_node: recipe_nodes.RecipeNode,
        *,
        log_resolution: bool = True,
    ) -> tuple[recipe_nodes.RecipeNode, ModelRecipeNode | None]:
        return build_recipe_to_merge(self, final_recipe_node, log_resolution=log_resolution)

    def _build_recipe_for_artifacts(self, final_recipe_node: recipe_nodes.RecipeNode) -> recipe_nodes.RecipeNode:
        return build_recipe_for_artifacts(self, final_recipe_node)

    def _prepare_model_recipe_args(
        self,
        initial_model_nodes: list[recipe_nodes.ModelRecipeNode],
        base_model_node: recipe_nodes.ModelRecipeNode | None,
        # FIX 1: Use the correct reference
        merge_method: merge_methods.MergeMethod,
    ) -> list[RecipeNodeOrValue]:
        return prepare_model_recipe_args(self, initial_model_nodes, base_model_node, merge_method)

    # V1.6 - Proper context for conversions
    def _prepare_param_recipe_args(
        self,
        params: dict[str, Any],  # Flat params from optimizer: {'OPT_PARAM_NAME': value}
        param_info: BoundsInfo,  # Metadata: {'OPT_PARAM_NAME': {'bounds': ..., 'strategy': ..., ...}}
        merge_method: MergeMethod,
    ) -> dict[str, RecipeNode]:
        return prepare_param_recipe_args(self, params, param_info, merge_method)

    # V1.2 - fallback_model_index for merge and recipe properly
    def _execute_recipe(
        self,
        final_recipe_node: recipe_nodes.RecipeNode,
        model_path: Path,
        cache_map: dict[recipe_nodes.MergeRecipeNode, dict] | None = None,
    ):
        execute_recipe(self, final_recipe_node, model_path, cache_map=cache_map)

    # This is our conductor function, now correctly implemented.
    def _save_recipe_etc(
        self,
        final_recipe_node: recipe_nodes.RecipeNode,
        model_path: Path,
        iteration: int,
    ):
        save_merge_artifacts(self, final_recipe_node, model_path, iteration)

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
            model_path = self.models_dir / f"merge_output_default_{cfg.merge_method}.safetensors"
            logger.warning(f"Using default output path: {model_path}")
            self.output_file = model_path  # Attempt to set it

        # --- Recipe Building ---
        logger.debug(f"Building merge recipe for method: {cfg.merge_method}")

        # 2. Resolve merge method
        merge_func = utils.resolve_merge_method(cfg.merge_method)  # Assumes utils exists

        # 3. Select base model (for delta subtraction, conversion context)
        base_model_node = self._select_base_model()

        logger.info(f"Selected base model node: {base_model_node.path if base_model_node else 'None'}")

        # 4. Prepare model input nodes (handles LoRA conversion, delta subtraction)
        prepared_model_nodes = self._prepare_model_recipe_args(self.models, base_model_node, merge_func)

        # 5. Slice model nodes if merge method has fixed arity
        sliced_model_nodes = self._slice_models(prepared_model_nodes, merge_func)

        # 6. Prepare parameter nodes using param_info for expansion
        param_nodes = self._prepare_param_recipe_args(
            params,
            param_info,
            merge_func,  # Pass metadata here
        )

        # 7. Build the core merge recipe node
        logger.info(f"Calling '{merge_func.identifier}' with {len(sliced_model_nodes)} model args, {len(param_nodes)} param nodes.")
        core_recipe_node = merge_func(*sliced_model_nodes, **param_nodes)

        # 8. Handle potential delta output (wrap with add_difference)
        final_recipe_node = self._handle_delta_output(core_recipe_node, base_model_node, merge_func)
        cache_map = utils.build_recipe_cache_map(final_recipe_node, cache)
        # --- End Recipe Building ---

        # 9. Optional steps (save recipe, code, add keys)
        self._save_recipe_etc(final_recipe_node, model_path, iteration)

        # 10. Execute the final recipe (includes fallback logic)
        self._execute_recipe(final_recipe_node, model_path, cache_map=cache_map)

        logger.info(f"Merge process completed. Output: {model_path}")
        return model_path

    def recipe_optimization(
        self,
        params: dict[str, Any],
        param_info: BoundsInfo,
        cache: dict | None,
        iteration: int,
    ) -> Path:
        """
        Orchestrates the optimization of a .mecha recipe.
        This "thinker" method validates, generates node objects, and then calls
        "doer" utilities to rewrite the recipe text and execute.
        """
        logger.info(f"--- Coordinating Recipe Optimization for Iteration {iteration} ---")
        cache = cache if cache is not None else {}
        recipe_cfg = self.cfg.recipe_optimization
        recipe_path = Path(recipe_cfg.recipe_path)
        original_recipe_text = recipe_path.read_text(encoding="utf-8")

        # Preprocess: insert conversion lines for LoRA/LyCORIS models so that
        # sd_mecha's merge-space validation passes (e.g., delta_widen expects delta).
        # original_recipe_text = utils.preprocess_recipe_merge_spaces(original_recipe_text)

        # --- Step 1: VALIDATION (The "Thinker" validates its own plan) ---
        target_node_ref = recipe_cfg.target_nodes
        target_node_idx = int(target_node_ref.strip("&"))

        # Deserialize once to get the target node for validation
        all_nodes_map = {
            i: sd_mecha.deserialize(original_recipe_text.split("\n")[: i + 2])
            for i in range(len(original_recipe_text.strip().split("\n")) - 1)
        }
        target_node = all_nodes_map.get(target_node_idx)

        if not isinstance(target_node, MergeRecipeNode):
            raise TypeError(f"Target node {target_node_ref} is not a merge node.")

        # --- FIX IS HERE! ---
        param_names = target_node.merge_method.get_param_names()
        valid_params = set(param_names.args) | set(param_names.kwargs.keys())
        # --- END FIX ---

        for param_name in recipe_cfg.target_params:
            if param_name not in valid_params:
                raise ValueError(f"Target parameter '{param_name}' not found in method '{target_node.merge_method.identifier}'.")
        logger.info("Pre-validation successful.")

        # Set output file path
        self.output_file = self.create_model_output_name(iteration=iteration, recipe_node=target_node)
        logger.info(f"Set output path for this iteration to: {self.output_file}")

        # --- Step 2: GENERATION (The "Thinker" creates the node objects) ---
        new_param_nodes = self._prepare_param_recipe_args(params, param_info, target_node.merge_method)

        # --- Step 3: SERIALIZATION (Call a simple utility) ---
        new_node_strings, param_to_replacement = utils.serialize_nodes_for_rewrite(new_param_nodes)

        # --- Step 4: REWRITING (Call the main "doer" utility) ---
        final_recipe_text = utils.rewrite_recipe_text(
            original_recipe_text=original_recipe_text,
            target_node_idx=target_node_idx,
            new_node_strings=new_node_strings,
            param_to_replacement=param_to_replacement,
        )

        # --- Step 5: EXECUTION ---
        try:
            final_recipe_node = sd_mecha.deserialize(_sanitize_recipe_text_for_deserialize(final_recipe_text))
        except Exception as e:
            debug_path = Path(HydraConfig.get().runtime.output_dir) / f"iteration_{iteration}_failed_recipe.mecha"
            debug_path.write_text(final_recipe_text, encoding="utf-8")
            raise ValueError(f"Final recipe deserialization failed. Saved debug recipe to {debug_path}: {e}") from e

        cache_map = utils.build_recipe_cache_map(final_recipe_node, cache)

        model_path = self.output_file
        self._save_recipe_etc(final_recipe_node, model_path, iteration)
        self._execute_recipe(final_recipe_node, model_path, cache_map=cache_map)

        logger.info(f"Recipe optimization coordination complete. Output: {model_path}")
        return model_path

    def layer_adjust(self, params: dict, cfg: DictConfig) -> Path:  # Takes params
        """Loads a model, applies layer adjustments, and saves the modified model."""
        # Ensure output_file is set correctly for the current iteration by the Optimizer
        output_path = self.output_file
        if not output_path:
            logger.error("Output file path not set in Merger before layer_adjust call.")
            # Create a fallback name if needed
            model_name_for_fallback = Path(cfg.model_paths[0]).stem if cfg.model_paths else "unknown_model"
            output_path = self.models_dir / f"layer_adjusted_{model_name_for_fallback}_fallback.safetensors"
            logger.warning(f"Using fallback output path: {output_path}")
            self.output_file = output_path

        # Determine model path: use first model from model_paths if not specified
        if not cfg.model_paths:
            raise ValueError("No model paths specified for layer adjustment.")

        model_path_str = cfg.model_paths[0]
        model_path = Path(model_path_str)

        # Try resolving the path relative to models_dir if it doesn't exist directly
        if not model_path.is_file():
            resolved_path = Path(cfg.models_dir) / model_path_str  # Use original string path
            if resolved_path.is_file():
                model_path = resolved_path
                logger.info(f"Resolved layer_adjust model path to: {model_path}")
            else:
                raise FileNotFoundError(f"Model for layer_adjust not found at '{model_path_str}' or '{resolved_path}'")

        # Load the model
        logger.info(f"Loading model for layer adjustment: {model_path}")
        try:
            if model_path.suffix == ".safetensors":
                # Load onto the specified device directly
                state_dict = safetensors.torch.load_file(model_path, device=cfg.device)
            # Add support for other formats if needed (e.g., .ckpt)
            elif model_path.suffix in (".ckpt", ".pth", ".pt"):
                state_dict = torch.load(model_path, map_location=cfg.device)
                # Handle potential nesting in checkpoint files
                state_dict = state_dict.get("state_dict", state_dict)
            else:
                raise ValueError(f"Unsupported file type for layer adjustment: {model_path.suffix}")
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}", exc_info=True)
            raise

        # Determine if the model is an SDXL model by checking for a characteristic key
        # Use 'any' for efficiency - stops searching once found
        is_xl_model = any("conditioner.embedders.1" in key for key in state_dict)
        logger.info(f"Determined model type for layer adjustment: {'SDXL' if is_xl_model else 'Non-SDXL'}")

        # Apply adjustments (Assuming utils.modify_state_dict handles the logic)
        logger.info("Applying layer adjustments...")
        try:
            # Pass the raw params dict directly
            modified_state_dict = utils.modify_state_dict(state_dict, params, is_xl_model)
        except Exception as e:
            logger.error(f"Error applying layer adjustments: {e}", exc_info=True)
            raise

        # Save the modified model
        logger.info(f"Saving adjusted model to {output_path}")
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
            # Save directly to the target device if needed, though safetensors handles this
            safetensors.torch.save_file(modified_state_dict, output_path)
        except Exception as e:
            logger.error(f"Failed to save adjusted model {output_path}: {e}", exc_info=True)
            raise

        logger.info(f"Layer adjusted model saved successfully to {output_path}")
        return output_path
