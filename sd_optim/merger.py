# merger.py - Version 1.1 - initial changes

# merger.py - Version 1.0
import logging
import sd_mecha
import torch
import os
import requests
import inspect
import safetensors
import safetensors.torch

from hydra.core.hydra_config import HydraConfig
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any, re

from omegaconf import DictConfig, open_dict
from sd_mecha import recipe_serializer, extensions, recipe_nodes
from sd_mecha.extensions.merge_methods import MergeMethod, RecipeNodeOrValue
from sd_mecha.recipe_nodes import RecipeNodeOrValue, ModelRecipeNode
from sd_mecha.extensions import model_configs # Import model_configs

# Assuming utils contains MergeMethodCodeSaver and add_extra_keys
from sd_optim import utils
# Assuming your custom methods are in MergeMethods and decorated correctly
from sd_optim.merge_methods import MergeMethods
from sd_optim.bounds import BoundsInfo


logger = logging.getLogger(__name__)

# Map precision strings to torch.dtype objects
precision_mapping = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "fp64": torch.float64,
}


@dataclass
class Merger:
    cfg: DictConfig
    models: List[ModelRecipeNode] = None # Store ModelRecipeNode objects
    models_dir: Path = None # ADDED: Instance attribute for the directory

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        if not cfg.model_paths: raise ValueError("'model_paths' cannot be empty.")
        self.models_dir = Path(cfg.model_paths[0]).resolve().parent
        self.models = self._create_model_nodes()
        self.base_model_config: Optional[model_configs.ModelConfig] = None
        self.custom_block_config: Optional[model_configs.ModelConfig] = None
        try:
             temp_nodes = [sd_mecha.model(p) for p in cfg.model_paths]
             with sd_mecha.open_input_dicts(temp_nodes, [self.models_dir]): # type: ignore
                 configs = sd_mecha.infer_model_configs(temp_nodes) # type: ignore
                 if configs: self.base_model_config = configs[0]
                 else: logger.error("Failed to infer base model config in Merger init.")
        except Exception as e: logger.error(f"Error inferring base config in Merger init: {e}")
        custom_id = self.cfg.optimization_guide.get("custom_block_config_id")
        if custom_id and self.base_model_config:
            try: self.custom_block_config = model_configs.resolve(custom_id)
            except ValueError as e: logger.warning(f"Merger could not resolve custom config '{custom_id}': {e}")
        self.output_file: Optional[Path] = None; self.best_output_file: Optional[Path] = None
        self.create_model_out_name(); self.create_best_model_out_name()

    def validate_config(self):
        # Removed model_arch check
        if self.cfg.optimization_mode == "merge":
            if not self.cfg.model_paths or len(self.cfg.model_paths) < 1: # Need at least 1 for merge
                raise ValueError(
                    "For 'merge' mode, 'model_paths' must contain at least one model path."
                )
            if not self.cfg.merge_method:
                 raise ValueError("Configuration missing required field: 'merge_method'")
        elif self.cfg.optimization_mode == "recipe":
            # Keep recipe validation as is for now
            if not self.cfg.recipe_optimization.recipe_path:
                raise ValueError("`recipe_optimization.recipe_path` must be specified.")
            if not self.cfg.recipe_optimization.target_nodes:
                raise ValueError("`recipe_optimization.target_nodes` must be specified.")
        elif self.cfg.optimization_mode == "layer_adjust":
            if not self.cfg.model_paths or len(self.cfg.model_paths) < 1:
                raise ValueError("`model_paths` must contain at least one model for 'layer_adjust' mode.")
        else:
            raise ValueError(f"Invalid optimization mode: {self.cfg.optimization_mode}")

        # Check precision settings existence
        if not hasattr(self.cfg, 'merge_dtype') or self.cfg.merge_dtype not in precision_mapping:
            raise ValueError(f"Invalid or missing 'merge_dtype': {self.cfg.get('merge_dtype')}. Must be one of {list(precision_mapping.keys())}")
        if not hasattr(self.cfg, 'save_dtype') or self.cfg.save_dtype not in precision_mapping:
             raise ValueError(f"Invalid or missing 'save_dtype': {self.cfg.get('save_dtype')}. Must be one of {list(precision_mapping.keys())}")

    def _create_model_nodes(self) -> List[ModelRecipeNode]:
        """Creates basic sd_mecha.model() nodes for all models in model_paths."""
        model_nodes = []
        # Use self.models_dir directly
        models_dir_path = self.models_dir.resolve()
        for model_path_str in self.cfg.get("model_paths", []):
            model_path = Path(model_path_str).resolve()
            try:
                # Use self.models_dir for relative path calculation
                relative_path = model_path.relative_to(models_dir_path)
            except ValueError:
                relative_path = model_path
                # Use self.models_dir in the warning
                logger.warning(f"Model {model_path} is outside of models_dir ({models_dir_path}). Using absolute path.")

            model_nodes.append(sd_mecha.model(str(relative_path)))
        logger.info(f"Created {len(model_nodes)} ModelRecipeNodes.")
        return model_nodes

    def _create_model_output_name(self, it: int = 0, best: bool = False) -> Path:
        """Generates the output file name for the merged model."""
        # Simplified logic for "merge" mode
        if self.cfg.optimization_mode == "merge":
            model_names = [Path(path).stem for path in self.cfg.model_paths]
            # Handle cases with single model or more models appropriately
            if len(model_names) == 1:
                 name_part = model_names[0]
            elif len(model_names) >= 2:
                 name_part = f"{model_names[0]}-{model_names[1]}" # Keep first two for consistency
            else:
                 name_part = "merged_model" # Fallback name
            merge_method_name = self.cfg.merge_method
            combined_name = f"{name_part}-{merge_method_name}-it_{it}"
        elif self.cfg.optimization_mode == "layer_adjust":
            model_name = Path(self.cfg.model_paths[0]).stem
            combined_name = f"layer_adjusted-{model_name}-it_{it}"
        elif self.cfg.optimization_mode == "recipe":
             # Keep recipe logic as is for now
            recipe_path = self.cfg.recipe_optimization.recipe_path
            recipe = sd_mecha.deserialize_path(recipe_path)
            model_names_recipe = utils.get_model_names_from_recipe(recipe) # Assuming utils.get_model_names_from_recipe exists
            if len(model_names_recipe) < 2: model_names_recipe.extend(["unknown"] * (2 - len(model_names_recipe)))
            target_node = self.cfg.recipe_optimization.target_nodes
            if isinstance(target_node, list): target_node = target_node[0]
            merge_method_name = utils.get_merge_method(recipe_path, target_node) # Assuming utils.get_merge_method exists
            combined_name = f"{model_names_recipe[0]}-{model_names_recipe[1]}-{merge_method_name}-it_{it}"
        else:
             raise ValueError(f"Invalid optimization mode for naming: {self.cfg.optimization_mode}")

        if best:
            combined_name += f"_best"

        # Ensure the output path is within the models directory
        output_dir = self.models_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f"{combined_name}.safetensors"

    def create_model_out_name(self, it: int = 0) -> None:
        self.output_file = self._create_model_output_name(it=it)

    def create_best_model_out_name(self, it: int = 0) -> None:
        self.best_output_file = self._create_model_output_name(it=it, best=True)

    def _select_base_model(self) -> Optional[ModelRecipeNode]:
        """Selects the base model node based on configuration index."""
        base_model_index = self.cfg.get("base_model_index", None)
        if base_model_index is None:
            return None # No base model needed or specified

        if not isinstance(base_model_index, int) or base_model_index < 0 or base_model_index >= len(self.models):
            raise ValueError(f"Invalid base_model_index: {base_model_index}. Must be an integer within range [0, {len(self.models) - 1}).")

        base_model_node = self.models[base_model_index]

        # Infer config to check if it's a LoRA - Requires sd_mecha context
        try:
            with sd_mecha.open_input_dicts(base_model_node, [Path(self.cfg.models_dir)]):
                 # A simple heuristic: check if common LoRA identifiers are in the config name
                 # This might need refinement based on how sd-mecha identifies LoRAs
                 if "lora" in base_model_node.model_config.identifier or \
                    "lycoris" in base_model_node.model_config.identifier:
                     raise ValueError(
                         f"The selected base model ({base_model_node.path}) appears to be a LoRA/LyCORIS based on its inferred config '{base_model_node.model_config.identifier}'. These cannot be used as base models."
                     )
        except Exception as e:
             logger.error(f"Error during base model config inference for LoRA check: {e}")
             # Decide whether to raise an error or proceed cautiously
             # raise ValueError(f"Could not infer configuration for base model {base_model_node.path} to check if it's a LoRA.") from e

        return base_model_node

    def _slice_models(self, prepared_model_nodes: List[RecipeNodeOrValue], merge_method: MergeMethod) -> List[RecipeNodeOrValue]:
        """Slices the model list to match the expected number for non-varargs methods."""
        param_info = merge_method.get_param_names()
        if param_info.has_varargs():
            return prepared_model_nodes # Method accepts variable args, no slicing needed

        expected_num_models = len(param_info.args) # Number of positional args is the expected count
        num_provided = len(prepared_model_nodes)

        if num_provided > expected_num_models:
            logger.warning(
                f"Merge method '{merge_method.identifier}' expects {expected_num_models} model arguments, "
                f"but {num_provided} were prepared. Using the first {expected_num_models}."
            )
            return prepared_model_nodes[:expected_num_models]
        elif num_provided < expected_num_models:
             # This case should ideally be caught earlier or handled by defaults
             logger.warning(
                f"Merge method '{merge_method.identifier}' expects {expected_num_models} model arguments, "
                f"but only {num_provided} were prepared. This might lead to errors."
            )
        return prepared_model_nodes

    def _handle_delta_output(
            self,
            core_recipe_node: recipe_nodes.MergeRecipeNode,  # FIXED: Changed type hint
            base_model_node: Optional[ModelRecipeNode],
            merge_method: MergeMethod
    ) -> recipe_nodes.RecipeNode:
        """Wraps the core recipe with add_difference if the output is a delta and a base model exists."""
        # Now accessing .args and .kwargs is safe according to the type hint
        input_spaces_args = [node.merge_space for node in core_recipe_node.args]
        input_spaces_kwargs = {k: node.merge_space for k, node in core_recipe_node.kwargs.items()}

        try:
            output_space = merge_method.get_return_merge_space(input_spaces_args, input_spaces_kwargs)
        except Exception as e:
            logger.error(
                f"Could not determine output merge space for {merge_method.identifier}: {e}. Assuming 'weight'.")
            # Resolve the 'weight' space correctly using sd_mecha's tools
            output_space = sd_mecha.extensions.merge_spaces.resolve("weight")

        # Resolve the 'delta' space correctly
        delta_space = sd_mecha.extensions.merge_spaces.resolve("delta")

        if output_space == delta_space:
            if base_model_node:
                logger.info("Output is a delta, applying to base model.")
                # Ensure we use sd_mecha's add_difference
                return sd_mecha.add_difference(base_model_node, core_recipe_node, alpha=1.0)
            else:
                logger.warning(
                    f"Merge method '{merge_method.identifier}' outputs a delta, but no base model was selected. Returning the delta directly.")
        return core_recipe_node  # Return the original node if output is not delta or no base model

    def _serialize_and_save_recipe(self, final_recipe_node: recipe_nodes.RecipeNode, model_path: Path):
        """Serializes and saves the merged model recipe to a file."""
        try:
            log_dir = Path(HydraConfig.get().runtime.output_dir)
        except ValueError: # Handle case where Hydra is not initialized (e.g., direct script run)
            log_dir = Path(os.getcwd()) / "logs" / "unknown_run"
            logger.warning("Hydra config not found, saving recipe to default log directory.")

        recipes_dir = log_dir / "recipes"
        os.makedirs(recipes_dir, exist_ok=True)

        iteration_file_name = model_path.stem
        recipe_file_path = recipes_dir / f"{iteration_file_name}.mecha"

        try:
            serialized_recipe = sd_mecha.recipe_serializer.serialize(final_recipe_node)
            with open(recipe_file_path, "w", encoding="utf-8") as f:
                f.write(serialized_recipe)
            logger.info(f"Saved recipe to {recipe_file_path}")
        except Exception as e:
            logger.error(f"Failed to serialize or save recipe: {e}")

    def _prepare_model_recipe_args(
        self,
        initial_model_nodes: List[ModelRecipeNode],
        base_model_node: Optional[ModelRecipeNode],
        merge_method: MergeMethod
    ) -> List[RecipeNodeOrValue]:
        """Prepares model nodes for the recipe, handling LoRA conversion and delta subtraction."""
        prepared_nodes = []
        input_types_args = merge_method.get_input_types().args # Get positional arg types
        input_spaces_args = merge_method.get_input_merge_spaces().args # Get positional arg merge spaces

        # Use base_model_node for conversion target if available
        conversion_target_node = base_model_node if base_model_node else initial_model_nodes[0]

        for i, model_node in enumerate(initial_model_nodes):
            current_node = model_node
            is_lora = False

            # --- LoRA Detection & Conversion ---
            # Infer config requires context, do it carefully
            try:
                # Temporarily open dicts to infer config - might be inefficient if done repeatedly
                with sd_mecha.open_input_dicts(current_node, [Path(self.cfg.models_dir)]):
                    inferred_config_id = current_node.model_config.identifier
                    # Simple check based on common identifiers in sd-mecha configs
                    if "lora" in inferred_config_id or "lycoris" in inferred_config_id:
                         is_lora = True
                         logger.info(f"Detected LoRA/LyCORIS: {current_node.path}")
            except Exception as e:
                 logger.warning(f"Could not reliably infer config for {current_node.path} to check for LoRA: {e}. Assuming it's not a LoRA.")

            if is_lora:
                logger.info(f"Converting LoRA node {current_node.path} relative to {conversion_target_node.path}")
                try:
                     # Convert requires the target node for config reference
                    current_node = sd_mecha.convert(current_node, conversion_target_node)
                except Exception as e:
                     logger.error(f"Failed to create conversion recipe for LoRA {current_node.path}: {e}")
                     # Handle error: skip this model, raise, or use original node?
                     raise ValueError(f"LoRA conversion failed for {current_node.path}") from e

            # --- Delta Subtraction ---
            # Check if the corresponding positional parameter expects a delta
            if i < len(input_spaces_args):
                 expected_space = input_spaces_args[i]
                 # Check if expected_space is a set containing delta or the delta space itself
                 is_delta_expected = False
                 if isinstance(expected_space, set):
                     is_delta_expected = sd_mecha.extensions.merge_spaces.resolve("delta") in expected_space
                 elif isinstance(expected_space, sd_mecha.extensions.merge_spaces.MergeSpace):
                     is_delta_expected = expected_space == sd_mecha.extensions.merge_spaces.resolve("delta")

                 if is_delta_expected:
                     if base_model_node:
                         if current_node != base_model_node: # Don't subtract base from itself
                             logger.info(f"Creating delta for model {current_node.path} relative to base.")
                             current_node = sd_mecha.subtract(current_node, base_model_node)
                         else:
                              # This happens if base model itself is passed to a delta slot
                              logger.warning(f"Base model passed to a delta parameter slot for method '{merge_method.identifier}'. Using zero delta.")
                              # Create a zero delta - might need a more robust way
                              current_node = sd_mecha.literal(0.0) # Represent zero delta as literal 0
                     else:
                         raise ValueError(f"Merge method '{merge_method.identifier}' requires a delta for positional argument {i}, but no base model was selected.")

            prepared_nodes.append(current_node)

        return prepared_nodes

    # V1.4 - Uses param_info metadata to expand group/single strategies
    def _prepare_param_recipe_args(
            self,
            params: Dict[str, Any], # Flat params from optimizer: {'OPT_PARAM_NAME': value}
            param_info: BoundsInfo, # Metadata: {'OPT_PARAM_NAME': {'bounds': ..., 'strategy': ..., ...}}
            merge_method: MergeMethod
    ) -> Dict[str, RecipeNodeOrValue]:
        """
        Prepares sd_mecha nodes for parameters, using param_info metadata
        to expand group/single strategies correctly.
        """
        final_param_nodes: Dict[str, RecipeNodeOrValue] = {}
        # Group values by base_param and target_type
        block_based_values_per_param: Dict[str, Dict[str, Any]] = {} # {'alpha': {'BLK1': v, ...}}
        key_based_values_per_param: Dict[str, Dict[str, Any]] = {}   # {'alpha': {'key1': v, ...}}

        logger.debug("Parsing optimizer params using parameter info metadata...")
        # --- Iterate through the metadata generated by bounds.py ---
        for opt_param_name, info in param_info.items():
            if opt_param_name not in params:
                 # This case might happen if resuming optimization with fewer params?
                 logger.warning(f"Optimizer did not provide value for parameter '{opt_param_name}'. Skipping.")
                 continue

            value = params[opt_param_name]
            strategy = info.get('strategy')
            target_type = info.get('target_type')
            base_param = info.get('base_param')
            item_name = info.get('item_name') # Specific block/key for 'all'/'select'
            items_covered = info.get('items_covered', []) # List for 'group'/'single'

            if not base_param:
                 logger.warning(f"Metadata for '{opt_param_name}' missing 'base_param'. Skipping.")
                 continue
            if not target_type:
                 logger.warning(f"Metadata for '{opt_param_name}' missing 'target_type'. Skipping.")
                 continue

            # Initialize dicts for the base_param if needed
            if target_type == 'block': block_based_values_per_param.setdefault(base_param, {})
            elif target_type == 'key': key_based_values_per_param.setdefault(base_param, {})

            # --- Expand value based on strategy ---
            if strategy in ['all', 'select']:
                if not item_name: logger.warning(f"Missing 'item_name' for '{opt_param_name}' ({strategy})."); continue
                if target_type == 'block': block_based_values_per_param[base_param][item_name] = value
                elif target_type == 'key': key_based_values_per_param[base_param][item_name] = value
            elif strategy in ['group', 'single']:
                if not items_covered: logger.warning(f"Missing 'items_covered' for '{opt_param_name}' ({strategy})."); continue
                for item in items_covered:
                    if target_type == 'block': block_based_values_per_param[base_param][item] = value
                    elif target_type == 'key': key_based_values_per_param[base_param][item] = value
            elif strategy == 'custom_fixed':
                 # Handle fixed parameters added directly in get_bounds
                 # Decide if these should be passed as nodes or handled differently
                 # For now, assume they are global literals if they match expected kwargs
                 if base_param in merge_method.get_params().kwargs:
                      logger.debug(f"Found fixed param '{base_param}', adding as literal.")
                      final_param_nodes[base_param] = sd_mecha.literal(value)
                 else:
                      logger.warning(f"Fixed param '{base_param}' from custom_bounds not an expected kwarg for '{merge_method.identifier}'.")
            else:
                 logger.warning(f"Unknown strategy '{strategy}' in metadata for '{opt_param_name}'. Skipping.")

        # --- Create sd-mecha nodes from the populated dictionaries ---
        target_model_node = self._select_base_model() or (self.models[0] if self.models else None)

        # Create nodes for block-targeted parameters (require conversion)
        for base_param, block_dict in block_based_values_per_param.items():
             if base_param in final_param_nodes: logger.warning(f"Parameter '{base_param}' generated by multiple strategies? Overwriting."); continue # Avoid overwriting
             if not block_dict: logger.warning(f"No block values collected for base param '{base_param}'."); continue
             if not self.custom_block_config: logger.error(f"Cannot create block node for '{base_param}': custom_block_config not loaded."); continue
             if not target_model_node: logger.error(f"Cannot create convert node for '{base_param}': target model node not available."); continue
             try:
                 literal_node = sd_mecha.literal(block_dict, config=self.custom_block_config.identifier)
                 converted_node = sd_mecha.convert(literal_node, target_model_node)
                 final_param_nodes[base_param] = converted_node
                 logger.debug(f"Created CONVERT node for '{base_param}' ({len(block_dict)} blocks).")
             except Exception as e: logger.error(f"Failed creating block/convert node for '{base_param}': {e}", exc_info=True)

        # Create nodes for key-targeted parameters (no conversion needed)
        for base_param, key_dict in key_based_values_per_param.items():
             if base_param in final_param_nodes: logger.warning(f"Parameter '{base_param}' generated by multiple strategies? Overwriting."); continue
             if not key_dict: logger.warning(f"No key values collected for base param '{base_param}'."); continue
             if not self.base_model_config: logger.error(f"Cannot create key node for '{base_param}': base_model_config not loaded."); continue
             try:
                 literal_node = sd_mecha.literal(key_dict, config=self.base_model_config.identifier)
                 final_param_nodes[base_param] = literal_node
                 logger.debug(f"Created KEY literal node for '{base_param}' ({len(key_dict)} keys).")
             except Exception as e: logger.error(f"Failed creating key node for '{base_param}': {e}", exc_info=True)

        # --- Handle any leftover global params from optimizer (less common now) ---
        expected_kwarg_names = set(merge_method.get_params().kwargs.keys())
        for opt_param_name, value in params.items():
             # If it's a direct kwarg name AND hasn't been processed via metadata
             if opt_param_name in expected_kwarg_names and opt_param_name not in final_param_nodes and opt_param_name not in param_info:
                  logger.debug(f"Adding globally defined optimizer param '{opt_param_name}' as literal.")
                  final_param_nodes[opt_param_name] = sd_mecha.literal(value)

        logger.info(f"Prepared {len(final_param_nodes)} final parameter nodes for merge method '{merge_method.identifier}'.")
        return final_param_nodes

    # V1.1 - Added handling for fallback_model_index, including -1/None check.
    def _execute_recipe(self, final_recipe_node: recipe_nodes.RecipeNode, model_path: Path):
        """Executes the final recipe using sd_mecha.merge, including fallback."""
        logger.info(f"Executing merge recipe and saving to: {model_path}")

        # --- Determine Fallback Model ---
        fallback_node: Optional[ModelRecipeNode] = None
        # Use .get() with a default of None to handle missing key gracefully
        fallback_index = self.cfg.get("fallback_model_index", None)

        # Check if fallback is explicitly disabled or not provided
        if fallback_index is None or fallback_index == -1:
            logger.info("No fallback model specified (index is None or -1).")
        elif not isinstance(fallback_index, int):
             logger.error(f"Invalid fallback_model_index type: {type(fallback_index)}. Must be an integer or null. No fallback will be used.")
        elif not self.models:
             logger.error(f"fallback_model_index {fallback_index} specified, but no models were loaded (self.models is empty). No fallback will be used.")
        elif not (0 <= fallback_index < len(self.models)):
             logger.error(f"Invalid fallback_model_index: {fallback_index}. Must be between 0 and {len(self.models) - 1}. No fallback will be used.")
        else:
            # Valid index provided
            fallback_node = self.models[fallback_index]
            logger.info(f"Using model at index {fallback_index} ('{fallback_node.path}') as fallback source for missing keys.")

        # --- Execute Merge ---
        try:
            # Make sure self.models_dir is correctly set in __post_init__
            if not self.models_dir or not self.models_dir.is_dir():
                 logger.warning(f"Merger.models_dir ('{self.models_dir}') is not set or invalid. Relative paths in sd_mecha might fail.")
                 effective_model_dirs = [] # Pass empty list if models_dir is bad
            else:
                 effective_model_dirs = [self.models_dir]

            sd_mecha.merge(
                recipe=final_recipe_node,
                output=model_path,
                fallback_model=fallback_node, # Pass the selected node (or None)
                merge_device=self.cfg.get("device", "cpu"), # Default merge device if not set
                merge_dtype=precision_mapping.get(self.cfg.merge_dtype), # Get dtype object
                output_device="cpu", # Keep saving to CPU
                output_dtype=precision_mapping.get(self.cfg.save_dtype), # Get dtype object
                threads=self.cfg.get("threads"),
                model_dirs=effective_model_dirs, # Use the directory containing models
                # Add other relevant sd_mecha.merge options as needed:
                # strict_weight_space=True, check_finite=True, etc.
            )
            logging.info(f"Successfully merged and saved model to {model_path}")
        except Exception as e:
             logger.error(f"sd-mecha merge execution failed: {e}", exc_info=True)
             # Re-raise the exception to signal failure to the optimizer
             raise

    def _save_recipe_etc(self, final_recipe_node: recipe_nodes.RecipeNode, model_path: Path):
         """Handles optional saving of recipe, code, and adding extra keys."""
         try:
             self._serialize_and_save_recipe(final_recipe_node, model_path)

             if self.cfg.get("save_merge_method_code", False):
                 # Assuming MergeMethods is accessible and methods are decorated
                 utils.MergeMethodCodeSaver.save_merge_method_code(self.cfg.merge_method, model_path, MergeMethods)

             # Add extra keys only if the option is enabled
             if self.cfg.get("add_extra_keys", False):
                 utils.add_extra_keys(model_path)
         except Exception as e:
              logger.error(f"Error during post-merge saving operations: {e}")


    # V1.1 - Accepts param_info metadata
    def merge(
            self,
            params: Dict[str, Any],  # Flat params from optimizer
            param_info: BoundsInfo, # <<< ADDED: Full metadata from ParameterHandler
            cache: Optional[Dict],
            # save_best: bool = False # Removed, handled internally by filename logic
    ) -> Path:
        """Builds and executes sd-mecha recipe, using param_info for expansion."""
        cfg = self.cfg
        cache = cache if cache is not None else {}
        logger.info(f"Starting merge process for iteration {self.cfg.get('iteration', '?')}") # Assuming iteration might be available

        # 1. Determine output path (using instance property self.output_file)
        model_path = self.output_file
        if not model_path: # Safety check
             logger.error("Output file path not set in Merger before merge call.")
             # Define a default path or raise error
             model_path = self.models_dir / f"merge_output_default_{cfg.merge_method}.safetensors"
             logger.warning(f"Using default output path: {model_path}")
             self.output_file = model_path # Attempt to set it

        # --- Recipe Building ---
        logger.debug(f"Building merge recipe for method: {cfg.merge_method}")

        # 2. Resolve merge method
        merge_func = utils.resolve_merge_method(cfg.merge_method) # Assumes utils exists

        # 3. Select base model (for delta subtraction, conversion context)
        base_model_node = self._select_base_model()

        # 4. Prepare model input nodes (handles LoRA conversion, delta subtraction)
        prepared_model_nodes = self._prepare_model_recipe_args(
            self.models, base_model_node, merge_func
        )

        # 5. Slice model nodes if merge method has fixed arity
        sliced_model_nodes = self._slice_models(prepared_model_nodes, merge_func)

        # 6. Prepare parameter nodes using param_info for expansion
        param_nodes = self._prepare_param_recipe_args(
             params, param_info, merge_func # Pass metadata here
        )

        # 7. Build the core merge recipe node, applying cache
        logger.info(f"Calling '{merge_func.identifier}' with {len(sliced_model_nodes)} model args, {len(param_nodes)} param nodes.")
        core_recipe_node = merge_func(*sliced_model_nodes, **param_nodes).set_cache(cache)

        # 8. Handle potential delta output (wrap with add_difference)
        final_recipe_node = self._handle_delta_output(
            core_recipe_node, base_model_node, merge_func
        )
        # --- End Recipe Building ---

        # 9. Execute the final recipe (includes fallback logic)
        self._execute_recipe(final_recipe_node, model_path)

        # 10. Optional post-merge steps (save recipe, code, add keys)
        self._save_recipe_etc(final_recipe_node, model_path)

        logger.info(f"Merge process completed. Output: {model_path}")
        return model_path


    def layer_adjust(self, params: Dict, cfg: DictConfig) -> Path:  # Takes params
        """Loads a model, applies layer adjustments, and saves the modified model."""
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
        is_xl_model = any("conditioner.embedders.1" in key for key in state_dict.keys())
        logger.info(f"Determined model type for layer adjustment: {'SDXL' if is_xl_model else 'Non-SDXL'}")

        # Apply adjustments (Assuming utils.modify_state_dict handles the logic)
        logger.info("Applying layer adjustments...")
        try:
            # Pass the raw params dict directly
            modified_state_dict = utils.modify_state_dict(state_dict, params, is_xl_model)
        except Exception as e:
            logger.error(f"Error applying layer adjustments: {e}", exc_info=True)
            raise

        # Save the modified model (using instance property output_file)
        # Ensure output_file is set correctly for the current iteration by the Optimizer
        output_path = self.output_file
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