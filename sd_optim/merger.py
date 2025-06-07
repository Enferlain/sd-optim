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
from typing import Dict, Optional, List, Tuple, Any

from omegaconf import DictConfig, open_dict, ListConfig
from sd_mecha import serialization, extensions, recipe_nodes
from sd_mecha.extensions.merge_methods import MergeMethod, RecipeNodeOrValue
from sd_mecha.recipe_nodes import RecipeNodeOrValue, ModelRecipeNode, RecipeNode
from sd_mecha.extensions import model_configs # Import model_configs

# Assuming utils contains MergeMethodCodeSaver and add_extra_keys
from sd_optim import utils
# Assuming your custom methods are in MergeMethods and decorated correctly
from sd_optim.merge_methods import MergeMethods
from sd_optim.bounds import BoundsInfo, ParameterHandler

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
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        if not cfg.model_paths: raise ValueError("'model_paths' cannot be empty.")

        # --- Base Config Inference (Corrected for infer_model_configs return type) ---
        self.base_model_config: Optional[model_configs.ModelConfig] = None
        representative_model_path_str = cfg.model_paths[0]
        logger.info(f"Merger: Inferring base ModelConfig from: {representative_model_path_str}")
        try:
            self.models_dir = Path(representative_model_path_str).resolve().parent
            if not self.models_dir.is_dir():
                 raise FileNotFoundError(f"Merger: Determined models directory not found: {self.models_dir}")
            logger.info(f"Merger: Determined models directory: {self.models_dir}")

            # Use temporary node for inference check
            rep_model_node = sd_mecha.model(representative_model_path_str)
            with sd_mecha.open_input_dicts(rep_model_node, [self.models_dir]):
                # Check isinstance AND state_dict exists before accessing keys
                if isinstance(rep_model_node, ModelRecipeNode) and rep_model_node.state_dict: # <<< Added isinstance check here too
                    # --- APPLY SAME FIX AS IN ParameterHandler ---
                    inferred_sets = sd_mecha.infer_model_configs(rep_model_node.state_dict.keys())
                    if inferred_sets:
                         best_set = inferred_sets[0]
                         if len(best_set) == 1:
                              self.base_model_config = next(iter(best_set))
                              logger.info(f"Merger: Inferred base ModelConfig: {self.base_model_config.identifier}")
                         else:
                              config_names = {c.identifier for c in best_set}
                              logger.warning(f"Merger: Ambiguous base ModelConfig inferred for {representative_model_path_str}. Possible matches: {config_names}. Picking first one arbitrarily.")
                              self.base_model_config = next(iter(best_set)) # Pick first
                    # --- END FIX ---
                    else:
                        # This path is taken if infer_model_configs returns empty list
                        raise ValueError(f"Merger: Cannot infer ModelConfig for {representative_model_path_str} (no matching configs found).")
                else:
                     # This path is taken if rep_model_node isn't ModelRecipeNode or state_dict is None
                     raise ValueError(f"Merger: Cannot load state dictionary for {representative_model_path_str} to infer config.")

        except FileNotFoundError as fnf_e: # Catch specific error
            logger.error(f"Merger: Model file or directory not found during init: {fnf_e}")
            raise ValueError("Merger could not initialize due to missing file/directory.") from fnf_e
        except ValueError as val_e: # Catch ValueErrors raised above
             logger.error(f"Merger: Error during configuration inference: {val_e}")
             raise # Re-raise ValueErrors as they indicate critical config issues
        except Exception as e: # Catch other unexpected errors
             logger.error(f"Merger: Unexpected error during init: {e}", exc_info=True)
             raise ValueError("Merger failed to initialize.") from e
        # --- End Base Config Inference ---

        # --- Load Custom Config (depends on successful base inference if needed later) ---
        self.custom_block_config_id = self.cfg.optimization_guide.get("custom_block_config_id")
        self.custom_block_config = None
        if self.custom_block_config_id:
            logger.debug(f"Merger: Attempting to load custom block config: '{self.custom_block_config_id}'")
            try:
                # Resolve relies on registration happening in main() BEFORE Merger is initialized
                self.custom_block_config = model_configs.resolve(self.custom_block_config_id)
                logger.info(f"Merger: Successfully loaded custom block config: {self.custom_block_config_id}")
            except ValueError as e:
                # Log warning, but maybe don't raise error here if only used conditionally later?
                # However, if prepare_param_args *requires* it, failure here is problematic.
                logger.warning(f"Merger: Could not resolve custom config '{self.custom_block_config_id}': {e}")
                # If resolution is critical for later steps, uncomment the raise:
                # raise ValueError(f"Invalid custom_block_config_id: {self.custom_block_config_id}") from e

        # --- Initialize other Merger attributes ---
        # Create ModelRecipeNode objects for ALL models AFTER determining models_dir
        self.models: List[ModelRecipeNode] = self._create_model_nodes() # Hint as specific type initially
        self.output_file: Optional[Path] = None
        self.best_output_file: Optional[Path] = None
        self.create_model_out_name() # Set initial output name
        self.create_best_model_out_name() # Set initial best name
        logger.info(f"Merger initialized with {len(self.models)} model nodes.")

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

    # --- Fully Written _create_model_nodes Function ---
    def _create_model_nodes(self) -> List[ModelRecipeNode]:
        """
        Creates sd_mecha ModelRecipeNodes using paths relative to the
        determined models_dir, ensuring only ModelRecipeNodes are returned.
        """
        # Hint the list we are building as the specific type
        model_nodes: List[ModelRecipeNode] = []

        # Check if models_dir was successfully determined in __init__
        if not hasattr(self, 'models_dir') or not self.models_dir or not self.models_dir.is_dir():
             logger.error("Cannot create model nodes: valid models_dir is not set or invalid.")
             return [] # Return empty list if base directory isn't valid

        logger.info(f"Creating model nodes relative to base directory: {self.models_dir}")

        # Safely get model_paths list from config, default to empty list
        model_paths_list = self.cfg.get("model_paths", [])
        # --- MODIFIED CHECK ---
        if not isinstance(model_paths_list, (list, ListConfig)):  # Check for list OR ListConfig
            logger.warning(
                f"'model_paths' in config is not a list or ListConfig ({type(model_paths_list)}). Treating as empty.")
            model_paths_list = []
        # --- END MODIFIED CHECK ---
        else:
            # Optional: Convert ListConfig to plain list for easier processing later if needed
            model_paths_list = list(model_paths_list)

        for model_path_str in model_paths_list:
            if not isinstance(model_path_str, str):
                logger.warning(f"Skipping non-string path in model_paths: {model_path_str}")
                continue

            try:
                resolved_path: Path # Type hint for clarity
                original_path = Path(model_path_str)

                # Attempt to resolve the path
                if original_path.is_absolute():
                    if original_path.exists():
                        resolved_path = original_path.resolve()
                    else:
                        logger.error(f"Absolute model path not found: {original_path}. Skipping node.")
                        continue
                else:
                    # Try relative to models_dir
                    path_relative_to_models_dir = self.models_dir / original_path
                    if path_relative_to_models_dir.exists():
                        resolved_path = path_relative_to_models_dir.resolve()
                    # Maybe try relative to current working directory as a fallback?
                    elif original_path.exists():
                         logger.warning(f"Model path '{original_path}' not found relative to models_dir, but found relative to current working directory. Using absolute path.")
                         resolved_path = original_path.resolve()
                    else:
                        logger.error(f"Relative model path not found: '{original_path}' (checked in '{self.models_dir}' and CWD). Skipping node.")
                        continue

                # Determine path string for sd_mecha.model() (prefer relative)
                try:
                    # Use os.path.relpath for cross-drive compatibility if needed, otherwise Path.relative_to
                    relative_path_str = os.path.relpath(resolved_path, self.models_dir)
                    # Check if relpath actually worked (it might return absolute if on different drives on Windows)
                    if Path(relative_path_str).is_absolute():
                         relative_path_str = str(resolved_path) # Use absolute if relpath failed
                except ValueError:
                    logger.warning(f"Cannot create relative path for {resolved_path} (potentially different drives). Using absolute path.")
                    relative_path_str = str(resolved_path)

                # Create the node using the determined path string
                # sd_mecha.model() is hinted to return RecipeNode
                node: RecipeNode = sd_mecha.model(relative_path_str)

                # --- Explicitly check if it's a ModelRecipeNode ---
                if isinstance(node, ModelRecipeNode):
                    model_nodes.append(node) # Add it to our specific list
                else:
                    # Log an error if sd_mecha.model(path) returns something unexpected
                    logger.error(f"Node created for path '{relative_path_str}' is not a ModelRecipeNode! Type: {type(node)}. Skipping.")
                    # Decide if this should be a fatal error:
                    # raise TypeError(f"Expected ModelRecipeNode for path '{relative_path_str}', got {type(node)}")

            except Exception as e:
                 # Catch potential errors during path resolution or node creation
                 logger.error(f"Error processing path '{model_path_str}': {e}", exc_info=True)
                 # Continue to the next path

        logger.info(f"Merger: Created {len(model_nodes)} ModelRecipeNodes using relative paths.")
        return model_nodes # Return the list guaranteed to contain ModelRecipeNodes

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
            serialized_recipe = sd_mecha.serialization.serialize(final_recipe_node)
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
    ) -> List[RecipeNodeOrValue]:  # Return type might not always be just RecipeNode now
        prepared_nodes = []
        input_types = merge_method.get_input_types()
        input_spaces = merge_method.get_input_merge_spaces()
        param_names = merge_method.get_param_names()
        delta_space_obj = sd_mecha.extensions.merge_spaces.resolve("delta")  # Resolve once

        conversion_target_node = base_model_node if base_model_node else (
            initial_model_nodes[0] if initial_model_nodes else None)
        if not conversion_target_node:
            logger.error("Cannot prepare model args: No target node for potential conversions.")
            return []

        for i, model_node in enumerate(initial_model_nodes):
            current_node: RecipeNode = model_node  # Start with original node
            is_lora = False
            should_add_node = True  # Flag to control adding to prepared_nodes

            # --- LoRA Detection & Conversion (remains the same) ---
            try:
                if not self.models_dir: raise FileNotFoundError("models_dir not set")
                with sd_mecha.open_input_dicts(current_node, [self.models_dir]):
                    if current_node.model_config:
                        inferred_config_id = current_node.model_config.identifier
                        if "lora" in inferred_config_id or "lycoris" in inferred_config_id: is_lora = True; logger.info(
                            f"Detected LoRA/LyCORIS: {current_node.path}")
            except Exception as e_inf:
                logger.warning(f"Error during LoRA check for {current_node.path}: {e_inf}.")

            if is_lora:
                logger.info(f"Converting LoRA node {current_node.path} relative to {conversion_target_node.path}")
                try:
                    current_node = sd_mecha.convert(current_node, conversion_target_node, model_dirs=[self.models_dir])
                except Exception as e:
                    logger.error(f"LoRA conversion failed for {current_node.path}: {e}"); raise ValueError from e

            # --- Delta Subtraction / Exclusion ---
            is_delta_expected = False
            expected_space_for_arg = None
            # Determine expected space (same logic as before)
            if param_names.has_varargs() and i >= len(param_names.args):
                expected_space_for_arg = input_spaces.vararg
            elif i < len(param_names.args):
                expected_space_for_arg = input_spaces.args[i]

            if expected_space_for_arg is not None:
                if isinstance(expected_space_for_arg, set):
                    is_delta_expected = delta_space_obj in expected_space_for_arg
                elif isinstance(expected_space_for_arg, sd_mecha.extensions.merge_spaces.MergeSpace):
                    is_delta_expected = expected_space_for_arg == delta_space_obj

            logger.debug(
                f"Arg {i}: Path={getattr(model_node, 'path', 'N/A')}, Expected space={expected_space_for_arg}, Needs Delta={is_delta_expected}")

            if is_delta_expected:
                logger.debug(f"  Parameter {i} expects delta. Base model selected: {base_model_node is not None}")
                if base_model_node:
                    is_base = current_node == base_model_node
                    logger.debug(f"  Node is base model: {is_base}")
                    if not is_base:
                        logger.info(f"  Creating delta for node {getattr(current_node, 'path', 'N/A')}...")
                        current_node = sd_mecha.subtract(current_node, base_model_node)  # Perform subtraction
                        logger.info(
                            f"  Delta node created. New merge space: {current_node.merge_space.identifier if current_node.merge_space else 'N/A'}")
                    else:
                        # --- CHANGE: Don't add the base model itself to the list ---
                        logger.warning(
                            f"  Base model '{getattr(current_node, 'path', 'N/A')}' matches arg {i} which expects delta. EXCLUDING this node from arguments passed to '{merge_method.identifier}'.")
                        should_add_node = False  # Set flag to skip adding
                        # --- END CHANGE ---
                else:
                    # No base model selected, cannot create delta - raise error
                    raise ValueError(
                        f"Merge method '{merge_method.identifier}' requires a delta for positional argument {i}, but no base model was selected.")

            # --- Add the node only if the flag allows ---
            if should_add_node:
                prepared_nodes.append(current_node)
            # --- End Add Node ---

        # Log the final list of nodes being passed
        logger.debug(
            f"Prepared {len(prepared_nodes)} nodes for method '{merge_method.identifier}': {[getattr(n, 'path', type(n).__name__) for n in prepared_nodes]}")
        return prepared_nodes

    # V1.5 - Block AND key, uses fallback to merge (and overwrite from keys to block)
    def _prepare_param_recipe_args(
            self,
            params: Dict[str, Any],  # Flat params from optimizer: {'OPT_PARAM_NAME': value}
            param_info: BoundsInfo,  # Metadata: {'OPT_PARAM_NAME': {'bounds': ..., 'strategy': ..., ...}}
            merge_method: MergeMethod
    ) -> Dict[str, RecipeNodeOrValue]:
        """
        Prepares sd_mecha nodes for parameters based on strategies and handles fixed kwargs.
        Now supports combining block and key configs using fallback merge.
        """
        final_param_nodes: Dict[str, RecipeNodeOrValue] = {}
        # Group values by base_param and target_type based on optimizer output and param_info
        block_based_values_per_param: Dict[str, Dict[str, Any]] = {}
        key_based_values_per_param: Dict[str, Dict[str, Any]] = {}
        handled_base_params = set()  # Keep track of base params covered by strategies

        logger.debug("Parsing optimizer params using parameter info metadata...")

        # --- Step 1: Populate values for Strategy-Based Parameters ---
        for opt_param_name, info in param_info.items():
            if opt_param_name not in params:
                logger.warning(f"Optimizer did not provide value for parameter '{opt_param_name}'. Skipping.")
                continue

            value = params[opt_param_name]
            strategy = info.get('strategy')
            target_type = info.get('target_type')
            base_param = info.get('base_param')
            item_name = info.get('item_name')
            items_covered = info.get('items_covered', [])

            if not base_param: continue
            handled_base_params.add(base_param)  # Mark this base_param as handled

            if target_type == 'block':
                block_based_values_per_param.setdefault(base_param, {})
            elif target_type == 'key':
                key_based_values_per_param.setdefault(base_param, {})
            else:
                logger.warning(f"Unknown target_type '{target_type}' for '{opt_param_name}'.")
                continue

            if strategy in ['all', 'select']:
                if not item_name:
                    logger.warning(f"Missing 'item_name' for '{opt_param_name}' ({strategy}).")
                    continue
                if target_type == 'block':
                    block_based_values_per_param[base_param][item_name] = value
                elif target_type == 'key':
                    key_based_values_per_param[base_param][item_name] = value

            elif strategy in ['group', 'single']:
                if not items_covered:
                    logger.warning(f"Missing 'items_covered' for '{opt_param_name}' ({strategy}).")
                    continue
                for item in items_covered:
                    if target_type == 'block':
                        block_based_values_per_param[base_param][item] = value
                    elif target_type == 'key':
                        key_based_values_per_param[base_param][item] = value

        # --- Step 1b: Create sd-mecha nodes for Strategy-Based Parameters ---
        target_model_node = self._select_base_model() or (self.models[0] if self.models else None)
        if not target_model_node:
            logger.error("Cannot prepare parameter nodes: Target/Base model node is missing.")
            return {}

        # Process each base parameter that has values from either blocks or keys
        all_base_params = set(block_based_values_per_param.keys()) | set(key_based_values_per_param.keys())

        for base_param in all_base_params:
            block_dict = block_based_values_per_param.get(base_param, {})
            key_dict = key_based_values_per_param.get(base_param, {})

            # Case 1: Only block values
            if block_dict and not key_dict:
                if not self.custom_block_config:
                    logger.error(f"Cannot make block node for '{base_param}': custom block config missing.")
                    continue
                try:
                    literal_node = sd_mecha.literal(block_dict, config=self.custom_block_config.identifier)
                    converted_node = sd_mecha.convert(
                        literal_node,
                        target_model_node,
                        model_dirs=[self.models_dir]
                    )
                    final_param_nodes[base_param] = converted_node
                    logger.debug(f"Created BLOCK-ONLY node for '{base_param}' ({len(block_dict)} blocks).")
                except Exception as e:
                    logger.error(f"Failed creating block node for '{base_param}': {e}", exc_info=True)

            # Case 2: Only key values
            elif key_dict and not block_dict:
                if not self.base_model_config:
                    logger.error(f"Cannot create key node for '{base_param}': base_model_config not loaded.")
                    continue
                try:
                    literal_node = sd_mecha.literal(key_dict, config=self.base_model_config.identifier)
                    final_param_nodes[base_param] = literal_node
                    logger.debug(f"Created KEY-ONLY node for '{base_param}' ({len(key_dict)} keys).")
                except Exception as e:
                    logger.error(f"Failed creating key node for '{base_param}': {e}", exc_info=True)

            # Case 3: Both block and key values - MERGE THEM with fallback
            elif block_dict and key_dict:
                if not self.custom_block_config:
                    logger.error(f"Cannot make block node for '{base_param}': custom block config missing.")
                    continue
                if not self.base_model_config:
                    logger.error(f"Cannot create key node for '{base_param}': base_model_config not loaded.")
                    continue

                logger.debug(
                    f"Merging block and key values for '{base_param}' ({len(block_dict)} blocks + {len(key_dict)} keys). Keys will override blocks.")
                try:
                    # Step 1: Create block literal with custom config
                    block_literal = sd_mecha.literal(block_dict, config=self.custom_block_config.identifier)

                    # Step 2: Convert blocks to base config (same as models)
                    block_converted = sd_mecha.convert(
                        block_literal,
                        target_model_node,
                        model_dirs=[self.models_dir]
                    )

                    # Step 3: Create key literal with base config (already compatible)
                    key_literal = sd_mecha.literal(key_dict, config=self.base_model_config.identifier)

                    # Step 4: Use fallback to combine - key values override block values
                    # Order matters: later arguments override earlier ones
                    merged_node = key_literal | block_converted

                    final_param_nodes[base_param] = merged_node
                    logger.debug(f"Created FALLBACK-MERGED node for '{base_param}' (blocks + keys, keys override).")
                except Exception as e:
                    logger.error(f"Failed creating merged node for '{base_param}': {e}", exc_info=True)

            # Case 4: Neither (shouldn't happen, but just in case)
            else:
                logger.warning(f"No values found for base parameter '{base_param}' - this shouldn't happen.")

        # --- Step 2: Handle Fixed Keyword Arguments (Not Optimized via Strategies) ---
        logger.debug(f"Checking for fixed keyword arguments using custom_bounds...")
        expected_kwargs = set(merge_method.get_params().kwargs.keys())

        unhandled_kwargs = expected_kwargs - handled_base_params
        logger.debug(f"Expected Kwargs: {expected_kwargs}")
        logger.debug(f"Handled Base Params (Strategies): {handled_base_params}")
        logger.debug(f"Unhandled Expected Kwargs: {unhandled_kwargs}")

        # Get custom bounds config safely and validate it
        custom_bounds_config = self.cfg.optimization_guide.get("custom_bounds", {})
        validated_custom_bounds = ParameterHandler.validate_custom_bounds(custom_bounds_config)

        for kwarg_name in unhandled_kwargs:
            if kwarg_name in validated_custom_bounds:
                # Fixed value provided directly in custom_bounds
                fixed_value = validated_custom_bounds[kwarg_name]
                final_param_nodes[kwarg_name] = sd_mecha.literal(fixed_value)
                logger.info(f"Using fixed value for '{kwarg_name}' from custom_bounds: {fixed_value}")
            else:
                # Not in custom_bounds, rely on merge method's own default value
                logger.info(
                    f"Using default value for parameter '{kwarg_name}' from merge method '{merge_method.identifier}'.")
                # Do NOT add to final_param_nodes, sd-mecha will use the method's default

        logger.info(
            f"Prepared {len(final_param_nodes)} final parameter nodes for merge method '{merge_method.identifier}'.")
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

            logger.info(f"Calling sd_mecha.merge with fallback_model: {fallback_node}")
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
                check_mandatory_keys=False,
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
            iteration: int = 0 # <<< ADD iteration parameter
    ) -> Path:
        """Builds and executes sd-mecha recipe, using param_info for expansion."""
        cfg = self.cfg
        cache = cache if cache is not None else {}
        logger.info(f"Starting merge process for iteration {iteration}") # <<< USE iteration

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

        logger.info(f"Selected base model node: {base_model_node.path if base_model_node else 'None'}")

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