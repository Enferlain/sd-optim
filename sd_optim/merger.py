# merger.py - Version 1.1 - initial changes
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
from sd_mecha.recipe_nodes import RecipeNodeOrValue, ModelRecipeNode, RecipeNode, MergeRecipeNode
from sd_mecha.extensions import model_configs, merge_methods  # Import model_configs

from sd_optim import utils
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
    def __init__(self, cfg: DictConfig, base_model_config: sd_mecha.extensions.model_configs.ModelConfig,
                 custom_block_config: sd_mecha.extensions.model_configs.ModelConfig | None, models_dir: Path):
        self.cfg = cfg
        self.base_model_config = base_model_config
        self.custom_block_config = custom_block_config
        self.models_dir = models_dir

        self.output_file: Optional[Path] = None
        self.best_output_file: Optional[Path] = None

        self.models: List[ModelRecipeNode] = []
        if self.cfg.optimization_mode == "merge":
            logger.info("Creating model nodes for 'merge' mode.")
            self.models = self._create_model_nodes()

        logger.info("Merger initialized successfully with pre-loaded configs.")

    def _create_model_nodes(self) -> List[ModelRecipeNode]:
        """
        Creates sd_mecha ModelRecipeNodes from the config. This version correctly
        calculates relative paths for the recipe, ensuring portability, while handling
        potential cross-drive issues on Windows.
        """
        model_nodes: List[ModelRecipeNode] = []
        model_paths_list = self.cfg.get("model_paths", [])

        if not isinstance(model_paths_list, (list, ListConfig)):
            logger.warning("'model_paths' in config is not a list. No model nodes will be created.")
            model_paths_list = []

        if not model_paths_list:
            logger.info("No model paths provided, returning an empty list of model nodes.")
            return model_nodes

        if not hasattr(self, 'models_dir') or not self.models_dir or not self.models_dir.is_dir():
            logger.error(
                f"Cannot create model nodes: models_dir '{getattr(self, 'models_dir', 'Not Set')}' is invalid.")
            return model_nodes

        logger.info(f"Creating model nodes relative to base directory: {self.models_dir}")
        for model_path_str in model_paths_list:
            try:
                if not isinstance(model_path_str, str):
                    logger.warning(f"Skipping non-string path in model_paths: {model_path_str}")
                    continue

                original_path = Path(model_path_str)
                resolved_path: Optional[Path] = None

                if original_path.is_absolute():
                    if original_path.exists():
                        resolved_path = original_path
                    else:
                        logger.error(f"Absolute model path not found: {original_path}. Skipping node.")
                        continue
                else:
                    path_in_models_dir = self.models_dir / original_path
                    if path_in_models_dir.exists():
                        resolved_path = path_in_models_dir
                    elif original_path.exists():
                        resolved_path = original_path.resolve()
                    else:
                        logger.error(f"Relative model path not found: '{original_path}'. Skipping node.")
                        continue

                # We now calculate the path to be used inside the recipe.
                path_for_recipe: str
                try:
                    # Attempt to create a path relative to our main models directory.
                    path_for_recipe = os.path.relpath(resolved_path, self.models_dir)
                except ValueError:
                    # This happens on Windows if paths are on different drives (e.g., C: vs D:).
                    # In that case, we have no choice but to fall back to the absolute path.
                    logger.warning(
                        f"Could not create relative path for {resolved_path} (likely on a different drive). Using absolute path in recipe.")
                    path_for_recipe = str(resolved_path)

                logger.debug(f"Resolved path '{resolved_path}' to recipe path '{path_for_recipe}'")

                # Create the node using our calculated (preferably relative) path string.
                node = sd_mecha.model(path_for_recipe)

                if isinstance(node, ModelRecipeNode):
                    model_nodes.append(node)
                else:
                    logger.warning(f"Node created for path '{path_for_recipe}' was not a ModelRecipeNode. Skipping.")

            except Exception as e_node:
                logger.error(f"Failed to create node for path '{model_path_str}': {e_node}", exc_info=True)
                continue

        logger.info(f"Finished creating nodes. Total successful: {len(model_nodes)}.")
        return model_nodes

    def create_model_output_name(
            self,
            iteration: int,
            best: bool = False,
            recipe_node: Optional[RecipeNode] = None
    ) -> Path:
        """
        Generates the output file name for the merged model based on the optimization mode.
        Uses full names and applies a hard cutoff only if the final name is excessively long.
        """
        combined_name = "fallback_merge_name"  # Default fallback name
        max_filename_len = 120  # A safe length for most filesystems

        # --- Mode 1: MERGE (Uses full names) ---
        if self.cfg.optimization_mode == "merge":
            # Get full model names without the extension
            model_names = [Path(path).stem for path in self.cfg.model_paths]

            if len(model_names) >= 2:
                name_part = f"{model_names[0]}-{model_names[1]}"
            elif len(model_names) == 1:
                name_part = model_names[0]
            else:
                name_part = "merged_model"

            merge_method_name = self.cfg.merge_method
            combined_name = f"{name_part}-{merge_method_name}-it_{iteration}"

        # --- Mode 2: LAYER_ADJUST (Uses full name) ---
        elif self.cfg.optimization_mode == "layer_adjust":
            model_name = Path(self.cfg.model_paths[0]).stem
            combined_name = f"layer_adjusted-{model_name}-it_{iteration}"

        # --- Mode 3: RECIPE (This is the fixed logic) ---
        elif self.cfg.optimization_mode == "recipe":
            # Pass the ROOT of the recipe graph to the utility
            if recipe_node is not None:
                recipe_cfg = self.cfg.get('recipe_optimization', {})
                target_node_ref = recipe_cfg.get("target_nodes")

                # Call our NEW utility function
                node_info = utils.get_info_from_target_node(recipe_node, target_node_ref)

                if node_info:
                    method_name = node_info["method_name"]
                    # Get the stems of the file paths
                    model_names = [Path(name).stem for name in node_info["model_names"]]

                    if len(model_names) >= 2:
                        name_part = f"{model_names[0]}-{model_names[1]}"
                    elif len(model_names) == 1:
                        name_part = model_names[0]
                    else:
                        name_part = "optimized_merge"  # Should not happen if models exist

                    combined_name = f"{name_part}-{method_name}-it_{iteration}"
                else:
                    # Fallback if info can't be retrieved from the node
                    recipe_name = Path(recipe_cfg.get("recipe_path", "unknown")).stem
                    combined_name = f"recipe_{recipe_name}-it_{iteration}"
            else:
                # Fallback if no recipe_node was passed at all
                combined_name = f"recipe_fallback-it_{iteration}"

        # --- Final Assembly and Truncation (unchanged) ---
        if best:
            combined_name += "_best"

        if len(combined_name) > max_filename_len:
            logger.warning(f"Generated filename is too long. Truncating to {max_filename_len} characters.")
            combined_name = combined_name[:max_filename_len]

        output_dir = self.models_dir
        # This will fail if models_dir isn't set, which is a critical error anyway
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f"{combined_name}.safetensors"

    def _select_base_model(self) -> Optional[recipe_nodes.ModelRecipeNode]:
        """Selects the base model node based on configuration index."""
        base_model_index = self.cfg.get("base_model_index", None)
        if base_model_index is None:
            return None

        if not isinstance(base_model_index, int) or not (0 <= base_model_index < len(self.models)):
            raise ValueError(
                f"Invalid base_model_index: {base_model_index}. Must be an integer within range [0, {len(self.models) - 1}).")

        base_model_node = self.models[base_model_index]

        # Infer config to check if it's a LoRA - Requires sd_mecha context
        try:
            # Use the self.models_dir attribute that was safely determined during __init__.
            # Do not access self.cfg directly for this path.
            if not hasattr(self, 'models_dir') or not self.models_dir:
                raise FileNotFoundError("Merger's models_dir attribute is not set.")

            with sd_mecha.open_input_dicts(base_model_node, [self.models_dir]):
                # The LoRA check logic itself is fine.
                if base_model_node.model_config and ("lora" in base_model_node.model_config.identifier or
                                                     "lycoris" in base_model_node.model_config.identifier):
                    raise ValueError(
                        f"The selected base model ('{base_model_node.path}') appears to be a LoRA/LyCORIS. These cannot be used as base models."
                    )
        except (ValueError, FileNotFoundError) as e:
            # Re-raise configuration and setup errors as they are critical.
            logger.error(f"Error during base model validation: {e}")
            raise
        except Exception as e:
            # Log other unexpected errors during the check
            logger.error(f"Unexpected error during base model config inference for LoRA check: {e}", exc_info=True)
            # We can decide to proceed cautiously or halt, halting is safer.
            raise ValueError(f"Could not verify base model '{base_model_node.path}'. Halting.") from e

        return base_model_node

    def _get_conversion_context_node(self) -> recipe_nodes.ModelRecipeNode:
        """
        Finds a suitable, reliable ModelRecipeNode to serve as a conversion target.
        This is critical for converting custom blocks to the base model's key space.
        It uses a clear priority system to find the best possible candidate.
        """
        logger.debug("Attempting to find a suitable model for conversion context...")

        # Priority 1: Use the explicitly defined base_model, if in a mode that uses it ('merge').
        if self.cfg.optimization_mode == "merge":
            base_model = self._select_base_model()  # Our already-fixed _select_base_model
            if base_model:
                logger.info(f"Using explicitly defined base_model '{base_model.path}' as conversion context.")
                return base_model

        # Priority 2: Use the explicitly defined fallback_model. This is a great, stable choice.
        fallback_index = self.cfg.get("fallback_model_index", -1)
        if fallback_index != -1 and 0 <= fallback_index < len(self.models):
            fallback_model = self.models[fallback_index]
            # We MUST validate that the fallback model isn't a LoRA!
            self._validate_node_is_not_lora(fallback_model)
            logger.info(
                f"Using fallback_model at index {fallback_index} ('{fallback_model.path}') as conversion context.")
            return fallback_model

        # Priority 3 (Recipe Mode): Intelligently scan the recipe for the first valid base model.
        if self.cfg.optimization_mode == "recipe":
            logger.debug("Scanning recipe for the first valid base model to use as context...")
            recipe_path = Path(self.cfg.recipe_optimization.recipe_path)
            original_recipe_text = recipe_path.read_text(encoding="utf-8")

            # We deserialize the whole recipe to inspect its nodes
            recipe_graph = sd_mecha.deserialize(original_recipe_text)
            visitor = utils.ModelVisitor()
            recipe_graph.accept(visitor)

            for model_node in visitor.models:
                try:
                    # Check each model in the recipe until we find one that isn't a LoRA.
                    self._validate_node_is_not_lora(model_node, raise_error=False)
                    logger.info(f"Found suitable model '{model_node.path}' inside recipe to use as conversion context.")
                    return model_node
                except ValueError:
                    logger.debug(f"Skipping '{model_node.path}' as conversion context because it appears to be a LoRA.")
                    continue

        # Priority 4: If all else fails, use the first model from the model_paths list as a last resort.
        if self.models:
            first_model = self.models[0]
            self._validate_node_is_not_lora(first_model)
            logger.warning(
                f"Could not find an explicit base/fallback model. Using the first model in model_paths ('{first_model.path}') as a last-resort conversion context.")
            return first_model

        # If we reach here, something is catastrophically wrong.
        raise RuntimeError("Could not determine any suitable model to use for conversion context.")

    # We also need a small, reusable validator for the LoRA check.
    def _validate_node_is_not_lora(self, node: recipe_nodes.ModelRecipeNode, raise_error: bool = True):
        """Checks if a given ModelRecipeNode is a LoRA/LyCORIS. Raises ValueError if it is."""
        try:
            with sd_mecha.open_input_dicts(node, [self.models_dir]):
                config_id = node.model_config.identifier
                if "lora" in config_id or "lycoris" in config_id:
                    raise ValueError(
                        f"Model '{node.path}' appears to be a LoRA/LyCORIS and cannot be used as a base/context model.")
        except ValueError as e:
            if raise_error:
                raise e
            else:
                # Silently re-raise to be caught by the calling function
                raise
        except Exception as e:
            logger.error(f"Could not verify model '{node.path}' for LoRA check: {e}")
            if raise_error:
                raise RuntimeError(f"Could not verify model '{node.path}'.") from e
            else:
                raise

    def _slice_models(self, prepared_model_nodes: List[RecipeNodeOrValue], merge_method: MergeMethod) -> List[
        RecipeNodeOrValue]:
        """Slices the model list to match the expected number of model-type arguments."""
        param_info = merge_method.get_param_names()
        if param_info.has_varargs():
            return prepared_model_nodes

        # --- THIS IS THE FIX ---
        # Instead of just counting all arguments, we check their types.
        input_types = merge_method.get_input_types().args

        # Count how many arguments expect a StateDict (i.e., a model) vs a plain Tensor/value
        expected_num_models = 0
        for arg_type in input_types:
            # get_origin helps handle types like StateDict[Tensor]
            origin_type = getattr(arg_type, '__origin__', arg_type)
            if origin_type and (
                issubclass(origin_type, sd_mecha.extensions.merge_methods.StateDict) or
                issubclass(origin_type, torch.Tensor)
            ):
                expected_num_models += 1
        # For weighted_sum, this will correctly count 2 (for 'a' and 'b').

        num_provided = len(prepared_model_nodes)

        if num_provided > expected_num_models:
            logger.warning(
                f"Merge method '{merge_method.identifier}' expects {expected_num_models} model arguments, "
                f"but {num_provided} were prepared. Using the first {expected_num_models}."
            )
            return prepared_model_nodes[:expected_num_models]
        elif num_provided < expected_num_models:
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
        except ValueError:  # Handle case where Hydra is not initialized (e.g., direct script run)
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
            initial_model_nodes: List[recipe_nodes.ModelRecipeNode],
            base_model_node: Optional[recipe_nodes.ModelRecipeNode],
            # FIX 1: Use the correct reference
            merge_method: merge_methods.MergeMethod
    ) -> List[RecipeNodeOrValue]:
        """
        Prepares the list of model nodes to be passed as positional arguments to a merge method.
        This function handles:
        1. Automatic detection and conversion of LoRA models to delta nodes.
        2. Creation of delta nodes for standard models if the merge method expects them.
        3. Exclusion of the base model itself when creating deltas.
        """
        prepared_nodes: List[RecipeNodeOrValue] = []
        param_info = merge_method.get_param_names()
        input_spaces = merge_method.get_input_merge_spaces()
        delta_space = extensions.merge_spaces.resolve("delta")

        conversion_target_node = base_model_node if base_model_node else (
            initial_model_nodes[0] if initial_model_nodes else None)

        for i, model_node in enumerate(initial_model_nodes):
            current_node: recipe_nodes.RecipeNode = model_node
            should_add_node = True
            is_lora = False

            # Keep a reference to the original path for logging, before it gets converted
            original_path_for_logging = model_node.path

            # --- Step 1: LoRA Detection ---
            try:
                with sd_mecha.open_input_dicts(current_node, [self.models_dir]):
                    inferred_config_id = current_node.model_config.identifier
                    if "lora" in inferred_config_id or "lycoris" in inferred_config_id:
                        is_lora = True
                        # FIX 2: We use the guaranteed original path for this log message
                        logger.info(
                            f"Identified LoRA/LyCORIS: '{original_path_for_logging}' with config '{inferred_config_id}'")
            except Exception as e:
                logger.error(f"Could not infer config for model {original_path_for_logging}: {e}")
                is_lora = False

            # --- Step 2: LoRA Conversion ---
            if is_lora:
                if not conversion_target_node:
                    raise ValueError(
                        f"LoRA '{original_path_for_logging}' detected, but no base model was provided to convert it against.")

                logger.info(f"Converting LoRA '{original_path_for_logging}' to a delta...")
                try:
                    current_node = sd_mecha.convert(current_node, conversion_target_node, model_dirs=[self.models_dir])
                    logger.info(
                        f"LoRA converted successfully. New node is in '{current_node.merge_space.identifier}' space.")
                except Exception as e:
                    logger.error(f"CRITICAL: LoRA conversion failed for {original_path_for_logging}: {e}",
                                 exc_info=True)
                    raise

            # --- Step 3: Delta Subtraction Logic ---
            is_delta_expected = False
            expected_space_for_arg = None
            if param_info.has_varargs() and i >= len(param_info.args):
                expected_space_for_arg = input_spaces.vararg
            elif i < len(param_info.args):
                expected_space_for_arg = input_spaces.args[i]

            if isinstance(expected_space_for_arg, set):
                is_delta_expected = delta_space in expected_space_for_arg
            elif isinstance(expected_space_for_arg, extensions.merge_spaces.MergeSpace):
                is_delta_expected = expected_space_for_arg == delta_space

            if is_delta_expected and current_node.merge_space != delta_space:
                if not base_model_node:
                    raise ValueError(
                        f"Merge method '{merge_method.identifier}' requires a delta for argument {i}, but no base_model was selected.")

                if current_node is base_model_node:
                    # FIX 2: Check the type before accessing .path
                    log_path = base_model_node.path if isinstance(base_model_node,
                                                                  recipe_nodes.ModelRecipeNode) else "the base model"
                    logger.warning(
                        f"Base model '{log_path}' matches arg {i} which expects a delta. Excluding it from arguments.")
                    should_add_node = False
                else:
                    # FIX 2: Check the type before accessing .path
                    log_path = current_node.path if isinstance(current_node,
                                                               recipe_nodes.ModelRecipeNode) else "a model node"
                    logger.info(f"Creating delta for '{log_path}'...")
                    current_node = sd_mecha.subtract(current_node, base_model_node)

            # --- Step 4: Final Append ---
            if should_add_node:
                prepared_nodes.append(current_node)

        logger.info(
            f"Finished preparing {len(prepared_nodes)} model arguments for merge method '{merge_method.identifier}'.")
        return prepared_nodes

    # V1.6 - Proper context for conversions
    def _prepare_param_recipe_args(
            self,
            params: Dict[str, Any],  # Flat params from optimizer: {'OPT_PARAM_NAME': value}
            param_info: BoundsInfo,  # Metadata: {'OPT_PARAM_NAME': {'bounds': ..., 'strategy': ..., ...}}
            merge_method: MergeMethod
    ) -> Dict[str, RecipeNode]:
        """
        Prepares sd_mecha nodes for parameters based on strategies and handles fixed kwargs.
        Now supports combining block and key configs using fallback merge.
        """
        final_param_nodes: Dict[str, RecipeNode] = {}
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

            if not base_param:
                continue
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
        target_model_node = self._get_conversion_context_node()

        if not target_model_node:  # This check is now for a truly fatal error
            logger.error("Cannot prepare parameter nodes: A conversion context model is missing.")
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
                # Not in custom_bounds, so its value depends on the mode.
                if self.cfg.optimization_mode == 'recipe':
                    # In recipe mode, we are NOT providing this kwarg, so the original value in the text file remains.
                    logger.info(
                        f"Using value for parameter '{kwarg_name}' from the source .mecha file."
                    )
                else:  # This applies to 'merge' mode and any other future modes
                    # In merge mode, we don't provide it, so sd-mecha uses the function's default.
                    logger.info(
                        f"Using default value for parameter '{kwarg_name}' from merge method '{merge_method.identifier}'."
                    )

        logger.info(
            f"Prepared {len(final_param_nodes)} final parameter nodes for merge method '{merge_method.identifier}'.")
        return final_param_nodes

    # V1.2 - fallback_model_index for merge and recipe properly
    def _execute_recipe(self, final_recipe_node: recipe_nodes.RecipeNode, model_path: Path):
        """Executes the final recipe, correctly handling the fallback model for ALL modes."""
        logger.info(f"Executing merge recipe and saving to: {model_path}")

        # --- Step 1: Determine the list of models available for fallback ---
        models_for_lookup: List[ModelRecipeNode] = []
        if self.cfg.optimization_mode == "merge":
            models_for_lookup = self.models
        elif self.cfg.optimization_mode == "recipe":
            # In recipe mode, we parse the FINAL recipe to get an accurate list.
            visitor = utils.ModelVisitor()
            final_recipe_node.accept(visitor)
            models_for_lookup = visitor.models

        # --- Step 2: Determine the Fallback Node using EXPLICIT checks ---
        fallback_node: Optional[ModelRecipeNode] = None
        fallback_index = self.cfg.get("fallback_model_index", -1)

        # This is the original, explicit checking structure you liked, now made mode-aware.
        if fallback_index is None or fallback_index == -1:
            logger.info("No fallback model specified.")
        elif not isinstance(fallback_index, int):
            logger.error(
                f"Invalid fallback_model_index type: {type(fallback_index)}. Must be an integer or null. No fallback will be used."
            )
        elif not models_for_lookup:  # This check now works for both modes.
            logger.error(
                f"fallback_model_index {fallback_index} specified, but no models were found in the current context. No fallback will be used."
            )
        elif not (0 <= fallback_index < len(models_for_lookup)):  # This check also now works for both modes.
            logger.error(
                f"Invalid fallback_model_index: {fallback_index}. Must be between 0 and {len(models_for_lookup) - 1}. No fallback will be used."
            )
        else:
            # If all checks pass, we have a valid index for our lookup list.
            fallback_node = models_for_lookup[fallback_index]
            logger.info(
                f"Using model at index {fallback_index} ('{fallback_node.path}') as fallback source for missing keys."
            )

        # --- Step 3: Execute the Merge (this part was already correct) ---
        try:
            if not self.models_dir or not self.models_dir.is_dir():
                raise FileNotFoundError("Merger.models_dir is not set or is not a valid directory.")

            logger.info(f"Calling sd_mecha.merge with fallback_model: {fallback_node}")
            sd_mecha.merge(
                recipe=final_recipe_node,
                output=model_path,
                fallback_model=fallback_node,  # Pass the selected node (or None)
                merge_device=self.cfg.get("device", "cpu"),  # Default merge device if not set
                merge_dtype=precision_mapping.get(self.cfg.merge_dtype),  # Get dtype object
                output_device="cpu",  # Keep saving to CPU
                output_dtype=precision_mapping.get(self.cfg.save_dtype),  # Get dtype object
                threads=self.cfg.get("threads"),
                model_dirs=[self.models_dir],  # Use the directory containing models
                check_mandatory_keys=False,
                # Add other relevant sd_mecha.merge options as needed:
                # strict_weight_space=True, check_finite=True, etc.
            )
            logging.info(f"Successfully merged and saved model to {model_path}")
        except Exception as e:
            logger.error(f"sd-mecha merge execution failed: {e}", exc_info=True)
            # Re-raise the exception to signal failure to the optimizer
            raise

    # This is our conductor function, now correctly implemented.
    def _save_recipe_etc(self, final_recipe_node: recipe_nodes.RecipeNode, model_path: Path, iteration: int):
        """
        Handles optional saving of artifacts by calling specialized functions.
        """
        try:
            # --- Step 1: ALWAYS save the mandatory .mecha recipe ---
            self._serialize_and_save_recipe(final_recipe_node, model_path)

            # --- Step 2: OPTIONALLY save the new runnable script ---
            # We add a new config option for this to keep it separate.
            if self.cfg.get("save_merge_artifacts", False):
                utils.save_merge_artifacts(self.cfg, self, final_recipe_node, model_path, iteration)

        except Exception as e:
            logger.error(f"Error during post-merge saving operations: {e}", exc_info=True)

    # V1.1 - Accepts param_info metadata
    def merge(
            self,
            params: Dict[str, Any],  # Flat params from optimizer
            param_info: BoundsInfo,  # <<< ADDED: Full metadata from ParameterHandler
            cache: Optional[Dict],
            iteration: int = 0  # <<< ADD iteration parameter
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
        prepared_model_nodes = self._prepare_model_recipe_args(
            self.models, base_model_node, merge_func
        )

        # 5. Slice model nodes if merge method has fixed arity
        sliced_model_nodes = self._slice_models(prepared_model_nodes, merge_func)

        # 6. Prepare parameter nodes using param_info for expansion
        param_nodes = self._prepare_param_recipe_args(
            params, param_info, merge_func  # Pass metadata here
        )

        # 7. Build the core merge recipe node, applying cache
        logger.info(
            f"Calling '{merge_func.identifier}' with {len(sliced_model_nodes)} model args, {len(param_nodes)} param nodes.")
        core_recipe_node = merge_func(*sliced_model_nodes, **param_nodes).set_cache(cache)

        # 8. Handle potential delta output (wrap with add_difference)
        final_recipe_node = self._handle_delta_output(
            core_recipe_node, base_model_node, merge_func
        )
        # --- End Recipe Building ---

        # 9. Optional steps (save recipe, code, add keys)
        self._save_recipe_etc(final_recipe_node, model_path, iteration)

        # 10. Execute the final recipe (includes fallback logic)
        self._execute_recipe(final_recipe_node, model_path)

        logger.info(f"Merge process completed. Output: {model_path}")
        return model_path

    def recipe_optimization(
            self,
            params: Dict[str, Any],
            param_info: BoundsInfo,
            cache: Optional[Dict],
            iteration: int,
    ) -> Path:
        """
        Orchestrates the optimization of a .mecha recipe.
        This "thinker" method validates, generates node objects, and then calls
        "doer" utilities to rewrite the recipe text and execute.
        """
        logger.info(f"--- Coordinating Recipe Optimization for Iteration {iteration} ---")
        recipe_cfg = self.cfg.recipe_optimization
        recipe_path = Path(recipe_cfg.recipe_path)
        original_recipe_text = recipe_path.read_text(encoding="utf-8")

        # --- Step 1: VALIDATION (The "Thinker" validates its own plan) ---
        target_node_ref = recipe_cfg.target_nodes
        target_node_idx = int(target_node_ref.strip('&'))

        # Deserialize once to get the target node for validation
        all_nodes_map = {
            i: sd_mecha.deserialize(original_recipe_text.split('\n')[:i + 2])
            for i in range(len(original_recipe_text.strip().split('\n')) - 1)
        }
        target_node = all_nodes_map.get(target_node_idx)

        if not isinstance(target_node, MergeRecipeNode):
            raise TypeError(f"Target node {target_node_ref} is not a merge node.")
        valid_params = set(target_node.merge_method.get_param_names().kwargs.keys())
        for param_name in recipe_cfg.target_params:
            if param_name not in valid_params:
                raise ValueError(
                    f"Target parameter '{param_name}' not found in method '{target_node.merge_method.identifier}'.")
        logger.info("Pre-validation successful.")

        # Set output file path
        self.output_file = self.create_model_output_name(iteration=iteration, recipe_node=target_node)
        logger.info(f"Set output path for this iteration to: {self.output_file}")

        # --- Step 2: GENERATION (The "Thinker" creates the node objects) ---
        new_param_nodes = self._prepare_param_recipe_args(params, param_info, target_node.merge_method)

        # --- Step 3: SERIALIZATION (Call a simple utility) ---
        new_node_strings, param_to_final_idx = utils.serialize_nodes_for_rewrite(new_param_nodes)

        # --- Step 4: REWRITING (Call the main "doer" utility) ---
        final_recipe_text = utils.rewrite_recipe_text(
            original_recipe_text=original_recipe_text,
            target_node_idx=target_node_idx,
            new_node_strings=new_node_strings,
            param_to_final_idx=param_to_final_idx
        )

        # --- Step 5: EXECUTION ---
        try:
            final_recipe_node = sd_mecha.deserialize(final_recipe_text)
        except Exception as e:
            debug_path = Path(HydraConfig.get().runtime.output_dir) / f"iteration_{iteration}_failed_recipe.mecha"
            debug_path.write_text(final_recipe_text, encoding="utf-8")
            raise ValueError(f"Final recipe deserialization failed. Saved debug recipe to {debug_path}: {e}")

        # We now perform a DEEP injection of the cache into every merge node.
        logger.info("Performing deep injection of the shared cache into the recipe graph...")
        cache_injector = utils.CacheInjectorVisitor(cache)
        final_recipe_node_with_cache = cache_injector.visit(final_recipe_node)

        model_path = self.output_file
        self._save_recipe_etc(final_recipe_node_with_cache, model_path, iteration)
        self._execute_recipe(final_recipe_node_with_cache, model_path)

        logger.info(f"Recipe optimization coordination complete. Output: {model_path}")
        return model_path

    def layer_adjust(self, params: Dict, cfg: DictConfig) -> Path:  # Takes params
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
