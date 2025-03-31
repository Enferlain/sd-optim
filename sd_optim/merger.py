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
from typing import Dict, Optional, List, Tuple

from omegaconf import DictConfig, open_dict
from sd_mecha import recipe_serializer, extensions, recipe_nodes
from sd_mecha.extensions.merge_methods import MergeMethod, RecipeNodeOrValue
from sd_mecha.recipe_nodes import ModelRecipeNode

# Assuming utils contains MergeMethodCodeSaver and add_extra_keys
from sd_optim import utils
# Assuming your custom methods are in MergeMethods and decorated correctly
from sd_optim.merge_methods import MergeMethods

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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

    def __post_init__(self) -> None:
        self.validate_config()

        # Ensure models_dir is set relative to the first model path
        if self.cfg.model_paths:
            self.models_dir = Path(self.cfg.model_paths[0]).resolve().parent
        else:
            # Handle cases where model_paths might be empty (e.g., recipe mode)
            # You might need a default or raise an error if models_dir is crucial
            self.models_dir = Path(os.getcwd()) # Default to current dir
            logger.warning("model_paths is empty, setting models_dir to current directory.")

        self.models = self._create_model_nodes() # Create nodes immediately
        self.create_model_out_name()
        self.create_best_model_out_name()

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

    def _prepare_param_recipe_args(self, params: Dict, merge_method: MergeMethod) -> Dict[str, RecipeNodeOrValue]:
        """Wraps optimizer parameters into sd_mecha.literal nodes."""
        param_nodes = {}
        expected_params = merge_method.get_param_names().kwargs # Get keyword param names
        input_types = merge_method.get_input_types().kwargs # Get keyword param types

        for name, value in params.items():
            # Check if this parameter is expected by the merge method's keyword args
            if name in expected_params:
                 expected_type = input_types.get(name)
                 # Wrap the value using sd_mecha.literal
                 # literal() handles basic type conversions (e.g., float to Tensor if needed)
                 param_nodes[name] = sd_mecha.literal(value)
            else:
                 # Check if it matches positional args (less common for optimized params but possible)
                 # This part might be complex if optimized params correspond to positional args
                 # For now, we primarily assume optimized params match keyword args of the merge method
                 logger.warning(f"Parameter '{name}' from optimizer doesn't match any keyword argument of merge method '{merge_method.identifier}'. It might be ignored.")

        return param_nodes

    def _execute_recipe(self, final_recipe_node: recipe_nodes.RecipeNode, model_path: Path):
        """Executes the final recipe using sd_mecha.merge."""
        logger.info(f"Executing merge recipe and saving to: {model_path}")
        try:
            sd_mecha.merge(
                final_recipe_node,
                output=model_path,
                # Pass relevant config options to sd_mecha.merge
                merge_device=self.cfg.get("device"), # Use 'device' from main config for merge
                merge_dtype=precision_mapping[self.cfg.merge_dtype],
                output_device="cpu", # Save to CPU by default
                output_dtype=precision_mapping[self.cfg.save_dtype],
                threads=self.cfg.get("threads"),
                model_dirs=[self.models_dir],
                # Add other relevant sd_mecha.merge options as needed
                # strict_weight_space=..., check_finite=..., etc.
            )
            logging.info(f"Successfully merged and saved model to {model_path}")
        except Exception as e:
             logger.error(f"sd-mecha merge execution failed: {e}", exc_info=True)
             # Re-raise the exception to stop the optimization iteration
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


    def merge(
            self,
            params: Dict,  # Input from the optimizer - ESSENTIAL
            cache: Optional[Dict],  # ADD BACK: Cache from Optimizer
            save_best: bool = False  # Internal flag for filename - Keep
    ) -> Path:
        """Builds and executes a sd-mecha recipe for merging."""

        cfg = self.cfg
        # Use the passed cache parameter, defaulting to an empty dict if None
        cache = cache if cache is not None else {}

        # 1. Determine output path
        model_path = self.best_output_file if save_best else self.output_file

        # --- Recipe Building ---
        logger.info(f"Building merge recipe for mode: {cfg.merge_method}")

        # 2. Resolve merge method
        merge_func = utils.resolve_merge_method(cfg.merge_method)

        # 3. Select base model
        base_model_node = self._select_base_model()

        # 4. Prepare model arguments
        prepared_model_nodes = self._prepare_model_recipe_args(
            self.models,
            base_model_node,
            merge_func
        )

        # 5. Slice models
        sliced_model_nodes = self._slice_models(prepared_model_nodes, merge_func)

        # 6. Prepare parameter arguments
        param_nodes = self._prepare_param_recipe_args(params, merge_func)

        # 7. Build the core merge recipe node, passing the cache
        logger.info(
            f"Calling merge method '{merge_func.identifier}' with {len(sliced_model_nodes)} model args and {len(param_nodes)} param args.")
        core_recipe_node = merge_func(*sliced_model_nodes, **param_nodes).set_cache(cache)  # Use passed cache

        # 8. Handle delta output
        final_recipe_node = self._handle_delta_output(
            core_recipe_node, base_model_node, merge_func
        )
        # --- End Recipe Building ---

        # 9. Execute the final recipe
        self._execute_recipe(final_recipe_node, model_path)

        # 10. Optional: Save recipe/code/extra keys
        self._save_recipe_etc(final_recipe_node, model_path)

        logger.info(f"Merge process completed for iteration. Output: {model_path}")
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