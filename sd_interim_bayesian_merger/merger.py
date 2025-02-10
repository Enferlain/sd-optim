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
from typing import Dict, Optional
from omegaconf import DictConfig, open_dict, OmegaConf
from sd_mecha import recipe_serializer
from sd_mecha.recipe_nodes import RecipeNode

from sd_interim_bayesian_merger import utils
from sd_interim_bayesian_merger.merge_methods import MergeMethods
from sd_interim_bayesian_merger.utils import MergeMethodCodeSaver

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

    def __post_init__(self) -> None:
        self.validate_config()

        # Add models_dir to the configuration object for easy access to the model directory
        with open_dict(self.cfg):
            self.cfg.models_dir = str(Path(self.cfg.model_paths[0]).parent)

        self.models = self._create_models()
        self.create_model_out_name()
        self.create_best_model_out_name()

    def validate_config(self):
        if self.cfg.optimization_mode == "merge":
            if not self.cfg.model_paths or len(self.cfg.model_paths) < 2:
                raise ValueError(
                    "For 'merge' mode, 'model_paths' must be a list containing at least two model paths."
                )
        elif self.cfg.optimization_mode == "recipe":
            if not self.cfg.recipe_optimization.recipe_path:
                raise ValueError(
                    "For 'recipe' mode, 'recipe_optimization.recipe_path' must be specified."
                )
            if not self.cfg.recipe_optimization.target_nodes:
                raise ValueError(
                    "For 'recipe' mode, 'recipe_optimization.target_nodes' must be specified."
                )
        elif self.cfg.optimization_mode == "layer_adjust":
            if not self.cfg.model_paths or len(self.cfg.model_paths) < 1:
                raise ValueError(
                    "For 'layer_adjust' mode, 'model_paths' must be a list containing at least one model path."
                )
        else:
            raise ValueError(f"Invalid optimization mode: {self.cfg.optimization_mode}")

        # merge_mode is only required for "merge" mode
        required_fields = ['model_arch']
        if self.cfg.optimization_mode == "merge":
            required_fields.append('merge_mode')

        missing_fields = [field for field in required_fields if not getattr(self.cfg, field, None)]
        if missing_fields:
            raise ValueError(f"Configuration missing required fields: {', '.join(missing_fields)}")

    def _create_models(self) -> list:
        """Dynamically creates ModelRecipeNodes, handling Loras based on key prefixes."""
        models = []
        for model_path in self.cfg.get("model_paths", []):
            relative_path = os.path.relpath(model_path, self.cfg.models_dir)
            is_lora = False

            if model_path.endswith((".safetensors", ".ckpt")):
                try:
                    with safetensors.safe_open(model_path, framework="pt") as f:
                        # Check for Lora without breaking
                        if any(key.startswith(("lora_unet", "lora_te")) for key in f.keys()):
                            models.append(sd_mecha.lora(relative_path, self.cfg.model_arch))
                            logger.info(f"Detected Lora: {model_path}")
                            is_lora = True
                except Exception as e:
                    logger.warning(f"Failed to open or read model file {model_path}: {e}")

            # Create regular model if not a Lora
            if not is_lora:
                models.append(sd_mecha.model(relative_path, self.cfg.model_arch))

        return models

    def _create_model_output_name(self, it: int = 0, best: bool = False) -> Path:
        """Generates the output file name for the merged model."""
        if self.cfg.optimization_mode == "recipe":
            recipe_path = self.cfg.recipe_optimization.recipe_path
            if not recipe_path:
                raise ValueError("`recipe_path` must be specified when `optimization_mode` is 'recipe'.")

            recipe = sd_mecha.deserialize(recipe_path)
            model_names = utils.get_model_names_from_recipe(recipe)
            if len(model_names) < 2:
                logger.warning(
                    f"Recipe '{recipe_path}' contains less than 2 models. Using available model names for file naming."
                )
                model_names.extend(["unknown_model"] * (2 - len(model_names)))  # Pad with "unknown_model" if necessary

            target_node_key = self.cfg.recipe_optimization.target_nodes
            if isinstance(target_node_key, list):
                target_node_key = target_node_key[0]
            merge_mode = utils.get_merge_mode(recipe_path, target_node_key)
            combined_name = f"{model_names[0]}-{model_names[1]}-{merge_mode}-it_{it}"

        elif self.cfg.optimization_mode == "layer_adjust":
            if not self.cfg.model_paths:
                raise ValueError("`model_paths` must contain at least one model for 'layer_adjust' mode.")
            model_name = Path(self.cfg.model_paths[0]).stem
            combined_name = f"layer_adjusted-{model_name}-it_{it}"
        else:  # Assume "merge" mode
            if not self.cfg.model_paths or len(self.cfg.model_paths) < 2:
                raise ValueError("`model_paths` must contain at least two models for 'merge' mode.")
            model_names = [Path(path).stem for path in self.cfg.model_paths]
            merge_mode = self.cfg.merge_mode
            combined_name = f"{model_names[0]}-{model_names[1]}-{merge_mode}-it_{it}"

        if best:
            combined_name += f"_best-{self.cfg.save_dtype.lower()}"
        return Path(Path(self.cfg.model_paths[0]).parent, f"bbwm-{combined_name}.safetensors")

    def create_model_out_name(self, it: int = 0) -> None:
        self.output_file = self._create_model_output_name(it=it)

    def create_best_model_out_name(self, it: int = 0) -> None:
        self.best_output_file = self._create_model_output_name(it=it, best=True)

    def _get_expected_num_models(self) -> int:
        mecha_merge_method = sd_mecha.extensions.merge_method.resolve(self.cfg.merge_mode)
        if mecha_merge_method.get_model_varargs_name() is not None:
            return len(self.cfg.model_paths)  # Use the actual number of models for *args
        else:
            return len(mecha_merge_method.get_model_names())

    def _requires_base_model(self, mecha_merge_method, input_merge_spaces, varargs_merge_space):
        """Checks if the merging method requires a base model for delta operations."""
        if mecha_merge_method.get_model_varargs_name() is not None:
            return varargs_merge_space == sd_mecha.recipe_nodes.MergeSpace.DELTA
        else:
            return any(space == sd_mecha.recipe_nodes.MergeSpace.DELTA for space in input_merge_spaces)

    def _select_base_model(self, cfg):
        """Selects the base model, ensuring it's not a Lora."""
        base_model_index = cfg.get("base_model_index", None)

        if base_model_index is None:
            raise ValueError(
                "A base_model_index must be specified in the configuration. "
                "Please provide the index (starting from 0) of the desired base model in your configuration file."
            )

        if base_model_index >= len(self.models):
            raise ValueError(
                f"Invalid base_model_index: {base_model_index}. You specified {len(self.models)} models, but the index refers to a non-existent model. Please ensure a valid base_model_index is provided."
            )

        base_model = self.models[base_model_index]

        if base_model.model_type.identifier == "lora":
            raise ValueError(
                f"The selected base model ({base_model.path}) is a Lora. Loras cannot be used as base models. Please choose a different base model from the list."
            )

        return base_model

    def _prepare_models(self, base_model, input_merge_spaces, varargs_merge_space):
        """Handles Lora integration and delta conversion."""
        updated_models = []
        mecha_merge_method = sd_mecha.extensions.merge_method.resolve(self.cfg.merge_mode)
        requires_delta = self._requires_base_model(mecha_merge_method, input_merge_spaces, varargs_merge_space)

        for i, model in enumerate(self.models):
            # Skip the base model only if it's required for delta operations
            if i == self.cfg.base_model_index and requires_delta:
                continue

            if model.model_type.identifier == "lora":
                updated_models.append(model)
                logger.info(f"Using Lora model {i} as a delta.")
            elif requires_delta:
                logger.info(f"Creating delta model from model {i} relative to base model.")
                updated_models.append(sd_mecha.subtract(model, base_model))
            else:
                updated_models.append(model)

        return updated_models

    def _slice_models(self, updated_models):
        """Slices the model list to match the expected number of models."""
        mecha_merge_method = sd_mecha.extensions.merge_method.resolve(self.cfg.merge_mode)
        num_models = len(updated_models)
        expected_num_models = self._get_expected_num_models()

        if mecha_merge_method.get_model_varargs_name() is None and num_models > expected_num_models:
            updated_models = updated_models[:expected_num_models]  # Slice the list
            logger.warning(
                f"Merging method '{self.cfg.merge_mode}' expects {expected_num_models} models, but {num_models} were provided. Using the first {expected_num_models} models."
            )
        return updated_models

    def _merge_models(self, updated_models, assembled_params, cache):
        """Calls the merging method with appropriate parameters, handling cache if needed."""
        logger.info(f"Expected models: {self._get_expected_num_models()}, Actual models: {len(updated_models)}")
        logger.info(f"Attempting to merge {len(updated_models)} models")
        merge_method = getattr(MergeMethods, self.cfg.merge_mode)
        logger.info(f"Using merge method: {self.cfg.merge_mode}")
        merge_method_signature = inspect.signature(merge_method)
        if 'cache' in merge_method_signature.parameters:
            return merge_method(*updated_models, device=self.cfg.device, cache=cache, **assembled_params)
        else:
            return merge_method(*updated_models, device=self.cfg.device, **assembled_params)

    def _handle_delta_output(self, merged_model, mecha_merge_method, updated_models, base_model):
        """Applies the delta output to the base model if necessary."""
        if mecha_merge_method.get_return_merge_space(
                [model.merge_space for model in updated_models]
        ) == sd_mecha.recipe_nodes.MergeSpace.DELTA:
            logger.info("Applying merged delta to base model.")
            return sd_mecha.add_difference(base_model, merged_model, alpha=1.0)
        return merged_model

    def _serialize_and_save_recipe(self, merged_model, model_path):  # Add model_path as a parameter
        """Serializes and saves the merged model recipe to a file."""
        log_dir = Path(HydraConfig.get().runtime.output_dir)
        recipes_dir = log_dir / "recipes"
        os.makedirs(recipes_dir, exist_ok=True)

        # Extract the iteration file name from model_path
        iteration_file_name = model_path.stem
        recipe_file_path = recipes_dir / f"{iteration_file_name}.mecha"  # Use iteration file name for recipe

        with open(recipe_file_path, "w", encoding="utf-8") as f:
            f.write(sd_mecha.recipe_serializer.serialize(merged_model))

    def _execute_sd_mecha_merge(self, merged_model, model_path):
        """Executes the merge using sd-mecha and saves the model."""
        recipe_merger = sd_mecha.RecipeMerger(
            models_dir=Path(self.cfg.model_paths[0]).parent,
            default_dtype=precision_mapping[self.cfg.merge_dtype]
        )
        recipe_merger.merge_and_save(
            merged_model,
            output=model_path,
            threads=self.cfg.threads,
            save_dtype=precision_mapping[self.cfg.save_dtype]
        )
        logging.info(f"Merged model using sd-mecha.")

    def recipe_optimization(
        self,
        assembled_params: Dict,
        model_path: Path,
        device: Optional[str],
        cache: Optional[Dict],
    ) -> Path:
        """Helper function to handle merging when recipe_optimization is enabled."""
        recipe_path = self.cfg.recipe_optimization.recipe_path
        target_nodes = self.cfg.recipe_optimization.target_nodes
        recipe = recipe_serializer.deserialize(recipe_path)

        # Inject cache into the recipe
        visitor = utils.CacheInjectorVisitor(cache)
        modified_recipe = recipe.accept(visitor)

        # Update the target nodes in the modified recipe
        modified_recipe = utils.update_recipe(
            modified_recipe,
            target_nodes,
            assembled_params
        )

        # Get merge method from target node
        target_node_key = self.cfg.recipe_optimization.target_nodes
        if isinstance(target_node_key, list):
            target_node_key = target_node_key[0]

        merge_method_name = utils.get_merge_mode(recipe_path, target_node_key)
        mecha_merge_method = sd_mecha.extensions.merge_method.resolve(merge_method_name)

        # Configure RecipeMerger
        merger = sd_mecha.RecipeMerger(
            models_dir=Path(self.cfg.model_paths[0]).parent,
            default_device=device or self.cfg.device,
        )

        # Check if the merge method has 'cache' in its signature
        merge_method_signature = inspect.signature(mecha_merge_method)
        kwargs = {}
        if 'cache' in merge_method_signature.parameters:
            kwargs['cache'] = cache

        merger.merge_and_save(
            modified_recipe,
            output=model_path,
            threads=self.cfg.threads,
            save_dtype=precision_mapping[self.cfg.save_dtype]
        )
        logging.info(f"Merged model using sd-mecha recipe with {merge_method_name} method.")

        self._serialize_and_save_recipe(modified_recipe, model_path)

        return model_path

    def merge(
            self,
            assembled_params: Dict,
            save_best: bool = False,
            cfg=None,
            device=None,
            cache=None,
            models_dir=None,
    ) -> Path:

        r = requests.post(url=f"{self.cfg.url}/bbwm/unload-model", params={"webui": self.cfg.webui, "url": self.cfg.url})
        r.raise_for_status()

        model_path = self.best_output_file if save_best else self.output_file
        mecha_merge_method = sd_mecha.extensions.merge_method.resolve(self.cfg.merge_mode)
        input_merge_spaces, varargs_merge_space = mecha_merge_method.get_input_merge_spaces()

        if self.cfg.optimization_mode == "recipe":
            return self.recipe_optimization(assembled_params, model_path, device, cache)

        requires_base = self._requires_base_model(mecha_merge_method, input_merge_spaces, varargs_merge_space)
        if requires_base:
            base_model = self._select_base_model(cfg)
        else:
            base_model = None  # or a dummy object, depending on how exactly you want to use it

        updated_models = self._prepare_models(base_model, input_merge_spaces, varargs_merge_space)
        updated_models = self._slice_models(updated_models)
        merged_model = self._merge_models(updated_models, assembled_params, cache)

        if requires_base:
            merged_model = self._handle_delta_output(merged_model, mecha_merge_method, updated_models, base_model)

        self._serialize_and_save_recipe(merged_model, model_path)
        self._execute_sd_mecha_merge(merged_model, model_path)

        if self.cfg.get("save_merge_method_code", False):
            MergeMethodCodeSaver.save_merge_method_code(self.cfg.merge_mode, model_path, MergeMethods)

        # Add extra keys only if the option is enabled
        if self.cfg.get("add_extra_keys", False):
            utils.add_extra_keys(model_path)

        return model_path

    def layer_adjust(self, assembled_params: Dict, cfg: DictConfig) -> Path:
        """Loads a model, applies layer adjustments, and saves the modified model."""
        # Determine model path: use first model from model_paths if not specified
        if not cfg.model_paths:
            raise ValueError("No model paths specified for layer adjustment.")

        model_path = Path(cfg.model_paths[0])

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        # Load the model
        if model_path.suffix == ".safetensors":
            state_dict = safetensors.torch.load_file(model_path, device=self.cfg.device)
        elif model_path.suffix in (".pth", ".pt"):
            state_dict = torch.load(model_path, map_location=self.cfg.device)
        else:
            raise ValueError(f"Unsupported file type: {model_path.suffix}")

        # Determine if the model is an SDXL model
        is_xl_model = cfg.model_arch == "sdxl"

        # Apply color adjustments
        modified_state_dict = utils.modify_state_dict(state_dict, assembled_params, is_xl_model)

        # Save the modified model
        output_path = self.output_file  # You might want to generate a specific name based on parameters
        safetensors.torch.save_file(modified_state_dict, output_path)

        print(f"Modified model saved to {output_path}")
        return output_path
