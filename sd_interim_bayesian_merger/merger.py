import logging
import sd_mecha
import torch
import os
import requests
import inspect
import safetensors

from hydra.core.hydra_config import HydraConfig
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from omegaconf import DictConfig, open_dict
from sd_mecha.recipe_nodes import RecipeNode
from sd_interim_bayesian_merger.merge_methods import MergeMethods

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
        required_fields = ['model_paths', 'merge_mode', 'model_arch']
        missing_fields = [field for field in required_fields if not getattr(self.cfg, field, None)]
        if missing_fields:
            raise ValueError(f"Configuration missing required fields: {', '.join(missing_fields)}")

    def _create_models(self) -> list:
        """Dynamically creates ModelRecipeNodes, handling Loras based on key prefixes."""
        models = []
        for model_path in self.cfg.get("model_paths", []):
            relative_path = os.path.relpath(model_path, self.cfg.models_dir)

            if model_path.endswith((".safetensors", ".ckpt")):  # Handle both safetensors and .ckpt files
                try:
                    with safetensors.safe_open(model_path, framework="pt") as f:
                        for key in f.keys():
                            if key.startswith(("lora_unet", "lora_te")):
                                models.append(sd_mecha.lora(relative_path, self.cfg.model_arch))
                                logger.info(f"Detected Lora: {model_path}")
                                break  # Move to the next model after detecting a Lora
                except Exception as e:
                    logger.warning(f"Failed to open or read model file {model_path}: {e}")
            else:
                logger.warning(f"Unsupported model file format: {model_path}")

            # If not detected as a Lora, create a regular model node
            if relative_path not in [model.path for model in models]:
                models.append(sd_mecha.model(relative_path, self.cfg.model_arch))

        return models

    def _create_model_output_name(self, it: int = 0, best: bool = False) -> Path:
        """Generates the output file name for the merged model."""
        model_names = [Path(path).stem for path in self.cfg.model_paths]
        combined_name = f"{model_names[0]}-{model_names[1]}-{self.cfg.merge_mode}-it_{it}"
        if best:
            combined_name += f"_best-{self.cfg.precision.lower()}"
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
        merge_method = getattr(MergeMethods, self.cfg.merge_mode)
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
        recipe_merger = sd_mecha.RecipeMerger(models_dir=Path(self.cfg.model_paths[0]).parent)
        recipe_merger.merge_and_save(
            merged_model,
            output=model_path,
            threads=self.cfg.threads,
            save_dtype=precision_mapping[self.cfg.precision]
        )
        logging.info(f"Merged model using sd-mecha.")

    def merge(
            self,
            assembled_params: Dict,
            save_best: bool = False,
            cfg=None,
            device=None,
            cache=None,
            models_dir=None,
    ) -> Path:

        model_path = self.best_output_file if save_best else self.output_file
        mecha_merge_method = sd_mecha.extensions.merge_method.resolve(self.cfg.merge_mode)
        input_merge_spaces, varargs_merge_space = mecha_merge_method.get_input_merge_spaces()
        default_hypers = mecha_merge_method.get_default_hypers()

        requires_base = self._requires_base_model(mecha_merge_method, input_merge_spaces, varargs_merge_space)
        if requires_base:
            base_model = self._select_base_model(cfg)
        else:
            base_model = None  # or a dummy object, depending on how exactly you want to use it

        r = requests.post(url=f"{self.cfg.url}/bbwm/unload-model?webui={self.cfg.webui}")
        r.raise_for_status()

        updated_models = self._prepare_models(base_model, input_merge_spaces, varargs_merge_space)

        updated_models = self._slice_models(updated_models)

        merged_model = self._merge_models(updated_models, assembled_params, cache)

        if requires_base:
            merged_model = self._handle_delta_output(merged_model, mecha_merge_method, updated_models, base_model)

        self._serialize_and_save_recipe(merged_model, model_path)
        self._execute_sd_mecha_merge(merged_model, model_path)

        return model_path
