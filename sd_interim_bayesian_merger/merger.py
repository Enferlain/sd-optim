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
from typing import Dict
from omegaconf import DictConfig, open_dict
from sd_interim_bayesian_merger.merge_methods import MergeMethods

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class Merger:
    cfg: DictConfig

    def __post_init__(self) -> None:
        self.validate_config()

        # Add models_dir to the configuration object
        with open_dict(self.cfg):
            self.cfg.models_dir = str(Path(self.cfg.model_paths[0]).parent)

        self.models = self._create_models()
        self.create_model_out_name()
        self.create_best_model_out_name()

    def validate_config(self):
        required_fields = ['model_paths', 'merge_mode', 'model_arch']  # Update required fields
        for field in required_fields:
            if not getattr(self.cfg, field, None):
                raise ValueError(f"Configuration missing required field: {field}")

    def _create_models(self) -> list:
        """Dynamically creates ModelRecipeNodes, handling Loras based on metadata."""
        models = []
        for model_path in self.cfg.get("model_paths", []):
            relative_path = os.path.relpath(model_path, self.cfg.models_dir)

            # Check if it's a safetensors or pt file and try to load metadata
            if model_path.endswith(".safetensors"):
                try:
                    # Open the file safely and retrieve metadata
                    with safetensors.safe_open(model_path, framework="pt") as f:
                        metadata = f.metadata()  # Fetch metadata dictionary
                        logger.debug(f"Metadata for {model_path}: {metadata}")

                        architecture = metadata.get("modelspec.architecture", "")
                        if "lora" in architecture.lower():
                            models.append(sd_mecha.lora(relative_path, self.cfg.model_arch))  # Use sd_mecha.lora
                            logger.info(f"Detected Lora: {model_path}")
                            continue  # Skip to the next model
                except Exception as e:
                    logger.warning(f"Failed to load metadata for {model_path}: {e}")

            models.append(sd_mecha.model(relative_path, self.cfg.model_arch))

        return models

    def create_model_out_name(self, it: int = 0) -> None:
        model_names = [Path(path).stem for path in self.cfg.model_paths]
        combined_name = f"{model_names[0]}-{model_names[1]}-{self.cfg.merge_mode}-it_{it}"
        self.model_out_name = f"bbwm-{combined_name}.safetensors"
        self.output_file = Path(Path(self.cfg.model_paths[0]).parent, self.model_out_name)

    def create_best_model_out_name(self, it: int = 0) -> None:
        model_names = [Path(path).stem for path in self.cfg.model_paths]
        combined_name = f"{model_names[0]}-{model_names[1]}-{self.cfg.merge_mode}-it_{it}_best-fp{self.cfg.best_precision}"
        self.best_output_file = Path(Path(self.cfg.model_paths[0]).parent, f"bbwm-{combined_name}.safetensors")

    def _get_expected_num_models(self) -> int:
        mecha_merge_method = sd_mecha.extensions.merge_method.resolve(self.cfg.merge_mode)
        if mecha_merge_method.get_model_varargs_name() is not None:
            # Return the actual number of models in the configuration, but at least 2
            return max(2, len(self.cfg.model_paths))
        else:
            return len(mecha_merge_method.get_model_names())

    def _select_base_model(self, cfg):
        """Selects the base model, ensuring it's not a Lora."""
        base_model_index = cfg.get("base_model_index", None)

        # Require a valid base_model_index
        if base_model_index is None:
            raise ValueError("A base_model_index must be specified in the configuration.")

        # Check if base_model_index is valid for the number of models
        if base_model_index is not None and base_model_index >= len(self.models):
            raise ValueError(
                f"Invalid base_model_index: {base_model_index}. You specified {len(self.models)} models, but the index refers to a non-existent model. Please ensure a valid base_model_index is provided."
            )

        base_model = self.models[base_model_index] if base_model_index is not None else self.models[-1]

        # Ensure the base model is not a Lora
        if base_model.model_type.identifier == "lora":
            raise ValueError(
                "The specified base model is a Lora. Loras cannot be used as base models. Please choose a different base model."
            )

        return base_model

    def _create_updated_models(self, base_model, input_merge_spaces):
        """Dynamically creates deltas, treating Loras as pre-existing deltas."""
        updated_models = [None] * len(self.models)
        updated_models[self.cfg.base_model_index] = base_model

        input_merge_spaces = input_merge_spaces[:len(self.models)]

        for i, model in enumerate(self.models):
            if i == self.cfg.base_model_index:
                continue

            if model.model_type.identifier == "lora":  # Treat Loras as pre-existing deltas
                updated_models[i] = model
                logger.info(f"Using Lora model {i} as a delta.")
            elif i < len(input_merge_spaces) and input_merge_spaces[
                i] == sd_mecha.recipe_nodes.MergeSpace.DELTA:  # Only convert non-Lora models to deltas if required
                logger.info(f"Creating delta model from model {i} relative to base model.")
                updated_models[i] = sd_mecha.subtract(model, base_model)
            else:
                updated_models[i] = model

        return updated_models

    def _merge_models(self, updated_models, all_hypers, cache):
        """Calls the merging method with appropriate parameters, handling cache if needed."""
        merge_method = getattr(MergeMethods, self.cfg.merge_mode)
        merge_method_signature = inspect.signature(merge_method)
        if 'cache' in merge_method_signature.parameters:
            return merge_method(*updated_models, device=self.cfg.device, cache=cache, **all_hypers)
        else:
            return merge_method(*updated_models, device=self.cfg.device, **all_hypers)

    def _handle_delta_output(self, merged_model, mecha_merge_method, updated_models, base_model):
        """Applies the delta output to the base model if necessary."""
        if mecha_merge_method.get_return_merge_space(
                [model.merge_space for model in updated_models]
        ) == sd_mecha.recipe_nodes.MergeSpace.DELTA:
            logger.info("Applying merged delta to base model.")
            return sd_mecha.add_difference(base_model, merged_model, alpha=1.0)
        return merged_model

    def _serialize_and_save_recipe(self, merged_model, it: int = 0):
        """Serializes and saves the merged model recipe to a file."""
        log_dir = Path(HydraConfig.get().runtime.output_dir)
        recipes_dir = log_dir / "recipes"
        os.makedirs(recipes_dir, exist_ok=True)

        model_names = [Path(path).stem for path in self.cfg.model_paths]
        model_name = f"bbwm-{model_names[0]}-{model_names[1]}-{self.cfg.merge_mode}-it_{it + 1}"  # Increment iteration
        recipe_file_path = recipes_dir / f"{model_name}.mecha"

        with open(recipe_file_path, "w", encoding="utf-8") as f:
            f.write(sd_mecha.recipe_serializer.serialize(merged_model))

    def _execute_sd_mecha_merge(self, merged_model, model_path):
        """Executes the merge using sd-mecha and saves the model."""
        recipe_merger = sd_mecha.RecipeMerger(models_dir=Path(self.cfg.model_paths[0]).parent)
        recipe_merger.merge_and_save(
            merged_model,
            output=model_path,
            threads=self.cfg.threads,
            save_dtype=torch.float16 if self.cfg.best_precision == 16 else torch.float32,
        )
        logging.info(f"Merged model using sd-mecha.")

    def merge(
            self,
            weights_list: Dict,
            base_values: Dict,
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

        base_model = self._select_base_model(cfg)

        # Merge base_values and weights_list into a single dictionary, using the mapping
        all_hypers = {}
        for param_name in weights_list:
            # Get the base value for the current parameter from sd-mecha's defaults
            base_value = base_values.get(f"base_{param_name}", default_hypers.get(param_name, 0.5))

            print(f"param_name: {param_name}, base_value: {base_value}")

            # Merge the base value and block weights into a single dictionary
            all_hypers[param_name] = {
                **{f"{self.cfg.model_arch}_{component}_default": base_value for component in ["txt", "txt2"]},
                **weights_list[param_name]
            }

        print(f"Final all_hypers: {all_hypers}")

        # Unload the currently loaded model
        r = requests.post(url=f"{self.cfg.url}/bbwm/unload-model?webui={self.cfg.webui}")
        r.raise_for_status()

        updated_models = self._create_updated_models(base_model, input_merge_spaces)

        # Check if the number of models is compatible with the merging method, now with updated_models
        num_models = len(updated_models)  # Access models from self.models

        # Get the expected number of models using the helper function
        expected_num_models = self._get_expected_num_models()

        # Convert models to deltas if the method requires delta space for *args or if they are Loras
        if mecha_merge_method.get_model_varargs_name() is not None and (
                all(space == sd_mecha.recipe_nodes.MergeSpace.DELTA for space in input_merge_spaces) or any(
                model.model_type.identifier == "lora" for model in updated_models if model is not None
            )
        ):
            for i, model in enumerate(updated_models):
                if i != self.cfg.base_model_index and model is not None:
                    updated_models[i] = sd_mecha.subtract(model, base_model) if model.model_type.identifier != "lora" else model

        # Exclude the base model if the method requires only deltas
        if mecha_merge_method.get_model_varargs_name() is not None and all(
                space == sd_mecha.recipe_nodes.MergeSpace.DELTA for space in input_merge_spaces
        ):
            updated_models = [model for i, model in enumerate(updated_models) if i != self.cfg.base_model_index]

        # Only slice the models list if the method does NOT accept a variable number of models
        if mecha_merge_method.get_model_varargs_name() is None and num_models != expected_num_models:
            updated_models = updated_models[:expected_num_models]  # Slice self.models
            logger.warning(
                f"Merging method '{self.cfg.merge_mode}' expects {expected_num_models} models, but {num_models} were provided. Using the first {expected_num_models} models."
            )

        merged_model = self._merge_models(updated_models, all_hypers, cache)
        merged_model = self._handle_delta_output(merged_model, mecha_merge_method, updated_models, base_model)

        self._serialize_and_save_recipe(merged_model)
        self._execute_sd_mecha_merge(merged_model, model_path)

        return model_path
