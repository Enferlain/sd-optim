import logging
import sd_mecha
import torch
import os
import requests
import inspect

from hydra.core.hydra_config import HydraConfig
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from omegaconf import DictConfig
from sd_webui_bayesian_merger.merge_methods import MergeMethods

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class Merger:
    cfg: DictConfig

    def __post_init__(self) -> None:
        self.validate_config()
        self.create_model_out_name()
        self.create_best_model_out_name()

    def validate_config(self):
        required_fields = ['model_paths', 'merge_mode', 'model_arch']  # Update required fields
        for field in required_fields:
            if not getattr(self.cfg, field, None):
                raise ValueError(f"Configuration missing required field: {field}")

    def _get_expected_num_models(self, models: list):
        mecha_merge_method = sd_mecha.extensions.merge_method.resolve(self.cfg.merge_mode)
        if mecha_merge_method.get_model_varargs_name() is not None:
            return 2  # Assume a minimum of 2 models for variable-length methods
        else:
            expected_num_models = len(inspect.signature(mecha_merge_method.__call__).parameters)
            num_models = len(models)
            if num_models != expected_num_models:  # Use != for inequality check
                models = models[:expected_num_models]  # Slice the models list
                num_models = len(models)  # Update num_models after slicing
                logger.warning(
                    f"Merging method '{self.cfg.merge_mode}' expects {expected_num_models} models, but {num_models} were provided. Using the first {expected_num_models} models."
                )
            return expected_num_models

    def create_model_out_name(self, it: int = 0) -> None:
        max_filename_length = 255
        reserved_length = len(f"bbwm--it_{it}.safetensors")  # Calculate reserved length dynamically
        max_model_name_length = max_filename_length - reserved_length

        model_names = [Path(path).stem for path in self.cfg.model_paths[:self._get_expected_num_models(models)]]
        combined_name = '-'.join(model_names)[:max_model_name_length]  # Truncate within the join

        self.model_out_name = f"bbwm-{combined_name}-it_{it}.safetensors"
        self.output_file = Path(Path(self.cfg.model_paths[0]).parent, self.model_out_name)

    def create_best_model_out_name(self, it: int = 0) -> None:
        max_filename_length = 255
        reserved_length = len(f"bbwm--it_{it}_best-fp{self.cfg.best_precision}.safetensors")
        max_model_name_length = max_filename_length - reserved_length

        model_names = [Path(path).stem for path in self.cfg.model_paths[:self._get_expected_num_models(models)]]
        combined_name = '-'.join(model_names)[:max_model_name_length]

        self.best_output_file = Path(Path(self.cfg.model_paths[0]).parent,
                                     f"bbwm-{combined_name}-it_{it}_best-fp{self.cfg.best_precision}.safetensors")

    def merge(
            self,
            weights_list: Dict,
            base_values: Dict,
            save_best: bool = False,
            cfg=None,
            device=None,
            models_dir=None,  # Add models_dir as a parameter
    ) -> Path:  # Return the model path

        # Get the merging method's default hyperparameters
        mecha_merge_method = sd_mecha.extensions.merge_method.resolve(self.cfg.merge_mode)
        default_hypers = mecha_merge_method.get_default_hypers()

        # Use the correct model output path based on save_best
        model_path = self.best_output_file if save_best else self.output_file

        # Dynamically create ModelRecipeNodes for all models in cfg, using sd_mecha.model
        models = [sd_mecha.model(os.path.relpath(model_path, models_dir), self.cfg.model_arch) for model_path in
                  self.cfg.get("model_paths", [])]

        # Check if the number of models is compatible with the merging method
        num_models = len(models)

        # Get the expected number of models using the helper function
        expected_num_models = self._get_expected_num_models(models)

        # Only slice the models list if the method does NOT accept a variable number of models
        if mecha_merge_method.get_model_varargs_name() is None and num_models != expected_num_models:
            models = models[:expected_num_models]  # Slice the models list
            num_models = len(models)  # Update num_models after slicing
            logger.warning(
                f"Merging method '{self.cfg.merge_mode}' expects {expected_num_models} models, but {num_models} were provided. Using the first {expected_num_models} models."
            )

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

            print(f"Constructing all_hypers: {all_hypers[param_name]}")  # Move the print statement here

        print(f"Final all_hypers: {all_hypers}")  # Move this print statement outside the loop

        # Unload the currently loaded model
        r = requests.post(url=f"{self.cfg.url}/bbwm/unload-model?webui={self.cfg.webui}")  # Use query parameter
        r.raise_for_status()

        # Call the merging method from MergeMethods directly, passing device and default_dtype
        merged_model = getattr(MergeMethods, self.cfg.merge_mode)(*models, device=self.cfg.device, **all_hypers)

        # Get the Hydra log directory
        log_dir = Path(HydraConfig.get().runtime.output_dir)

        # Create the "recipes" folder if it doesn't exist
        recipes_dir = log_dir / "recipes"
        os.makedirs(recipes_dir, exist_ok=True)

        # Generate the recipe file path
        recipe_file_path = recipes_dir / f"{self.model_out_name}.mecha"

        # Serialize and save the recipe to the file
        with open(recipe_file_path, "w", encoding="utf-8") as f:
            f.write(sd_mecha.recipe_serializer.serialize(merged_model))

        # Execute the merge using sd-mecha and save to the determined model path
        recipe_merger = sd_mecha.RecipeMerger(models_dir=Path(self.cfg.model_paths[0]).parent)
        recipe_merger.merge_and_save(
            merged_model,
            output=model_path,
            threads=self.cfg.threads,
            save_dtype=torch.float16 if self.cfg.best_precision == 16 else torch.float32,
        )

        logging.info(f"Merged model using sd-mecha.")

        return model_path  # Return the model path
