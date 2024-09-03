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

    def create_model_out_name(self, it: int = 0) -> None:
        max_filename_length = 255  # Set the maximum filename length
        reserved_length = len("bbwm-") + len(f"-it_{it}") + len(".safetensors")  # Length of reserved components
        max_model_name_length = max_filename_length - reserved_length

        model_names = [Path(path).stem for path in self.cfg.model_paths]
        combined_name = '-'.join(model_names)

        # Truncate the combined name if it exceeds the limit
        if len(combined_name) > max_model_name_length:
            combined_name = combined_name[:max_model_name_length]

        model_out_name = f"bbwm-{combined_name}-it_{it}.safetensors"
        self.model_out_name = model_out_name
        self.output_file = Path(Path(self.cfg.model_paths[0]).parent, model_out_name)

    def create_best_model_out_name(self, it: int = 0) -> None:
        max_filename_length = 255  # Set the maximum filename length
        reserved_length = len("bbwm-") + len(f"-it_{it}_best") + len(f"-fp{self.cfg.best_precision}") + len(
            ".safetensors")  # Length of reserved components
        max_model_name_length = max_filename_length - reserved_length

        model_names = [Path(path).stem for path in self.cfg.model_paths]
        combined_name = '-'.join(model_names)

        # Truncate the combined name if it exceeds the limit
        if len(combined_name) > max_model_name_length:
            combined_name = combined_name[:max_model_name_length]

        model_out_name = f"bbwm-{combined_name}-it_{it}_best-fp{self.cfg.best_precision}.safetensors"
        self.best_output_file = Path(Path(self.cfg.model_paths[0]).parent, model_out_name)

    def merge(
            self,
            weights_list: Dict,
            base_values: Dict,
            save_best: bool = False,
            cfg=None,
            device=None,
            models_dir=None,  # Add models_dir as a parameter
    ) -> Path:  # Return the model path

        # Use the correct model output path based on save_best
        if save_best:
            model_path = self.best_output_file
        else:
            model_path = self.output_file

        models = []
        i = 1
        while f"model_{i}" in self.cfg:
            model_key = f"model_{i}"
            relative_path = os.path.relpath(self.cfg[model_key], models_dir)
            model = sd_mecha.model(relative_path, self.cfg.model_arch)
            models.append(model)
            i += 1

        # Get the merging method's default hyperparameters
        mecha_merge_method = sd_mecha.extensions.merge_method.resolve(self.cfg.merge_mode)
        default_hypers = mecha_merge_method.get_default_hypers()

        # Merge base_values and weights_list into a single dictionary, using the mapping
        all_hypers = {}
        for param_name in weights_list:
            # Get the base value for the current parameter from sd-mecha's defaults
            base_value = base_values.get(f"base_{param_name}", default_hypers.get(param_name, 0.5)) # Remove [0]

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
