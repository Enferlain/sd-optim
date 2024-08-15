import logging
import sd_mecha
import torch
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
        required_fields = ['model_a', 'model_b', 'merge_mode', 'model_arch']
        for field in required_fields:
            if not getattr(self.cfg, field, None):
                raise ValueError(f"Configuration missing required field: {field}")

    def create_model_out_name(self, it: int = 0) -> None:
        model_out_name = f"bbwm-{Path(self.cfg.model_a).stem}-{Path(self.cfg.model_b).stem}"
        model_out_name += f"-it_{it}"
        model_out_name += ".safetensors"
        self.model_out_name = model_out_name
        self.output_file = Path(Path(self.cfg.model_a).parent, model_out_name)

    def create_best_model_out_name(self):
        model_out_name = f"bbwm-{Path(self.cfg.model_a).stem}-{Path(self.cfg.model_b).stem}"
        model_out_name += "-best"
        model_out_name += f"-fp{self.cfg.best_precision}"
        model_out_name += f".safetensors"
        self.best_output_file = Path(Path(self.cfg.model_a).parent, model_out_name)

    def merge(
            self,
            weights_list: Dict,
            base_values: Dict,
            save_best: bool = False,
            cfg=None,  # Add cfg as a parameter
    ) -> Dict:  # Specify the correct return type: Dict

        # Use the correct model output path based on save_best
        if save_best:
            model_path = self.best_output_file
        else:
            model_path = self.output_file

        # Dynamically create ModelRecipeNodes for all models in cfg, using sd_mecha.model
        models = []
        for model_key in ["model_a", "model_b", "model_c"]:
            if model_key in self.cfg:
                model = sd_mecha.model(self.cfg[model_key], self.cfg.model_arch)
                models.append(model)

        # Get the merging method's default hyperparameters
        mecha_merge_method = sd_mecha.extensions.merge_method.resolve(self.cfg.merge_mode)
        default_hypers = mecha_merge_method.get_default_hypers()

        # Merge base_values and weights_list into a single dictionary, using the mapping
        all_hypers = {}
        for param_name in weights_list:
            # Get the base value for the current parameter from sd-mecha's defaults
            base_value = base_values.get(f"base_{param_name}", [default_hypers.get(param_name, 0.5)])[0]

            # Merge the base value and block weights into a single dictionary
            all_hypers[param_name] = {
                **{f"{self.cfg.model_arch}_{component}_default": base_value for component in ["txt", "txt2"]},
                **weights_list[param_name]
            }

        # Call the merging method from MergeMethods directly, passing the combined hyperparameters
        merged_model = getattr(MergeMethods, self.cfg.merge_mode)(*models, **all_hypers)

        # Execute the merge using sd-mecha
        recipe_merger = sd_mecha.RecipeMerger(models_dir=Path(self.cfg.model_a).parent)
        merged_state_dict = {}  # Create an empty dictionary to store the merged state dict
        recipe_merger.merge_and_save(
            merged_model,  # Pass the merged model directly
            output=model_path,  # Save to the determined model path
            threads=self.cfg.threads,
            save_dtype=torch.float16 if self.cfg.best_precision == 16 else torch.float32,
        )

        logging.info(f"Merged model using sd-mecha.")

        return merged_state_dict  # Return the merged state dictionary