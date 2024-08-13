import logging
import sd_mecha
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from omegaconf import DictConfig, open_dict
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
        required_fields = ['model_a', 'model_b', 'merge_mode']
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
    ) -> None:

        if save_best:
            with open_dict(self.cfg):
                self.cfg.destination = str(self.best_output_file)
        else:
            with open_dict(self.cfg):
                self.cfg.destination = "memory"

        # Dynamically create ModelRecipeNodes for all models in cfg, using sd_mecha.model
        models = []
        for model_key in ["model_a", "model_b", "model_c"]:
            if model_key in self.cfg:
                model = sd_mecha.model(self.cfg[model_key], self.cfg.model_arch)
                models.append(model)

        # Call the merging method from MergeMethods directly
        merged_model = getattr(MergeMethods, self.cfg.merge_mode)(*models, **base_values, **weights_list)

        # Execute the merge using sd-mecha
        recipe_merger = sd_mecha.RecipeMerger(models_dir=Path(self.cfg.model_a).parent)
        recipe_merger.merge_and_save(
            merged_model,  # Pass the merged model directly
            output=self.cfg.destination,
            threads=self.cfg.threads,
            save_dtype=torch.float16 if self.cfg.best_precision == 16 else torch.float32,
        )

        logging.info(f"Merged model using sd-mecha.")