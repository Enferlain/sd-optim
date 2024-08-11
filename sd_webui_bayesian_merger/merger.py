import logging
import sd_mecha

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from omegaconf import DictConfig, open_dict
from mecha_recipe_generator import generate_mecha_recipe, translate_optimiser_parameters

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
        model_out_name += f"-{self.cfg.best_precision}bit"
        model_out_name += f".{self.cfg.best_format}"
        self.best_output_file = Path(Path(self.cfg.model_a).parent, model_out_name)

    def merge(
            self,
            weights: Dict,
            bases: Dict,
            save_best: bool = False,
    ) -> None:
        logging.debug(f"Device configuration: {self.cfg.device}")
        logging.debug(f"Work device configuration: {self.cfg.work_device}")

        if save_best:
            with open_dict(self.cfg):
                self.cfg.destination = str(self.best_output_file)
        else:
            with open_dict(self.cfg):
                self.cfg.destination = "memory"

        # Translate parameters using the new function
        base_values, weights_list = translate_optimiser_parameters(bases, weights)

        # Generate sd-mecha recipe using the updated function
        recipe_text = generate_mecha_recipe(
            base_values,
            weights_list,
            self.cfg.merge_mode,
            self.cfg,
        )

        # Deserialize the recipe
        recipe = sd_mecha.recipe_serializer.deserialize(recipe_text)

        # Execute the recipe using sd-mecha
        recipe_merger = sd_mecha.RecipeMerger()
        recipe_merger.merge_and_save(recipe, output=self.cfg.destination)

        logging.info(f"Merged model using sd-mecha.")  # Update log message

        # Remove A1111 API request:
        # r = requests.post(url=f"{self.cfg.url}/bbwm/merge-models", json=option_payload)
        # ... (remove remaining API request code)