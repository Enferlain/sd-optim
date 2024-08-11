import fastapi
import logging
from pathlib import Path

import sd_mecha
from modules import script_callbacks
from mecha_recipe_generator import generate_mecha_recipe, translate_optimiser_parameters

logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO)

def on_app_started(_gui, api):
    @api.post("/bbwm/merge-models")
    async def merge_models_api(
        base_values: dict = fastapi.Body(..., title="Base Values"),
        weights_list: dict = fastapi.Body(..., title="Weights List"),
        model_a: str = fastapi.Body(..., title="Path to Model A"),
        model_b: str = fastapi.Body(..., title="Path to Model B"),
        model_c: str = fastapi.Body(None, title="Path to Model C (Optional)"),
        merge_method: str = fastapi.Body(..., title="Merge Method"),
        save_path: str = fastapi.Body(None, title="Save Path"),
    ):
        """Merges models using sd-mecha."""

        # Create the configuration object
        cfg = {
            "model_a": model_a,
            "model_b": model_b,
            "model_arch": cfg.model_arch,  # Get model_arch from cfg
            "merge_mode": merge_method,
        }

        # Add model_c to the configuration if provided
        if model_c is not None:
            cfg["model_c"] = model_c

        # Generate the sd-mecha recipe
        recipe_text = generate_mecha_recipe(base_values, weights_list, merge_method, cfg)

        # Deserialize the recipe
        recipe = sd_mecha.recipe_serializer.deserialize(recipe_text)

        # Execute the recipe using sd-mecha
        recipe_merger = sd_mecha.RecipeMerger()
        recipe_merger.merge_and_save(recipe, output=save_path)

        return {"message": "Models merged successfully."}


script_callbacks.on_app_started(on_app_started)