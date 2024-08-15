import fastapi
import logging
from pathlib import Path

import sd_mecha
from modules import script_callbacks, sd_models
from sd_webui_bayesian_merger.merge_methods import MergeMethods

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
        model_arch: str = fastapi.Body(..., title="Model Architecture"),
        save_path: str = fastapi.Body(None, title="Save Path"),
    ):
        """Merges models using sd-mecha and saves the result."""

        # Create the configuration object
        cfg = {
            "model_a": model_a,
            "model_b": model_b,
            "model_arch": model_arch,
            "merge_mode": merge_method,
        }

        # Add model_c to the configuration if provided
        if model_c is not None:
            cfg["model_c"] = model_c

        # Create a list of sd-mecha models
        models = [
            sd_mecha.model(cfg[model_key], model_arch)
            for model_key in ["model_a", "model_b", "model_c"] if model_key in cfg
        ]

        # Call the merging method using MergeMethods class
        merged_model = getattr(MergeMethods, merge_method)(*models, **base_values, **weights_list)

        # Execute the merge using sd-mecha and save to the specified path
        recipe_merger = sd_mecha.RecipeMerger(models_dir=Path(cfg['model_a']).parent)
        recipe_merger.merge_and_save(merged_model, output=save_path)

        return {"message": "Models merged successfully."}

    @api.post("/bbwm/load-model")
    async def load_model_api(
            model_path: str = fastapi.Body(..., title="Path to Model"),
            unload_default: bool = fastapi.Body(False, title="Unload Default Model"),  # Add unload_default parameter
    ):
        """Loads a model into A1111."""

        if unload_default:
            sd_models.unload_model_weights()  # Unload the default model
            print("Unloaded default model to free up memory.")  # Provide user feedback

        sd_models.load_model(sd_models.CheckpointInfo(model_path))
        return {"message": f"Model loaded successfully from: {model_path}"}

    @api.post("/bbwm/unload-model")
    async def unload_model_api():
        """Unloads the currently loaded model in A1111."""

        sd_models.unload_model_weights()
        return {"message": "Model unloaded successfully."}


script_callbacks.on_app_started(on_app_started)