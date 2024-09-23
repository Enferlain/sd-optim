import fastapi
import logging
import os
from pathlib import Path

import sd_mecha

from modules import script_callbacks, sd_models, shared
from modules_forge import main_entry
from backend import memory_management

from sd_interim_bayesian_merger.merge_methods import MergeMethods

logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO)


def on_app_started(_gui, api):
    @api.post("/bbwm/merge-models")
    async def merge_models_api(
        base_values: dict = fastapi.Body(..., title="Base Values"),
        weights_list: dict = fastapi.Body(..., title="Weights List"),
        model_paths: list[str] = fastapi.Body(..., title="Paths to Models"),
        merge_method: str = fastapi.Body(..., title="Merge Method"),
        model_arch: str = fastapi.Body(..., title="Model Architecture"),
        save_path: str = fastapi.Body(None, title="Save Path"),
        webui: str = fastapi.Body(..., title="WebUI Type"),  # Add webui parameter
    ):
        """Merges models using sd-mecha and saves the result."""

        # Create the configuration object
        cfg = {
            "model_paths": model_paths,  # Use the model_paths list directly
            "model_arch": model_arch,
            "merge_mode": merge_method,
            "webui": webui,
        }

        # Create a list of sd-mecha models
        models = [
            sd_mecha.model(model_path, model_arch)
            for model_path in cfg["model_paths"]
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
            webui: str = fastapi.Body(..., title="WebUI Type"),  # Add webui parameter
    ):
        """Loads a model into the specified WebUI."""
        if webui == "a1111":
            sd_models.load_model(sd_models.CheckpointInfo(model_path))
            print(f"Bayesian Merger: Loaded model from {model_path}")
        elif webui == "forge":
            # Directly update forge_loading_parameters
            sd_models.model_data.forge_loading_parameters = {
                "checkpoint_info": sd_models.CheckpointInfo(model_path),  # Create a CheckpointInfo object
                "additional_modules": []  # Add other parameters as needed
            }

            # Call forge_model_reload to load the model
            sd_models.forge_model_reload()

            # Set the loaded model as the active model using checkpoint_change
            main_entry.checkpoint_change(os.path.basename(model_path))

            print(f"Bayesian Merger: Loaded model from {model_path} in Forge")
        else:
            raise fastapi.HTTPException(status_code=400, detail="Invalid WebUI type specified")
        return {"message": f"Model loaded successfully from: {model_path}"}

    @api.post("/bbwm/unload-model")
    async def unload_model_api(
        webui: str = fastapi.Query(..., title="WebUI Type"),
    ):
        """Unloads the currently loaded model in A1111 or Forge."""

        try:
            if webui == "a1111":
                sd_models.unload_model_weights()
                print("Bayesian Merger: Unloaded model in A1111")
            elif webui == "forge":
                # Use memory_management.free_memory for selective unloading
                memory_management.free_memory(1e30, shared.device, keep_loaded=[], free_all=True)
                print("Bayesian Merger: Unloaded model in Forge")
            else:
                raise fastapi.HTTPException(status_code=400, detail="Invalid WebUI type specified")

            return {"message": "Model unloaded successfully."}
        except Exception as e:
            logger.error(f"Error unloading model: {e}", exc_info=True)
            return {"message": "Failed to unload model."}  # Return an error message


script_callbacks.on_app_started(on_app_started)
