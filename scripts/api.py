import logging
import os
import requests

from fastapi import FastAPI, Body, HTTPException, Query

from modules import script_callbacks, sd_models, shared
from modules_forge import main_entry
from backend import memory_management

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def on_app_started(_gui, api: FastAPI):
    @api.post("/bbwm/load-model", response_model=None)
    async def load_model_api(
            model_path: str = Body(..., title="Path to Model"),
            webui: str = Body(..., title="WebUI Type"),
            url: str = Body(..., title="WebUI URL")
    ):
        """Loads a model into the specified WebUI."""
        try:
            if webui.lower() == "a1111":
                sd_models.load_model(sd_models.CheckpointInfo(model_path))
                logger.info(f"Loaded model in A1111 from {model_path}")
            elif webui.lower() == "forge":
                sd_models.model_data.forge_loading_parameters = {
                    "checkpoint_info": sd_models.CheckpointInfo(model_path),
                    "additional_modules": []
                }
                sd_models.forge_model_reload()
                main_entry.checkpoint_change(os.path.basename(model_path))
                logger.info(f"Loaded model in Forge from {model_path}")
            elif webui.lower() == "swarm":
                # Use requests for SwarmUI
                api_url = f"{url}/API/SelectModel"
                payload = {
                    "session_id": "",
                    "model": model_path
                }
                # Get new session
                session_id = requests.post(url=f"{url}/API/GetNewSession", json={}).json()["session_id"]
                payload["session_id"] = session_id
                response = requests.post(api_url, json=payload)
                response.raise_for_status()
                logger.info(f"Loaded model in SwarmUI from {model_path}")
            elif webui.lower() == "comfy":
                logger.warning("ComfyUI model loading not implemented yet")
            else:
                raise HTTPException(status_code=400, detail="Invalid WebUI type specified")

            return {"message": f"Model loaded successfully on {webui} from: {model_path}"}
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    @api.post("/bbwm/unload-model", response_model=None)
    async def unload_model_api(
            webui: str = Query(..., title="WebUI Type"),
            url: str = Query(..., title="WebUI URL")
    ):
        """Unloads the currently loaded model in the specified WebUI."""
        try:
            if webui.lower() == "a1111":
                sd_models.unload_model_weights()
                logger.info("Unloaded model in A1111")
            elif webui.lower() == "forge":
                memory_management.free_memory(1e30, shared.device, keep_loaded=[], free_all=True)
                logger.info("Unloaded model in Forge")
            elif webui.lower() == "swarm":
                # Use requests for SwarmUI
                api_url = f"{url}/API/FreeBackendMemory"
                payload = {"system_ram": False, "backend": "all", "session_id": ""}
                # Get new session
                session_id = requests.post(url=f"{url}/API/GetNewSession", json={}).json()["session_id"]
                payload["session_id"] = session_id
                response = requests.post(api_url, json=payload)
                response.raise_for_status()
                logger.info("Unloaded model in SwarmUI")
            elif webui.lower() == "comfy":
                logger.warning("ComfyUI model unloading not implemented yet")
            else:
                raise HTTPException(status_code=400, detail="Invalid WebUI type specified")

            return {"message": f"Model unloaded successfully on {webui}."}
        except Exception as e:
            logger.error(f"Error unloading model: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to unload model.")


script_callbacks.on_app_started(on_app_started)
