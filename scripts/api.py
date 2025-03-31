# scripts/api.py - Version 1.1 - Reverted load/unload, adjusted URL handling

import logging
import os
import requests

from fastapi import FastAPI, Body, HTTPException, Query
from pydantic import BaseModel # For request body structure
from typing import Optional


logger = logging.getLogger(__name__)
# Consider making logging level configurable or inherit from main app
logging.basicConfig(level=logging.INFO)


# Conditional imports based on WebUI environment
try:
    # Try A1111/Forge imports first
    from modules import script_callbacks, sd_models, shared
    # Forge specific (confirm these are the correct modules)
    from modules_forge import main_entry
    from backend import memory_management
    A1111_FORGE_AVAILABLE = True
    logger.info("A1111/Forge environment detected.")
except ImportError:
    A1111_FORGE_AVAILABLE = False
    # Define dummy variables or skip logic if needed when running outside A1111/Forge
    class DummyShared: device = "cpu" # Example dummy
    shared = DummyShared()
    script_callbacks = None # Or a dummy object with an on_app_started method
    logger.warning("A1111/Forge environment not detected. API functionality for these UIs will be disabled.")

# --- Request Body Models ---
class LoadModelPayload(BaseModel):
    model_path: str
    webui: str
    # URL is optional, only needed for specific targets like Swarm proxying
    target_url: Optional[str] = None

# --- API Endpoint Definition ---
def api_endpoints(app: FastAPI):
    """Defines API endpoints for model loading/unloading."""

    @app.post("/sd_optim/load-model", response_model=None) # Use sd-optim prefix
    async def load_model_api(payload: LoadModelPayload = Body(...)):
        """Loads a model into the specified WebUI."""
        logger.info(f"Received request to load model on {payload.webui}: {payload.model_path}")
        try:
            if payload.webui.lower() in ["a1111", "forge", "reforge"]:
                if not A1111_FORGE_AVAILABLE:
                     raise HTTPException(status_code=501, detail=f"{payload.webui} environment not detected.")

                # --- Keep verified A1111/Forge Load Logic ---
                checkpoint_info = sd_models.CheckpointInfo(payload.model_path)
                sd_models.load_model(checkpoint_info) # Load the model into memory
                # Forge specific reload/update steps (Verify correctness for Forge)
                if payload.webui.lower() in ["forge", "reforge"]:
                     # main_entry might update UI elements, check if needed after sd_models.load_model
                     main_entry.checkpoint_change(os.path.basename(payload.model_path))
                     # forge_model_reload seems more relevant for applying the loaded model
                     sd_models.forge_model_reload()
                # --- End A1111/Forge Load Logic ---
                logger.info(f"Loaded model in {payload.webui} from {payload.model_path}")

            elif payload.webui.lower() == "swarm":
                 if not payload.target_url:
                      raise HTTPException(status_code=400, detail="target_url is required in request body for SwarmUI operations.")
                 target_url = payload.target_url
                 api_url = f"{target_url}/API/SelectModel"
                 try:
                     # Simplistic session get - robust session management might be needed
                     session_id = requests.post(url=f"{target_url}/API/GetNewSession", json={}, timeout=5).json()["session_id"]
                     # Swarm might expect just the model name
                     request_payload = {"session_id": session_id, "model": os.path.basename(payload.model_path)}
                     response = requests.post(api_url, json=request_payload, timeout=10)
                     response.raise_for_status()
                     logger.info(f"Sent model load request to SwarmUI ({target_url}) for {os.path.basename(payload.model_path)}")
                 except requests.exceptions.RequestException as req_err:
                     logger.error(f"Failed to communicate with SwarmUI at {target_url}: {req_err}")
                     raise HTTPException(status_code=502, detail=f"Failed to contact SwarmUI: {req_err}")

            elif payload.webui.lower() == "comfy":
                 logger.warning("ComfyUI model loading via API requires specific workflow triggers.")
                 raise HTTPException(status_code=501, detail="ComfyUI loading not supported via direct load API call.")
            else:
                raise HTTPException(status_code=400, detail=f"Invalid WebUI type specified: {payload.webui}")

            return {"message": f"Model load request processed for {payload.webui} / {os.path.basename(payload.model_path)}"}

        except HTTPException as http_err:
             raise http_err # Re-raise FastAPI errors
        except Exception as e:
            logger.error(f"Error processing load model request: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Internal server error processing load request: {e}")

    @app.post("/sd_optim/unload-model", response_model=None) # Use sd-optim prefix
    async def unload_model_api(
            webui: str = Query(..., title="WebUI Type"),
            # URL might be needed if proxied request to Swarm
            target_url: Optional[str] = Query(None, title="Target URL (e.g., for Swarm)")
    ):
        """Unloads the currently loaded model in the specified WebUI."""
        logger.info(f"Received request to unload model on {webui}")
        try:
            if webui.lower() in ["a1111", "forge", "reforge"]:
                if not A1111_FORGE_AVAILABLE:
                     raise HTTPException(status_code=501, detail=f"{webui} environment not detected.")

                # --- Keep verified A1111/Forge Unload Logic ---
                # Try both methods for broader compatibility, log if one fails
                unloaded = False
                try: # Forge memory management
                    if hasattr(memory_management, 'free_memory'):
                         memory_management.free_memory(1e30, shared.device, keep_loaded=[], free_all=True)
                         unloaded = True
                         logger.info("Used Forge memory_management to unload.")
                except Exception as forge_e:
                     logger.warning(f"Forge memory_management unload failed (may be harmless): {forge_e}")

                try: # A1111 unload
                    if hasattr(sd_models, 'unload_model_weights'):
                        sd_models.unload_model_weights()
                        unloaded = True
                        logger.info("Used A1111 sd_models.unload_model_weights.")
                except Exception as a1111_e:
                     logger.warning(f"A1111 unload_model_weights failed (may be harmless): {a1111_e}")

                if not unloaded:
                     logger.warning(f"Could not confirm model unload for {webui} using known methods.")
                else:
                     logger.info(f"Model unload executed for {webui}")
                 # --- End A1111/Forge Unload Logic ---

            elif webui.lower() == "swarm":
                 if not target_url:
                      raise HTTPException(status_code=400, detail="target_url query parameter is required for SwarmUI operations.")
                 api_url = f"{target_url}/API/FreeBackendMemory"
                 try:
                     session_id = requests.post(url=f"{target_url}/API/GetNewSession", json={}, timeout=5).json()["session_id"]
                     payload = {"session_id": session_id, "system_ram": False, "backend": "all"}
                     response = requests.post(api_url, json=payload, timeout=10)
                     response.raise_for_status()
                     logger.info(f"Sent model unload request to SwarmUI ({target_url})")
                 except requests.exceptions.RequestException as req_err:
                     logger.error(f"Failed to communicate with SwarmUI at {target_url}: {req_err}")
                     raise HTTPException(status_code=502, detail=f"Failed to contact SwarmUI: {req_err}")

            elif webui.lower() == "comfy":
                 logger.warning("ComfyUI model unloading not typically done via direct API.")
                 raise HTTPException(status_code=501, detail="ComfyUI unloading not supported via this API.")
            else:
                raise HTTPException(status_code=400, detail=f"Invalid WebUI type specified: {webui}")

            return {"message": f"Model unload request processed for {webui}."}

        except HTTPException as http_err:
             raise http_err
        except Exception as e:
            logger.error(f"Error processing unload model request: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Internal server error processing unload request: {e}")

# Register endpoints when the extension loads in A1111/Forge context
if script_callbacks and A1111_FORGE_AVAILABLE:
    # Ensure the callback receives the app instance
    def register_api_routes(app: FastAPI):
        logger.info("Registering sd-optim API endpoints...")
        api_endpoints(app) # Call the function that defines routes

    script_callbacks.on_app_started(register_api_routes)
else:
     logger.warning("Not running in A1111/Forge context or script_callbacks not available. API endpoints not registered via callback.")
