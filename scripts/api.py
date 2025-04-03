# scripts/api.py - V1.2 - Corrected callback, load/unload logic, paths

import logging
import os
import requests # Keep requests for Swarm

# FastAPI imports
from fastapi import FastAPI, Body, HTTPException, Query
# Pydantic for request body validation
from pydantic import BaseModel, Field # Field helps with validation/defaults
from typing import Optional

# --- Conditional WebUI Imports ---
# Try to import modules that only exist within A1111/Forge environment
try:
    from modules import script_callbacks, sd_models, shared
    # Forge specific imports (confirm these are correct for your version)
    from modules_forge import main_entry
    from backend import memory_management
    A1111_FORGE_AVAILABLE = True
    logger = logging.getLogger(__name__) # Get logger AFTER potential module import
    logger.info("A1111/Forge environment detected. API endpoints enabled.")
except ImportError:
    A1111_FORGE_AVAILABLE = False
    # Define logger even if outside environment for consistency
    logger = logging.getLogger(__name__)
    logger.warning("Not running in A1111/Forge context. API endpoints will be inactive.")
    # Define dummy objects if absolutely needed for code analysis, but better to check the flag
    class DummyShared: device = "cpu"; state = None # Add state if needed
    shared = DummyShared()
    script_callbacks = None # Explicitly None

# Configure logger (can be basic here, Forge might override)
logging.basicConfig(level=logging.INFO)

# --- Pydantic Model for Load Request Body ---
class LoadModelPayload(BaseModel):
    model_path: str = Field(..., title="Absolute path to the model file")
    webui: str = Field(..., title="Target WebUI type (a1111, forge, swarm)")
    # target_url is only needed for Swarm, make it optional
    target_url: Optional[str] = Field(None, title="Target URL (required for Swarm)")

# --- Main Callback Function for Registration ---
def sd_optim_api_setup(_gui_unused, app: FastAPI):
    """
    This function is registered with script_callbacks.on_app_started.
    It defines the API endpoints.
    """
    if not A1111_FORGE_AVAILABLE:
        logger.warning("A1111/Forge modules not loaded. Skipping API route registration.")
        return # Don't register routes if environment isn't right

    logger.info("Registering sd-optim API endpoints...")

    @app.post("/sd_optim/load-model", response_model=None, tags=["SD Optim"])
    async def load_model_api(payload: LoadModelPayload = Body(...)):
        """Loads the specified model into the target WebUI."""
        logger.info(f"Received load request: UI='{payload.webui}', Path='{payload.model_path}'")
        webui_lower = payload.webui.lower()

        try:
            if webui_lower == "a1111":
                # --- A1111 Load Logic (from reference) ---
                logger.debug(f"Loading for A1111 using sd_models...")
                checkpoint_info = sd_models.CheckpointInfo(payload.model_path)
                # Prevent reloading if it's already loaded? sd_models might handle this.
                sd_models.load_model(checkpoint_info)
                logger.info(f"A1111: Loaded model '{os.path.basename(payload.model_path)}'")
                # --- End A1111 Logic ---

            elif webui_lower == "forge":
                # --- Forge Load Logic (from reference) ---
                logger.debug(f"Loading for Forge using forge_model_reload...")
                checkpoint_info = sd_models.CheckpointInfo(payload.model_path)
                # Set parameters for Forge loading
                # Ensure model_data exists and has forge_loading_parameters attribute
                if hasattr(sd_models, 'model_data') and hasattr(sd_models.model_data, 'forge_loading_parameters'):
                     sd_models.model_data.forge_loading_parameters = {
                         "checkpoint_info": checkpoint_info,
                         "additional_modules": [] # Add other modules if needed
                     }
                     # Trigger the actual reload mechanism in Forge
                     sd_models.forge_model_reload()
                     # Update UI elements if necessary (e.g., dropdowns)
                     if hasattr(main_entry, 'checkpoint_change'):
                          main_entry.checkpoint_change(os.path.basename(payload.model_path))
                     else:
                          logger.warning("main_entry.checkpoint_change not found.")
                     logger.info(f"Forge: Loaded model '{os.path.basename(payload.model_path)}'")
                else:
                     logger.error("Forge loading structures (sd_models.model_data.forge_loading_parameters) not found.")
                     raise RuntimeError("Forge environment setup appears incorrect.")
                # --- End Forge Logic ---

            elif webui_lower == "swarm":
                # --- Swarm Load Logic (from reference) ---
                if not payload.target_url:
                    raise HTTPException(status_code=400, detail="target_url is required in request body for SwarmUI.")
                target_url = payload.target_url
                api_url = f"{target_url.rstrip('/')}/API/SelectModel" # Ensure no double slash
                model_name = os.path.basename(payload.model_path) # Swarm likely uses name
                logger.debug(f"Sending load request to Swarm: URL='{api_url}', Model='{model_name}'")
                try:
                    # Get new session ID (synchronous requests for simplicity here)
                    session_resp = requests.post(f"{target_url.rstrip('/')}/API/GetNewSession", json={}, timeout=10)
                    session_resp.raise_for_status()
                    session_id = session_resp.json().get("session_id")
                    if not session_id: raise ValueError("Swarm GetNewSession did not return session_id")

                    # Send SelectModel request
                    request_payload = {"session_id": session_id, "model": model_name}
                    response = requests.post(api_url, json=request_payload, timeout=30) # Longer timeout for load
                    response.raise_for_status() # Check for HTTP errors
                    # Check response content if Swarm provides confirmation
                    logger.info(f"SwarmUI: Sent SelectModel request for '{model_name}'")
                except requests.exceptions.RequestException as req_err:
                    logger.error(f"Failed to communicate with SwarmUI at {target_url}: {req_err}")
                    raise HTTPException(status_code=502, detail=f"Failed to contact SwarmUI: {req_err}")
                except Exception as swarm_e:
                     logger.error(f"Error during Swarm load request: {swarm_e}")
                     raise HTTPException(status_code=500, detail=f"Swarm request failed: {swarm_e}")
                # --- End Swarm Logic ---

            elif webui_lower == "comfy":
                logger.warning("ComfyUI model loading via API requires workflow execution, not direct loading.")
                raise HTTPException(status_code=501, detail="ComfyUI loading not supported via this direct API call.")
            else:
                raise HTTPException(status_code=400, detail=f"Invalid WebUI type specified: {payload.webui}")

            # If successful up to here
            return {"message": f"Model load request processed for {payload.webui} / {os.path.basename(payload.model_path)}"}

        except HTTPException as http_err:
             raise http_err # Re-raise specific HTTP errors
        except Exception as e:
            # Catch any other unexpected errors during load process
            logger.error(f"Error processing load model request: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Internal server error processing load request: {e}")

    @app.post("/sd_optim/unload-model", response_model=None, tags=["SD Optim"])
    async def unload_model_api(
            webui: str = Query(..., title="WebUI Type (a1111, forge, swarm)"),
            # URL only needed for Swarm
            target_url: Optional[str] = Query(None, title="Target URL (required for Swarm)")
    ):
        """Unloads the currently loaded model in the specified WebUI."""
        logger.info(f"Received unload request for UI='{webui}'")
        webui_lower = webui.lower()

        try:
            if webui_lower == "a1111":
                # --- A1111 Unload Logic (from reference) ---
                logger.debug("Unloading model for A1111 using sd_models...")
                if hasattr(sd_models, 'unload_model_weights'):
                    sd_models.unload_model_weights()
                    logger.info("A1111: Unloaded model weights.")
                else:
                     logger.warning("sd_models.unload_model_weights not found.")
                # --- End A1111 Logic ---

            elif webui_lower == "forge":
                # --- Forge Unload Logic (from reference) ---
                logger.debug("Unloading model for Forge using memory_management...")
                if hasattr(memory_management, 'free_memory'):
                    # Freeing all should unload the main checkpoint
                    memory_management.free_memory(1e30, shared.device, keep_loaded=[], free_all=True)
                    logger.info("Forge: Called memory_management.free_memory(free_all=True).")
                else:
                    logger.warning("backend.memory_management.free_memory not found.")
                # --- End Forge Logic ---

            elif webui_lower == "swarm":
                # --- Swarm Unload Logic (from reference) ---
                if not target_url:
                    raise HTTPException(status_code=400, detail="target_url query parameter is required for SwarmUI.")
                target_url = target_url.rstrip('/')
                api_url = f"{target_url}/API/FreeBackendMemory"
                logger.debug(f"Sending unload request to Swarm: URL='{api_url}'")
                try:
                    # Get new session ID
                    session_resp = requests.post(f"{target_url}/API/GetNewSession", json={}, timeout=10)
                    session_resp.raise_for_status()
                    session_id = session_resp.json().get("session_id")
                    if not session_id: raise ValueError("Swarm GetNewSession did not return session_id")

                    # Send FreeBackendMemory request
                    request_payload = {"session_id": session_id, "system_ram": False, "backend": "all"}
                    response = requests.post(api_url, json=request_payload, timeout=20) # Slightly longer timeout
                    response.raise_for_status()
                    logger.info("SwarmUI: Sent FreeBackendMemory request.")
                except requests.exceptions.RequestException as req_err:
                    logger.error(f"Failed to communicate with SwarmUI at {target_url}: {req_err}")
                    raise HTTPException(status_code=502, detail=f"Failed to contact SwarmUI: {req_err}")
                except Exception as swarm_e:
                     logger.error(f"Error during Swarm unload request: {swarm_e}")
                     raise HTTPException(status_code=500, detail=f"Swarm request failed: {swarm_e}")
                # --- End Swarm Logic ---

            elif webui_lower == "comfy":
                logger.warning("ComfyUI model unloading not typically done via direct API call.")
                raise HTTPException(status_code=501, detail="ComfyUI unloading not supported via this API.")
            else:
                raise HTTPException(status_code=400, detail=f"Invalid WebUI type specified: {webui}")

            # If successful up to here
            return {"message": f"Model unload request processed for {webui}."}

        except HTTPException as http_err:
             raise http_err # Re-raise specific HTTP errors
        except Exception as e:
            # Catch any other unexpected errors during unload process
            logger.error(f"Error processing unload model request: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Internal server error processing unload request: {e}")

# --- Register the Callback with Script Callbacks ---
# Ensure this runs only if script_callbacks was successfully imported
if script_callbacks and A1111_FORGE_AVAILABLE:
    script_callbacks.on_app_started(sd_optim_api_setup)
    logger.info("sd-optim API setup callback registered.")
else:
    logger.warning("script_callbacks not available or not in A1111/Forge env. API endpoints will not be registered.")
