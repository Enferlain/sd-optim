import os
import sys
import logging
import torch

from modules import script_callbacks, sd_models, shared

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_model(model_path: str, model_arch: str):
    """Loads a model into A1111 using its API."""

    try:
        # Read the state dictionary using A1111's read_state_dict
        state_dict = sd_models.read_state_dict(model_path)  # Corrected line

        # Get checkpoint info from the model path
        checkpoint_info = sd_models.CheckpointInfo(model_path, model_arch, hash=None)

        # Load the model into A1111
        sd_model = sd_models.load_model(checkpoint_info, already_loaded_state_dict=state_dict)
        logger.info(f"Model loaded successfully from: {model_path}")
        return sd_model

    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        raise

def unload_model():
    """Unloads the currently loaded model in A1111."""

    try:
        sd_models.unload_model_weights()
        logger.info("Model unloaded successfully.")

    except Exception as e:
        logger.error(f"Error unloading model: {e}", exc_info=True)
        raise
