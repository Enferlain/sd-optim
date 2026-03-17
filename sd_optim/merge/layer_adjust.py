from __future__ import annotations

import logging

from pathlib import Path
from typing import TYPE_CHECKING, Any

import safetensors.torch
import torch

from omegaconf import DictConfig

from sd_optim.utils.images import modify_state_dict

if TYPE_CHECKING:
    from sd_optim.merger import Merger

logger = logging.getLogger(__name__)


def resolve_layer_adjust_output_path(merger: Merger, cfg: DictConfig) -> Path:
    """Return the output path for layer-adjust mode, creating a fallback when unset."""
    output_path = merger.output_file
    if output_path is not None:
        return output_path

    logger.error("Output file path not set in Merger before layer_adjust call.")
    model_name_for_fallback = Path(cfg.model_paths[0]).stem if cfg.model_paths else "unknown_model"
    output_path = merger.models_dir / f"layer_adjusted_{model_name_for_fallback}_fallback.safetensors"
    logger.warning("Using fallback output path: %s", output_path)
    merger.output_file = output_path
    return output_path


def resolve_layer_adjust_model_path(*, models_dir: Path, model_path_str: str) -> Path:
    """Resolve the source model path for layer-adjust mode, preferring the models_dir fallback."""
    model_path = Path(model_path_str)
    if model_path.is_file():
        return model_path

    resolved_path = models_dir / model_path_str
    if resolved_path.is_file():
        logger.info("Resolved layer_adjust model path to: %s", resolved_path)
        return resolved_path

    raise FileNotFoundError(f"Model for layer_adjust not found at '{model_path_str}' or '{resolved_path}'")


def load_layer_adjust_state_dict(model_path: Path, device: str) -> dict[str, Any]:
    """Load a checkpoint for layer-adjust mode from supported formats."""
    if model_path.suffix == ".safetensors":
        return safetensors.torch.load_file(model_path, device=device)
    if model_path.suffix in (".ckpt", ".pth", ".pt"):
        state_dict = torch.load(model_path, map_location=device)
        return state_dict.get("state_dict", state_dict)
    raise ValueError(f"Unsupported file type for layer adjustment: {model_path.suffix}")


def detect_is_xl_model(state_dict: dict[str, Any]) -> bool:
    """Return whether the loaded checkpoint appears to be an SDXL model."""
    return any("conditioner.embedders.1" in key for key in state_dict)


def layer_adjust(merger: Merger, params: dict[str, Any], cfg: DictConfig) -> Path:
    """Load a model, apply layer adjustments, and save the modified state dict."""
    output_path = resolve_layer_adjust_output_path(merger, cfg)

    if not cfg.model_paths:
        raise ValueError("No model paths specified for layer adjustment.")

    model_path = resolve_layer_adjust_model_path(
        models_dir=Path(cfg.models_dir),
        model_path_str=cfg.model_paths[0],
    )

    logger.info("Loading model for layer adjustment: %s", model_path)
    try:
        state_dict = load_layer_adjust_state_dict(model_path, cfg.device)
    except Exception as error:
        logger.error("Failed to load model %s: %s", model_path, error, exc_info=True)
        raise

    is_xl_model = detect_is_xl_model(state_dict)
    logger.info("Determined model type for layer adjustment: %s", "SDXL" if is_xl_model else "Non-SDXL")

    logger.info("Applying layer adjustments...")
    try:
        modified_state_dict = modify_state_dict(state_dict, params, is_xl_model)
    except Exception as error:
        logger.error("Error applying layer adjustments: %s", error, exc_info=True)
        raise

    logger.info("Saving adjusted model to %s", output_path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        safetensors.torch.save_file(modified_state_dict, output_path)
    except Exception as error:
        logger.error("Failed to save adjusted model %s: %s", output_path, error, exc_info=True)
        raise

    logger.info("Layer adjusted model saved successfully to %s", output_path)
    return output_path
