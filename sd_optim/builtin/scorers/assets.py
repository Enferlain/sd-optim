"""Asset/path management helpers for scorer runtime setup."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import requests
from omegaconf import DictConfig, ListConfig, open_dict

from sd_optim.builtin.scorers.registry import MODEL_DATA

logger = logging.getLogger(__name__)


def setup_evaluator_paths(scorer: Any) -> None:
    """Populate scorer model paths and default per-scorer config values."""
    logger.debug("Setting up evaluator paths...")
    scorer_model_dir_path = Path(scorer.cfg.scorer_model_dir)

    with open_dict(scorer.cfg):
        if not isinstance(scorer.cfg.get("scorer_device"), DictConfig):
            scorer.cfg.scorer_device = {}
        if not isinstance(scorer.cfg.get("scorer_weight"), DictConfig):
            scorer.cfg.scorer_weight = {}

        configured_scorers = scorer.cfg.get("scorer_method", [])
        if not isinstance(configured_scorers, (list, ListConfig)):
            logger.warning("scorer_method is not a list, cannot process evaluators.")
            configured_scorers = []

        for evaluator in configured_scorers:
            evaluator_lower = str(evaluator).lower()
            if evaluator_lower == "manual":
                continue

            model_data_entry = MODEL_DATA.get(evaluator_lower)
            if not model_data_entry:
                logger.warning(
                    "No MODEL_DATA entry for '%s'. Cannot set path or defaults.",
                    evaluator,
                )
                continue

            alt_location = scorer.cfg.get("scorer_alt_location", {}) or {}
            evaluator_alt_config = alt_location.get(evaluator_lower)
            current_model_dir = scorer_model_dir_path
            primary_filename = model_data_entry.get("file_name")

            if isinstance(evaluator_alt_config, (dict, DictConfig)):
                alt_name = evaluator_alt_config.get("model_name")
                alt_dir_str = evaluator_alt_config.get("model_dir")
                if alt_name and alt_dir_str:
                    logger.info(
                        "Using alternative location for '%s': Dir='%s', File='%s'",
                        evaluator,
                        alt_dir_str,
                        alt_name,
                    )
                    try:
                        current_model_dir = Path(alt_dir_str)
                        primary_filename = alt_name
                    except Exception as path_error:
                        logger.warning(
                            "Invalid alternative path for '%s': %s. Using default path.",
                            evaluator,
                            path_error,
                        )
                else:
                    logger.warning(
                        "Alternative location config for '%s' incomplete. Using default path.",
                        evaluator,
                    )

            if primary_filename and evaluator_lower != "aestheticv25":
                try:
                    scorer.model_path[evaluator_lower] = current_model_dir / primary_filename
                except TypeError as path_join_error:
                    logger.error(
                        "Error creating path for '%s': %s. Ensure directory and filename are valid.",
                        evaluator,
                        path_join_error,
                    )
                    continue
            elif evaluator_lower == "aestheticv25":
                logger.debug("No file path needed for '%s'.", evaluator)
            else:
                logger.warning(
                    "MODEL_DATA for '%s' missing primary 'file_name'. Cannot set base path.",
                    evaluator,
                )

            try:
                default_device = scorer.cfg.get("scorer_default_device", "cpu")
                scorer.cfg.scorer_device.setdefault(evaluator_lower, default_device)
                scorer.cfg.scorer_weight.setdefault(evaluator_lower, 1.0)
            except Exception as setdefault_error:
                logger.error(
                    "Error setting default config for '%s': %s",
                    evaluator,
                    setdefault_error,
                )

    logger.debug("Populated model paths: %s", scorer.model_path)
    logger.debug("Final scorer devices: %s", scorer.cfg.get("scorer_device", {}))
    logger.debug("Final scorer weights: %s", scorer.cfg.get("scorer_weight", {}))


def get_models(scorer: Any) -> None:
    """Download required scorer model files if they do not already exist."""
    logger.debug("Checking for necessary scorer model files...")
    scorer_model_dir_path = Path(scorer.cfg.scorer_model_dir)

    med_config_path = scorer_model_dir_path / "med_config.json"
    if (
        any(x.lower() in ["blip", "imagereward"] for x in scorer.cfg.scorer_method)
        and not med_config_path.is_file()
    ):
        logger.info("Downloading med_config.json (needed for BLIP/ImageReward)")
        download_file(
            "https://huggingface.co/THUDM/ImageReward/resolve/main/med_config.json?download=true",
            med_config_path,
        )

    clip_l_path = scorer_model_dir_path / "CLIP-ViT-L-14.pt"
    if (
        any(x.lower() in ["laion", "chad"] for x in scorer.cfg.scorer_method)
        and not clip_l_path.is_file()
    ):
        logger.info("Downloading CLIP ViT-L-14 model (required for Laion/Chad)")
        download_file(
            "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt?raw=true",
            clip_l_path,
        )

    clip_b_path = scorer_model_dir_path / "CLIP-ViT-B-32.safetensors"
    if (
        any(x.lower() in ["wdaes"] for x in scorer.cfg.scorer_method)
        and not clip_b_path.is_file()
    ):
        logger.warning(
            "CLIP ViT-B-32 model needed by WDAes not found at %s. "
            "Please ensure it exists or WDAes might fail.",
            clip_b_path,
        )

    downloaded_this_run: set[str] = set()
    for evaluator in scorer.cfg.scorer_method:
        evaluator_lower = evaluator.lower()
        if evaluator_lower in ["manual", "aestheticv25"]:
            continue

        model_data_entry = MODEL_DATA.get(evaluator_lower)
        if not model_data_entry:
            continue

        filenames_to_check = []
        for key, value in model_data_entry.items():
            if (
                key.endswith("_name")
                or key == "file_name"
                or key in ["class", "real", "anime"]
            ) and isinstance(value, str):
                filenames_to_check.append(value)

        for filename in set(filenames_to_check):
            if not filename or filename in downloaded_this_run:
                continue

            file_path = scorer_model_dir_path / filename
            if not file_path.is_file():
                url = model_data_entry.get("url")
                if filename != model_data_entry.get("file_name"):
                    url = model_data_entry.get(
                        f"url_{filename.split('.')[0].lower()}",
                        url,
                    )

                if url:
                    logger.info("Downloading %s file: %s", evaluator, filename)
                    download_file(url, file_path)
                    downloaded_this_run.add(filename)
                else:
                    logger.warning(
                        "Required file '%s' for scorer '%s' not found and no "
                        "download URL could be determined. Please place it "
                        "manually in '%s'.",
                        filename,
                        evaluator,
                        scorer_model_dir_path,
                    )
            else:
                downloaded_this_run.add(filename)


def download_file(url: str, path: Path) -> None:
    """Download a file to the target path, deleting partial outputs on failure."""
    logger.info("Attempting to download file from %s to %s...", url, path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            chunk_size = 8192
            with open(path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    handle.write(chunk)
        logger.info("Download successful. Saved to %s", path)
    except requests.exceptions.RequestException as request_error:
        logger.error("Failed to download %s: %s", url, request_error)
        if path.exists():
            path.unlink(missing_ok=True)
    except Exception as error:
        logger.error("An unexpected error occurred during download: %s", error)
        if path.exists():
            path.unlink(missing_ok=True)


__all__ = ["download_file", "get_models", "setup_evaluator_paths"]
