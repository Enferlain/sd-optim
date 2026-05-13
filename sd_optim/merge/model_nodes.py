from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import sd_mecha

from omegaconf import ListConfig
from sd_mecha.recipe_nodes import ModelRecipeNode

if TYPE_CHECKING:
    from sd_optim.merger import Merger

logger = logging.getLogger(__name__)


def create_model_nodes(merger: Merger) -> list[ModelRecipeNode]:
    """
    Create sd_mecha ModelRecipeNodes from the config.

    This preserves relative recipe paths where possible while handling cross-drive
    edge cases on Windows.
    """
    model_nodes: list[ModelRecipeNode] = []
    model_paths_list = merger.cfg.merge.model_paths

    if not isinstance(model_paths_list, (list, ListConfig)):
        logger.warning("'model_paths' in config is not a list. No model nodes will be created.")
        model_paths_list = []

    if not model_paths_list:
        logger.info("No model paths provided, returning an empty list of model nodes.")
        return model_nodes

    if not hasattr(merger, "models_dir") or not merger.models_dir or not merger.models_dir.is_dir():
        logger.error("Cannot create model nodes: models_dir '%s' is invalid.", getattr(merger, "models_dir", "Not Set"))
        return model_nodes

    logger.info("Creating model nodes relative to base directory: %s", merger.models_dir)
    for model_path_str in model_paths_list:
        try:
            if not isinstance(model_path_str, str):
                logger.warning("Skipping non-string path in model_paths: %s", model_path_str)
                continue

            original_path = Path(model_path_str)
            resolved_path: Path | None = None

            if original_path.is_absolute():
                if original_path.exists():
                    resolved_path = original_path
                else:
                    logger.error("Absolute model path not found: %s. Skipping node.", original_path)
                    continue
            else:
                path_in_models_dir = merger.models_dir / original_path
                if path_in_models_dir.exists():
                    resolved_path = path_in_models_dir
                elif original_path.exists():
                    resolved_path = original_path.resolve()
                else:
                    logger.error("Relative model path not found: '%s'. Skipping node.", original_path)
                    continue

            try:
                path_for_recipe = os.path.relpath(resolved_path, merger.models_dir)
            except ValueError:
                logger.warning(
                    "Could not create relative path for %s (likely on a different drive). Using absolute path in recipe.",
                    resolved_path,
                )
                path_for_recipe = str(resolved_path)

            logger.debug("Resolved path '%s' to recipe path '%s'", resolved_path, path_for_recipe)
            node = sd_mecha.model(path_for_recipe)

            if isinstance(node, ModelRecipeNode):
                model_nodes.append(node)
            else:
                logger.warning("Node created for path '%s' was not a ModelRecipeNode. Skipping.", path_for_recipe)
        except Exception as error:
            logger.error("Failed to create node for path '%s': %s", model_path_str, error, exc_info=True)
            continue

    logger.info("Finished creating nodes. Total successful: %s.", len(model_nodes))
    return model_nodes
