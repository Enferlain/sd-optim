from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from sd_mecha import recipe_nodes

from sd_optim.merge.fallback import build_recipe_to_merge
from sd_optim.utils.recipes import merge_with_model_dirs

if TYPE_CHECKING:
    from sd_optim.merger import Merger

logger = logging.getLogger(__name__)

PRECISION_MAPPING = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "fp64": torch.float64,
}


def execute_recipe(
    merger: Merger,
    final_recipe_node: recipe_nodes.RecipeNode,
    model_path: Path,
    *,
    cache_map: dict[recipe_nodes.MergeRecipeNode, dict] | None = None,
) -> None:
    """Execute the final recipe, correctly handling the fallback model for all modes."""
    logger.info("Executing merge recipe and saving to: %s", model_path)
    recipe_to_merge, fallback_node = build_recipe_to_merge(merger, final_recipe_node)

    try:
        if not merger.models_dir or not merger.models_dir.is_dir():
            raise FileNotFoundError("Merger.models_dir is not set or is not a valid directory.")

        logger.info("Calling sd_mecha.merge with recipe-level fallback: %s", fallback_node)
        merge_with_model_dirs(
            model_dirs_to_add=[merger.models_dir],
            recipe=recipe_to_merge,
            output=model_path,
            fallback_model=None,
            merge_device=merger.cfg.merge.device,
            merge_dtype=PRECISION_MAPPING.get(merger.cfg.merge.merge_dtype),
            output_device="cpu",
            output_dtype=PRECISION_MAPPING.get(merger.cfg.merge.save_dtype),
            threads=merger.cfg.merge.threads,
            strict_mandatory_keys=False,
            cache=cache_map,
        )
        logger.info("Successfully merged and saved model to %s", model_path)
    except Exception as error:
        logger.error("sd-mecha merge execution failed: %s", error, exc_info=True)
        raise
