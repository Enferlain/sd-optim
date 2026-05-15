from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sd_optim.bounds import BoundsInfo
from sd_optim.guide_runtime import GraphRuntimeBundle
from sd_optim.merge.artifacts import save_recipe_artifacts
from sd_optim.merge.execution import execute_recipe
from sd_optim.merge.model_selection import select_base_model
from sd_optim.merge.recipe_builder import (
    handle_delta_output,
    prepare_model_recipe_args,
    prepare_param_recipe_args,
    slice_models,
)
from sd_optim.utils.methods import resolve_merge_method
from sd_optim.utils.recipes import build_recipe_cache_map

if TYPE_CHECKING:
    from sd_optim.merger import Merger

logger = logging.getLogger(__name__)
GuideRuntimeInput = BoundsInfo | GraphRuntimeBundle


def _resolve_output_path(merger: Merger) -> Path:
    """Return the current output path, falling back to a deterministic default."""
    if merger.output_file is not None:
        return merger.output_file

    logger.error("Output file path not set in Merger before merge call.")
    model_path = merger.models_dir / f"merge_output_default_{merger.cfg.merge.merge_method}.safetensors"
    logger.warning("Using default output path: %s", model_path)
    merger.output_file = model_path
    return model_path


def run_merge(
    merger: Merger,
    params: dict[str, Any],
    param_info: GuideRuntimeInput,
    cache: dict | None,
    iteration: int = 0,
) -> Path:
    """Build and execute an sd-mecha merge recipe for one optimizer iteration."""
    cache = cache if cache is not None else {}
    logger.info("Starting merge process for iteration %s", iteration)

    model_path = _resolve_output_path(merger)
    logger.debug("Building merge recipe for method: %s", merger.cfg.merge.merge_method)

    merge_func = resolve_merge_method(merger.cfg.merge.merge_method)
    base_model_node = select_base_model(merger)
    logger.info("Selected base model node: %s", base_model_node.path if base_model_node else "None")

    prepared_model_nodes = prepare_model_recipe_args(merger, merger.models, base_model_node, merge_func)
    sliced_model_nodes = slice_models(merger, prepared_model_nodes, merge_func)
    param_nodes = prepare_param_recipe_args(merger, params, param_info, merge_func)

    logger.info(
        "Calling '%s' with %s model args, %s param nodes.",
        merge_func.identifier,
        len(sliced_model_nodes),
        len(param_nodes),
    )
    core_recipe_node = merge_func(*sliced_model_nodes, **param_nodes)
    final_recipe_node = handle_delta_output(merger, core_recipe_node, base_model_node, merge_func)
    cache_map = build_recipe_cache_map(final_recipe_node, cache)

    save_recipe_artifacts(merger, final_recipe_node, model_path, iteration)
    execute_recipe(merger, final_recipe_node, model_path, cache_map=cache_map)

    logger.info("Merge process completed. Output: %s", model_path)
    return model_path
