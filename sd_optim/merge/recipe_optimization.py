from __future__ import annotations

import logging

from pathlib import Path
from typing import TYPE_CHECKING, Any

import sd_mecha

from hydra.core.hydra_config import HydraConfig
from sd_mecha.recipe_nodes import MergeRecipeNode

from sd_optim.bounds import BoundsInfo
from sd_optim.merge.artifacts import save_recipe_artifacts
from sd_optim.merge.execution import execute_recipe
from sd_optim.merge.recipe_builder import prepare_param_recipe_args
from sd_optim.utils.artifacts import rewrite_recipe_text, serialize_nodes_for_rewrite
from sd_optim.utils.recipes import build_recipe_cache_map

if TYPE_CHECKING:
    from sd_optim.merger import Merger

logger = logging.getLogger(__name__)


def sanitize_recipe_text_for_deserialize(recipe_text: str) -> str:
    """
    Normalize recipe text before sd_mecha deserialization.

    sd_mecha's parser raises on empty lines, so we drop blank/whitespace-only lines.
    """
    return "\n".join(line for line in recipe_text.splitlines() if line.strip())


def load_validated_target_node(merger: Merger, original_recipe_text: str) -> MergeRecipeNode:
    """Deserialize and validate the configured target merge node for recipe optimization."""
    recipe_cfg = merger.cfg.recipe_optimization
    target_node_ref = recipe_cfg.target_nodes
    target_node_idx = int(target_node_ref.strip("&"))

    all_nodes_map = {
        index: sd_mecha.deserialize(original_recipe_text.split("\n")[: index + 2])
        for index in range(len(original_recipe_text.strip().split("\n")) - 1)
    }
    target_node = all_nodes_map.get(target_node_idx)

    if not isinstance(target_node, MergeRecipeNode):
        raise TypeError(f"Target node {target_node_ref} is not a merge node.")

    param_names = target_node.merge_method.get_param_names()
    valid_params = set(param_names.args) | set(param_names.kwargs.keys())

    for param_name in recipe_cfg.target_params:
        if param_name not in valid_params:
            raise ValueError(
                f"Target parameter '{param_name}' not found in method '{target_node.merge_method.identifier}'."
            )

    logger.info("Pre-validation successful.")
    return target_node


def recipe_optimization(
    merger: Merger,
    params: dict[str, Any],
    param_info: BoundsInfo,
    cache: dict | None,
    iteration: int,
) -> Path:
    """
    Orchestrate optimization of a .mecha recipe by rewriting target parameter nodes and executing the result.
    """
    logger.info("--- Coordinating Recipe Optimization for Iteration %s ---", iteration)
    cache = cache if cache is not None else {}
    recipe_cfg = merger.cfg.recipe_optimization
    recipe_path = Path(recipe_cfg.recipe_path)
    original_recipe_text = recipe_path.read_text(encoding="utf-8")

    target_node = load_validated_target_node(merger, original_recipe_text)
    target_node_ref = recipe_cfg.target_nodes
    target_node_idx = int(target_node_ref.strip("&"))

    merger.output_file = merger.create_model_output_name(iteration=iteration, recipe_node=target_node)
    logger.info("Set output path for this iteration to: %s", merger.output_file)

    new_param_nodes = prepare_param_recipe_args(merger, params, param_info, target_node.merge_method)
    new_node_strings, param_to_replacement = serialize_nodes_for_rewrite(new_param_nodes)

    final_recipe_text = rewrite_recipe_text(
        original_recipe_text=original_recipe_text,
        target_node_idx=target_node_idx,
        new_node_strings=new_node_strings,
        param_to_replacement=param_to_replacement,
    )

    try:
        final_recipe_node = sd_mecha.deserialize(sanitize_recipe_text_for_deserialize(final_recipe_text))
    except Exception as error:
        debug_path = Path(HydraConfig.get().runtime.output_dir) / f"iteration_{iteration}_failed_recipe.mecha"
        debug_path.write_text(final_recipe_text, encoding="utf-8")
        raise ValueError(f"Final recipe deserialization failed. Saved debug recipe to {debug_path}: {error}") from error

    cache_map = build_recipe_cache_map(final_recipe_node, cache)
    model_path = merger.output_file
    save_recipe_artifacts(merger, final_recipe_node, model_path, iteration)
    execute_recipe(merger, final_recipe_node, model_path, cache_map=cache_map)

    logger.info("Recipe optimization coordination complete. Output: %s", model_path)
    return model_path
