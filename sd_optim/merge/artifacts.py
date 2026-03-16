from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import sd_mecha

from hydra.core.hydra_config import HydraConfig
from sd_mecha import recipe_nodes
from sd_mecha.recipe_nodes import RecipeNode

from sd_optim import utils
from sd_optim.merge.fallback import build_recipe_for_artifacts

if TYPE_CHECKING:
    from sd_optim.merger import Merger

logger = logging.getLogger(__name__)


def create_model_output_name(
    merger: Merger,
    iteration: int,
    *,
    best: bool = False,
    recipe_node: RecipeNode | None = None,
) -> Path:
    """
    Generate the output file name for the merged model based on the optimization mode.

    Uses full names and applies a hard cutoff only if the final name is excessively long.
    """
    combined_name = "fallback_merge_name"
    max_filename_len = 120

    if merger.cfg.optimization_mode == "merge":
        model_names = [Path(path).stem for path in merger.cfg.model_paths]

        if len(model_names) >= 2:
            name_part = f"{model_names[0]}-{model_names[1]}"
        elif len(model_names) == 1:
            name_part = model_names[0]
        else:
            name_part = "merged_model"

        merge_method_name = merger.cfg.merge_method
        combined_name = f"{name_part}-{merge_method_name}-it_{iteration}"
    elif merger.cfg.optimization_mode == "layer_adjust":
        model_name = Path(merger.cfg.model_paths[0]).stem
        combined_name = f"layer_adjusted-{model_name}-it_{iteration}"
    elif merger.cfg.optimization_mode == "recipe":
        if recipe_node is not None:
            recipe_cfg = merger.cfg.get("recipe_optimization", {})
            target_node_ref = recipe_cfg.get("target_nodes")
            node_info = utils.get_info_from_target_node(recipe_node, target_node_ref)

            if node_info:
                method_name = node_info["method_name"]
                model_names = [Path(name).stem for name in node_info["model_names"]]

                if len(model_names) >= 2:
                    name_part = f"{model_names[0]}-{model_names[1]}"
                elif len(model_names) == 1:
                    name_part = model_names[0]
                else:
                    name_part = "optimized_merge"

                combined_name = f"{name_part}-{method_name}-it_{iteration}"
            else:
                recipe_name = Path(recipe_cfg.get("recipe_path", "unknown")).stem
                combined_name = f"recipe_{recipe_name}-it_{iteration}"
        else:
            combined_name = f"recipe_fallback-it_{iteration}"

    if best:
        combined_name += "_best"

    if len(combined_name) > max_filename_len:
        logger.warning("Generated filename is too long. Truncating to %s characters.", max_filename_len)
        combined_name = combined_name[:max_filename_len]

    output_dir = merger.models_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{combined_name}.safetensors"


def serialize_and_save_recipe(merger: Merger, final_recipe_node: recipe_nodes.RecipeNode, model_path: Path) -> None:
    """Serialize and save the merged model recipe to a file."""
    try:
        log_dir = Path(HydraConfig.get().runtime.output_dir)
    except ValueError:
        log_dir = Path(os.getcwd()) / "logs" / "unknown_run"
        logger.warning("Hydra config not found, saving recipe to default log directory.")

    recipes_dir = log_dir / "recipes"
    os.makedirs(recipes_dir, exist_ok=True)

    iteration_file_name = model_path.stem
    recipe_file_path = recipes_dir / f"{iteration_file_name}.mecha"

    try:
        recipe_for_artifact = build_recipe_for_artifacts(merger, final_recipe_node)
        finalized_recipe = utils.finalize_recipe_with_model_dirs(
            recipe_for_artifact,
            model_dirs_to_add=[merger.models_dir],
            model_config_preference=("singleton-mecha",),
            merge_space_preference=sd_mecha.extensions.merge_spaces.get_all(),
            check_mandatory_keys=False,
        )
        artifact_recipe = utils.relativize_model_paths(finalized_recipe, base_dir=merger.models_dir)
        serialized_recipe = utils.serialize_recipe_text(
            artifact_recipe,
            model_dirs_to_add=[merger.models_dir],
            finalize=False,
        )
        with open(recipe_file_path, "w", encoding="utf-8") as file:
            file.write(serialized_recipe)
        logger.info("Saved recipe to %s", recipe_file_path)
    except Exception as error:
        logger.error("Failed to serialize or save recipe: %s", error)


def save_recipe_artifacts(
    merger: Merger,
    final_recipe_node: recipe_nodes.RecipeNode,
    model_path: Path,
    iteration: int,
) -> None:
    """Handle optional saving of merge recipe artifacts."""
    try:
        serialize_and_save_recipe(merger, final_recipe_node, model_path)

        if merger.cfg.get("save_merge_artifacts", False):
            artifact_recipe_node = build_recipe_for_artifacts(merger, final_recipe_node)
            utils.save_merge_artifacts(merger.cfg, merger, artifact_recipe_node, model_path, iteration)
    except Exception as error:
        logger.error("Error during post-merge saving operations: %s", error, exc_info=True)

