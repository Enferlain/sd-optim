from __future__ import annotations

import logging
import re
from pathlib import Path

import torch

from omegaconf import DictConfig, ListConfig
from sd_mecha.recipe_nodes import MergeRecipeNode
from .methods import resolve_merge_method

logger = logging.getLogger(__name__)

PRECISION_MAPPING = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "fp64": torch.float64,
}

def validate_run_config(cfg: DictConfig) -> None:
    """Validate the runtime configuration before optimizer startup."""
    logger.info("Validating run configuration...")

    models_dir_str = cfg.get("models_dir")
    if not models_dir_str:
        raise ValueError("'models_dir' must be set in your config.yaml.")
    models_dir = Path(models_dir_str).resolve()
    if not models_dir.is_dir():
        raise FileNotFoundError(
            f"The specified models_dir is invalid or not found: {models_dir}"
        )

    if cfg.optimization_mode == "merge":
        if not cfg.model_paths or len(cfg.model_paths) < 1:
            raise ValueError(
                "For 'merge' mode, 'model_paths' must contain at least one model path."
            )
        if not cfg.merge_method:
            raise ValueError(
                "Configuration missing required field: 'merge_method' for 'merge' mode."
            )
    elif cfg.optimization_mode == "recipe":
        recipe_cfg = cfg.get("recipe_optimization")
        if not recipe_cfg:
            raise ValueError(
                "`optimization_mode` is 'recipe', but 'recipe_optimization' section is missing."
            )

        recipe_path_str = recipe_cfg.get("recipe_path")
        target_nodes_raw = recipe_cfg.get("target_nodes")
        target_params_list = recipe_cfg.get("target_params")

        if not recipe_path_str:
            raise ValueError("Recipe optimization requires 'recipe_path'.")
        if not target_nodes_raw:
            raise ValueError("Recipe optimization requires 'target_nodes'.")
        if not target_params_list:
            raise ValueError("Recipe optimization requires 'target_params'.")

        recipe_path = Path(recipe_path_str)
        if not recipe_path.exists():
            raise FileNotFoundError(f"Recipe file does not exist: {recipe_path}")

        try:
            if isinstance(target_nodes_raw, str):
                target_nodes_list = [target_nodes_raw]
            elif isinstance(target_nodes_raw, (list, ListConfig)):
                target_nodes_list = list(target_nodes_raw)
            else:
                raise TypeError(
                    f"target_nodes must be a string or a list, but got {type(target_nodes_raw)}"
                )

            logger.debug(
                "Performing advanced validation on recipe for targets: %s",
                target_nodes_list,
            )
            original_recipe_text = recipe_path.read_text(encoding="utf-8")
            all_lines = original_recipe_text.strip().split("\n")

            for target_node_str in target_nodes_list:
                target_node_idx = int(target_node_str.strip("&"))
                if not (0 <= target_node_idx < len(all_lines) - 1):
                    raise IndexError(
                        f"target_nodes entry '{target_node_str}' is out of bounds for the recipe."
                    )

                target_line = all_lines[target_node_idx + 1]
                match = re.search(r'merge\s+"([^"]+)"', target_line)
                if not match:
                    raise TypeError(
                        f"Target node {target_node_str} does not appear to be a valid merge line."
                    )

                method_name = match.group(1)
                method_obj = resolve_merge_method(method_name)
                param_names = method_obj.get_param_names()
                valid_params = set(param_names.args) | set(param_names.kwargs.keys())

                for param_name in target_params_list:
                    if param_name not in valid_params:
                        raise ValueError(
                            f"For target '{target_node_str}', parameter '{param_name}' is not valid for method '{method_obj.identifier}'. "
                            f"Valid params are: {sorted(valid_params)}"
                        )
            logger.debug(
                "Advanced recipe validation successful for all target nodes."
            )
        except (ValueError, IndexError, TypeError, FileNotFoundError) as error:
            raise ValueError(
                f"Recipe configuration validation failed: {error}"
            ) from error
        except Exception as error:  # noqa: BLE001 - preserve broad validation guard.
            logger.error(
                "Unexpected error validating recipe file: %s",
                error,
                exc_info=True,
            )
            raise ValueError("Unexpected error during recipe validation.") from error

        if cfg.get("model_paths"):
            logger.info(
                "NOTE: In 'recipe' mode, `model_paths` is only used to locate the `models_dir`."
            )
    elif cfg.optimization_mode == "layer_adjust":
        if not cfg.model_paths or len(cfg.model_paths) < 1:
            raise ValueError(
                "`model_paths` must contain at least one model for 'layer_adjust' mode."
            )
    else:
        raise ValueError(f"Invalid optimization_mode: '{cfg.optimization_mode}'")

    if not hasattr(cfg, "merge_dtype") or cfg.merge_dtype not in PRECISION_MAPPING:
        raise ValueError(
            f"Invalid 'merge_dtype': '{cfg.get('merge_dtype')}'. Must be one of {list(PRECISION_MAPPING.keys())}"
        )
    if not hasattr(cfg, "save_dtype") or cfg.save_dtype not in PRECISION_MAPPING:
        raise ValueError(
            f"Invalid 'save_dtype': '{cfg.get('save_dtype')}'. Must be one of {list(PRECISION_MAPPING.keys())}"
        )

    logger.info("Configuration successfully validated.")


def _get_valid_params_for_node(node: MergeRecipeNode) -> list[str]:
    """Return valid keyword parameter names for a merge node."""
    if not isinstance(node, MergeRecipeNode):
        return []
    param_names = node.merge_method.get_param_names()
    return list(param_names.kwargs.keys())
