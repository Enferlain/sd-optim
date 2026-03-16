from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import sd_mecha

from sd_mecha import recipe_nodes

from sd_optim import utils

if TYPE_CHECKING:
    from sd_optim.merger import Merger

logger = logging.getLogger(__name__)


def _sanitize_recipe_text_for_deserialize(recipe_text: str) -> str:
    """Normalize recipe text before sd_mecha deserialization."""
    return "\n".join(line for line in recipe_text.splitlines() if line.strip())


def get_model_config_cache_key(merger: Merger, node: recipe_nodes.ModelRecipeNode) -> tuple[str, str]:
    """Build a stable cache key for inferred model-config candidates."""
    return (type(node).__name__, str(node.path))


def get_model_config_candidates_cached(
    merger: Merger,
    node: recipe_nodes.ModelRecipeNode,
) -> tuple[sd_mecha.extensions.model_configs.ModelConfig, ...]:
    """Return cached sd-mecha model-config candidates for a model node."""
    cache_key = get_model_config_cache_key(merger, node)
    cached = merger._model_config_candidates_cache.get(cache_key)
    if cached is not None:
        return cached

    inferred_candidates = utils.get_model_config_candidates(node, [merger.models_dir])
    candidates = tuple(inferred_candidates)
    merger._model_config_candidates_cache[cache_key] = candidates
    return candidates


def is_adapter_model_config(merger: Merger, config: sd_mecha.extensions.model_configs.ModelConfig) -> bool:
    """Return True when a config represents an adapter-style model rather than a base checkpoint."""
    identifier = getattr(config, "identifier", "")
    implementation = ""
    get_impl = getattr(config, "get_implementation_identifier", None)
    if callable(get_impl):
        try:
            implementation = get_impl()
        except Exception:
            implementation = ""
    return identifier.endswith("_lora") or implementation.endswith("_lora")


def get_adapter_candidate_ids(merger: Merger, node: recipe_nodes.ModelRecipeNode) -> tuple[str, ...]:
    """Return the inferred adapter-style config identifiers for a model node."""
    candidates = get_model_config_candidates_cached(merger, node)
    return tuple(cfg.identifier for cfg in candidates if is_adapter_model_config(merger, cfg))


def validate_node_is_not_lora(
    merger: Merger,
    node: recipe_nodes.ModelRecipeNode,
    *,
    raise_error: bool = True,
) -> None:
    """Check whether a model node appears to be a LoRA/LyCORIS."""
    try:
        if get_adapter_candidate_ids(merger, node):
            raise ValueError(f"Model '{node.path}' appears to be a LoRA/LyCORIS and cannot be used as a base/context model.")
    except ValueError as error:
        if raise_error:
            raise error
        raise
    except Exception as error:
        logger.error("Could not verify model '%s' for LoRA check: %s", node.path, error)
        if raise_error:
            raise RuntimeError(f"Could not verify model '{node.path}'.") from error
        raise


def select_base_model(merger: Merger) -> recipe_nodes.ModelRecipeNode | None:
    """Select the base model node based on configuration index."""
    base_model_index = merger.cfg.get("base_model_index", None)
    if base_model_index is None:
        return None

    if not isinstance(base_model_index, int) or not (0 <= base_model_index < len(merger.models)):
        raise ValueError(f"Invalid base_model_index: {base_model_index}. Must be an integer within range [0, {len(merger.models) - 1}).")

    base_model_node = merger.models[base_model_index]

    try:
        if not hasattr(merger, "models_dir") or not merger.models_dir:
            raise FileNotFoundError("Merger's models_dir attribute is not set.")

        if get_adapter_candidate_ids(merger, base_model_node):
            raise ValueError(
                f"The selected base model ('{base_model_node.path}') appears to be a LoRA/LyCORIS. These cannot be used as base models."
            )
    except (ValueError, FileNotFoundError) as error:
        logger.error("Error during base model validation: %s", error)
        raise
    except Exception as error:
        logger.error(
            "Unexpected error during base model config inference for LoRA check: %s",
            error,
            exc_info=True,
        )
        raise ValueError(f"Could not verify base model '{base_model_node.path}'. Halting.") from error

    return base_model_node


def get_conversion_context_node(merger: Merger) -> recipe_nodes.ModelRecipeNode:
    """
    Find a suitable, reliable ModelRecipeNode to serve as a conversion target.

    This is critical for converting custom blocks to the base model's key space.
    """
    logger.debug("Attempting to find a suitable model for conversion context...")

    if merger.cfg.optimization_mode == "merge":
        base_model = select_base_model(merger)
        if base_model:
            logger.info("Using explicitly defined base_model '%s' as conversion context.", base_model.path)
            return base_model

    fallback_index = merger.cfg.get("fallback_model_index", -1)
    if fallback_index != -1 and 0 <= fallback_index < len(merger.models):
        fallback_model = merger.models[fallback_index]
        validate_node_is_not_lora(merger, fallback_model)
        logger.info(
            "Using fallback_model at index %s ('%s') as conversion context.",
            fallback_index,
            fallback_model.path,
        )
        return fallback_model

    if merger.cfg.optimization_mode == "recipe":
        logger.debug("Scanning recipe for the first valid base model to use as context...")
        recipe_path = Path(merger.cfg.recipe_optimization.recipe_path)
        original_recipe_text = recipe_path.read_text(encoding="utf-8")
        sanitized_recipe_text = _sanitize_recipe_text_for_deserialize(original_recipe_text)

        recipe_graph = sd_mecha.deserialize(sanitized_recipe_text)
        visitor = utils.ModelVisitor()
        recipe_graph.accept(visitor)

        for model_node in visitor.models:
            try:
                validate_node_is_not_lora(merger, model_node, raise_error=False)
                logger.info("Found suitable model '%s' inside recipe to use as conversion context.", model_node.path)
                return model_node
            except ValueError:
                logger.debug("Skipping '%s' as conversion context because it appears to be a LoRA.", model_node.path)
                continue

    if merger.models:
        first_model = merger.models[0]
        validate_node_is_not_lora(merger, first_model)
        logger.warning(
            "Could not find an explicit base/fallback model. Using the first model in model_paths ('%s') as a last-resort conversion context.",
            first_model.path,
        )
        return first_model

    raise RuntimeError("Could not determine any suitable model to use for conversion context.")

