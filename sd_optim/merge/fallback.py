import logging
from typing import TYPE_CHECKING, TypeVar

from sd_mecha import Parameter, Return, StateDict, merge_method, recipe_nodes
from sd_mecha.keys_map import KeyMapBuilder
from sd_mecha.recipe_nodes import ModelRecipeNode
from sd_mecha.streaming import StateDictKeyError

from sd_optim.merge.recipe_inspection import ModelVisitor

if TYPE_CHECKING:
    from sd_optim.merger import Merger

logger = logging.getLogger("sd_optim.merger")
T = TypeVar("T")


@merge_method(identifier="fallback_debug_logged", reuse_outputs=False)
class fallback_debug_logged:
    """Fallback wrapper that mirrors `sd_mecha.fallback` and surfaces first-hit visibility."""

    def __init__(self) -> None:
        self.fallback_hits = 0

    @staticmethod
    def map_keys(builder: KeyMapBuilder) -> None:
        for key in builder.keys():  # noqa: SIM118 - sd_mecha exposes a callable accessor, not a dict view.
            a_inputs = builder.a.keys[key] @ dict.fromkeys(["a"])
            default_inputs = builder.default.keys[key] @ dict.fromkeys(["default"])
            builder[key] = a_inputs & default_inputs | a_inputs | default_inputs

    def __call__(
        self,
        a: Parameter(StateDict[T]),
        default: Parameter(StateDict[T]),
        **kwargs,
    ) -> Return(T):
        (key,), _ = relation = kwargs["key_relation"]
        params = tuple(relation.meta) if relation.meta is not None else ("a", "default")

        for param in params:
            try:
                value = locals()[param][key]
            except StateDictKeyError:
                continue

            if param != "a":
                self.fallback_hits += 1
                if self.fallback_hits == 1:
                    logger.info("Fallback hit 1 for key: %s; per-key fallback logs continue at DEBUG.", key)
                logger.debug("Using fallback for key: %s", key)
            return value

        raise StateDictKeyError(key)


def get_models_for_fallback_lookup(merger: "Merger", final_recipe_node: recipe_nodes.RecipeNode) -> list[ModelRecipeNode]:
    """Return the model nodes available for fallback lookup in the current mode."""
    if merger.cfg.optimization_mode == "merge":
        return merger.models
    if merger.cfg.optimization_mode == "recipe":
        visitor = ModelVisitor()
        final_recipe_node.accept(visitor)
        return visitor.models
    return []


def resolve_fallback_node(
    merger: "Merger",
    final_recipe_node: recipe_nodes.RecipeNode,
    *,
    log_resolution: bool = True,
) -> ModelRecipeNode | None:
    """Resolve the configured fallback node for the current merge context."""
    models_for_lookup = get_models_for_fallback_lookup(merger, final_recipe_node)
    fallback_node: ModelRecipeNode | None = None
    fallback_index = merger.cfg.merge.fallback_model_index

    if fallback_index is None or fallback_index == -1:
        if log_resolution:
            logger.info("No fallback model specified.")
    elif not isinstance(fallback_index, int):
        if log_resolution:
            logger.error(
                "Invalid fallback_model_index type: %s. Must be an integer or null. No fallback will be used.",
                type(fallback_index),
            )
    elif not models_for_lookup:
        if log_resolution:
            logger.error(
                "fallback_model_index %s specified, but no models were found in the current context. No fallback will be used.",
                fallback_index,
            )
    elif not (0 <= fallback_index < len(models_for_lookup)):
        if log_resolution:
            logger.error(
                "Invalid fallback_model_index: %s. Must be between 0 and %s. No fallback will be used.",
                fallback_index,
                len(models_for_lookup) - 1,
            )
    else:
        fallback_node = models_for_lookup[fallback_index]
        if log_resolution:
            logger.info(
                "Using model at index %s ('%s') as fallback source for missing keys.",
                fallback_index,
                fallback_node.path,
            )

    return fallback_node


def build_recipe_to_merge(
    merger: "Merger",
    final_recipe_node: recipe_nodes.RecipeNode,
    *,
    log_resolution: bool = True,
) -> tuple[recipe_nodes.RecipeNode, ModelRecipeNode | None]:
    """Build the effective recipe mecha will execute before graph finalization."""
    fallback_node = resolve_fallback_node(merger, final_recipe_node, log_resolution=log_resolution)
    recipe_to_merge = final_recipe_node
    if fallback_node is not None:
        if log_resolution:
            logger.info("Wrapping final recipe in fallback_debug_logged for per-key fallback visibility.")
        recipe_to_merge = fallback_debug_logged(final_recipe_node, fallback_node)
    return recipe_to_merge, fallback_node


def build_recipe_for_artifacts(merger: "Merger", final_recipe_node: recipe_nodes.RecipeNode) -> recipe_nodes.RecipeNode:
    """Build the logical recipe artifact root without runtime output-cast wrappers."""
    recipe_to_merge, _ = build_recipe_to_merge(merger, final_recipe_node, log_resolution=False)
    return recipe_to_merge
