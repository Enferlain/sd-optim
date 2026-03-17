from __future__ import annotations

import contextlib
import logging
from pathlib import Path

import sd_mecha

from sd_mecha import recipe_nodes

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def temporary_model_dirs(model_dirs_to_add: list[Path] | tuple[Path, ...]):
    """Temporarily extend sd-mecha's global model-dir registry for graph opening."""
    registry = sd_mecha.extensions.model_dirs._registry
    original_registry = registry.copy()
    try:
        for model_dir in model_dirs_to_add:
            if model_dir not in registry:
                registry.append(model_dir)
        yield
    finally:
        registry[:] = original_registry


@contextlib.contextmanager
def open_model_graph_root(
    node: sd_mecha.recipe_nodes.RecipeNodeOrValue,
    model_dirs_to_add: list[Path] | tuple[Path, ...],
):
    """Open a recipe graph root while honoring the run's temporary model-dir search paths."""
    with temporary_model_dirs(model_dirs_to_add), sd_mecha.open_graph(node, root_only=True) as graph:
        yield graph.root_non_finalized


def get_model_config_candidates(
    node: sd_mecha.recipe_nodes.RecipeNodeOrValue,
    model_dirs_to_add: list[Path] | tuple[Path, ...],
) -> tuple[sd_mecha.extensions.model_configs.ModelConfig, ...]:
    """Return sd-mecha's current root model-config candidates for a node."""
    with temporary_model_dirs(model_dirs_to_add), sd_mecha.open_graph(node, root_only=True) as graph:
        return tuple(graph.root_candidates().model_config)


def convert_with_model_dirs(
    recipe: sd_mecha.recipe_nodes.RecipeNodeOrValue,
    config: str | sd_mecha.extensions.model_configs.ModelConfig | sd_mecha.recipe_nodes.RecipeNode,
    *,
    model_dirs_to_add: list[Path] | tuple[Path, ...],
):
    """Call `sd_mecha.convert` while temporarily registering model search paths."""
    with temporary_model_dirs(model_dirs_to_add):
        return sd_mecha.convert(recipe, config)


def merge_with_model_dirs(
    *,
    model_dirs_to_add: list[Path] | tuple[Path, ...],
    **merge_kwargs,
):
    """Call `sd_mecha.merge` while temporarily registering model search paths."""
    with temporary_model_dirs(model_dirs_to_add):
        return sd_mecha.merge(**merge_kwargs)


def serialize_recipe_text(
    node: sd_mecha.recipe_nodes.RecipeNode,
    *,
    model_dirs_to_add: list[Path] | tuple[Path, ...] = (),
    finalize: bool = False,
) -> str:
    """Serialize a recipe graph while honoring temporary model directories."""
    with temporary_model_dirs(model_dirs_to_add):
        return sd_mecha.serialize(node, finalize=finalize)


def finalize_recipe_with_model_dirs(
    node: sd_mecha.recipe_nodes.RecipeNode,
    *,
    model_dirs_to_add: list[Path] | tuple[Path, ...] = (),
    model_config_preference: tuple[str, ...] = ("singleton-mecha",),
    merge_space_preference: list[sd_mecha.extensions.merge_spaces.MergeSpace] | tuple[sd_mecha.extensions.merge_spaces.MergeSpace, ...] | None = None,
    check_extra_keys: bool = True,
    check_mandatory_keys: bool = False,
) -> sd_mecha.recipe_nodes.RecipeNode:
    """Finalize a recipe graph while honoring temporary model directories."""
    with temporary_model_dirs(model_dirs_to_add), sd_mecha.open_graph(node) as graph:
        return graph.finalize_root(
            model_config_preference=model_config_preference,
            merge_space_preference=merge_space_preference,
            check_extra_keys=check_extra_keys,
            check_mandatory_keys=check_mandatory_keys,
        )


class MergeNodeCollector(recipe_nodes.RecipeVisitor):
    """Collect each unique MergeRecipeNode reachable from a recipe graph."""

    def __init__(self) -> None:
        self.nodes: list[recipe_nodes.MergeRecipeNode] = []
        self.visited: set[recipe_nodes.RecipeNode] = set()

    def visit(self, node: recipe_nodes.RecipeNode):
        if node not in self.visited:
            self.visited.add(node)
            node.accept(self)

    def visit_model(self, node: recipe_nodes.ModelRecipeNode):
        return None

    def visit_literal(self, node: recipe_nodes.LiteralRecipeNode):
        for nested in node.value_dict.values():
            if isinstance(nested, recipe_nodes.RecipeNode):
                self.visit(nested)
        return None

    def visit_merge(self, node: recipe_nodes.MergeRecipeNode):
        self.nodes.append(node)
        for child in (*node.bound_args.args, *node.bound_args.kwargs.values()):
            self.visit(child)
        return None


def build_recipe_cache_map(
    root_node: recipe_nodes.RecipeNode,
    shared_cache: dict,
) -> dict[recipe_nodes.MergeRecipeNode, dict]:
    """Build a node->cache mapping for sd-mecha 1.1.x merge execution."""
    collector = MergeNodeCollector()
    collector.visit(root_node)
    return dict.fromkeys(collector.nodes, shared_cache)


class RelativeModelPathVisitor(recipe_nodes.RecipeVisitor):
    """Rewrite model node paths to be relative to a chosen base directory when possible."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir.resolve()
        self.visited: dict[recipe_nodes.RecipeNode, recipe_nodes.RecipeNode] = {}

    def rewrite(self, node: recipe_nodes.RecipeNode) -> recipe_nodes.RecipeNode:
        cached = self.visited.get(node)
        if cached is not None:
            return cached
        rewritten = node.accept(self)
        self.visited[node] = rewritten
        return rewritten

    def visit_literal(self, node: recipe_nodes.LiteralRecipeNode) -> recipe_nodes.LiteralRecipeNode:
        value_dict = {
            key: self.rewrite(value) if isinstance(value, recipe_nodes.RecipeNode) else value
            for key, value in node.value_dict.items()
        }
        return recipe_nodes.LiteralRecipeNode(value_dict, node.model_config, node.merge_space)

    def visit_model(self, node: recipe_nodes.ModelRecipeNode) -> recipe_nodes.ModelRecipeNode:
        path = node.path.resolve()
        try:
            relative_path = path.relative_to(self.base_dir)
        except ValueError:
            relative_path = node.path

        if node.is_open:
            return recipe_nodes.OpenModelRecipeNode(node.state_dict, relative_path, node.model_config, node.merge_space)
        return recipe_nodes.ClosedModelRecipeNode(relative_path, node.model_config, node.merge_space)

    def visit_merge(self, node: recipe_nodes.MergeRecipeNode) -> recipe_nodes.MergeRecipeNode:
        args = tuple(self.rewrite(value) for value in node.bound_args.args)
        kwargs = {key: self.rewrite(value) for key, value in node.bound_args.kwargs.items()}
        bound_args = node.merge_method.get_signature().bind(*args, **kwargs)
        return recipe_nodes.MergeRecipeNode(node.merge_method, bound_args, node.model_config, node.merge_space)


def relativize_model_paths(
    node: recipe_nodes.RecipeNode,
    *,
    base_dir: Path,
) -> recipe_nodes.RecipeNode:
    """Return a recipe graph with model paths rewritten relative to `base_dir` when possible."""
    return RelativeModelPathVisitor(base_dir).rewrite(node)
