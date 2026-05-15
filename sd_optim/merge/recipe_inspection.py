from __future__ import annotations

from typing import Any

import sd_mecha

from sd_mecha import recipe_nodes

from sd_optim.utils.recipes import serialize_recipe_text


class ModelVisitor(recipe_nodes.RecipeVisitor):
    """Find all unique model nodes reachable from a recipe graph."""

    def __init__(self):
        self.models: list[recipe_nodes.ModelRecipeNode] = []
        self.visited: set[recipe_nodes.RecipeNode] = set()

    def visit_model(self, node: recipe_nodes.ModelRecipeNode):
        if node not in self.visited:
            self.models.append(node)
            self.visited.add(node)

    def visit_merge(self, node: recipe_nodes.MergeRecipeNode):
        if node in self.visited:
            return
        self.visited.add(node)
        for arg in node.bound_args.args:
            arg.accept(self)
        for kwarg in node.bound_args.kwargs.values():
            kwarg.accept(self)

    def visit_literal(self, node: recipe_nodes.LiteralRecipeNode):
        if node in self.visited:
            return
        self.visited.add(node)
        for nested in node.value_dict.values():
            if isinstance(nested, recipe_nodes.RecipeNode):
                nested.accept(self)


def get_info_from_target_node(root_node: recipe_nodes.RecipeNode, target_node_ref: str) -> dict[str, Any]:
    """Extract the merge-method identifier and ancestor model paths for a target."""

    def find_node_by_ref(start_node, ref_str):
        target_line_num = int(ref_str.strip("&"))

        def get_all_nodes(node_to_serialize):
            text = serialize_recipe_text(node_to_serialize, finalize=False)
            lines = text.strip().split("\n")
            node_map = {}
            for index in range(1, len(lines)):
                node_map[index - 1] = sd_mecha.deserialize(lines[: index + 1])
            return node_map

        node_map = get_all_nodes(start_node)
        return node_map.get(target_line_num)

    target_node = find_node_by_ref(root_node, target_node_ref)
    if not isinstance(target_node, recipe_nodes.MergeRecipeNode):
        return {}

    model_visitor = ModelVisitor()
    target_node.accept(model_visitor)
    model_names = [model.path for model in model_visitor.models]
    return {
        "method_name": target_node.merge_method.identifier,
        "model_names": model_names,
    }
