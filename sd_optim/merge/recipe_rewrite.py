from __future__ import annotations

import json
import logging
import re
from typing import Any

import sd_mecha

from sd_mecha import recipe_nodes

from sd_optim.utils.recipes import serialize_recipe_text

logger = logging.getLogger(__name__)


def serialize_nodes_for_rewrite(
    nodes_dict: dict[str, sd_mecha.recipe_nodes.RecipeNode],
) -> tuple[list[str], dict[str, str]]:
    """
    Serialize nodes for recipe rewriting and return replacement tokens.

    Replacement tokens are either `&N` references to prepended nodes or inline
    literals for scalar-only literal nodes.
    """
    all_new_lines: list[str] = []
    param_to_replacement: dict[str, str] = {}
    current_offset = 0

    for param_name, node in sorted(nodes_dict.items()):
        serialized_text = serialize_recipe_text(node, finalize=False)
        node_lines = serialized_text.strip().split("\n")[1:]

        if not node_lines:
            if isinstance(node, recipe_nodes.LiteralRecipeNode):
                param_to_replacement[param_name] = _to_mecha_inline_literal(_get_inline_literal_value(node))
                continue
            raise ValueError(f"Recipe node for '{param_name}' produced no serializable lines.")

        def shift_ref(match, offset=current_offset):
            original_idx = int(match.group(1))
            return f"&{original_idx + offset}"

        shifted_lines = [re.sub(r"&(\d+)", shift_ref, line) for line in node_lines]
        all_new_lines.extend(shifted_lines)
        param_to_replacement[param_name] = f"&{current_offset + len(node_lines) - 1}"
        current_offset += len(node_lines)

    return all_new_lines, param_to_replacement


def _get_inline_literal_value(node: recipe_nodes.LiteralRecipeNode) -> Any:
    """Extract the scalar payload from a singleton literal node."""
    if len(node.value_dict) != 1:
        raise TypeError("Cannot inline a literal recipe node with multiple values during recipe rewrite.")
    return next(iter(node.value_dict.values()))


def _to_mecha_inline_literal(value: Any) -> str:
    """Serialize a Python scalar into a valid inline .mecha literal token."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (int, float)):
        return str(value)
    raise TypeError(f"Unsupported inline literal type for recipe rewrite: {type(value)}")


def rewrite_recipe_text(
    original_recipe_text: str,
    new_node_strings: list[str],
    param_to_replacement: dict[str, str],
    target_node_idx: int | None = None,
    target_node_indices: list[int] | set[int] | tuple[int, ...] | None = None,
) -> str:
    """Prepend nodes to a recipe text and patch the targeted merge line."""
    original_lines = original_recipe_text.strip().split("\n")[1:]
    num_new_nodes = len(new_node_strings)
    normalized_target_indices = set(target_node_indices or [])
    if target_node_idx is not None:
        normalized_target_indices.add(target_node_idx)

    def shift_old_ref(match):
        original_idx = int(match.group(1))
        return f"&{original_idx + num_new_nodes}"

    rewritten_lines = []
    for index, line in enumerate(original_lines):
        line_base = line.split("#", 1)[0].strip()
        if not line_base:
            continue

        shifted_line = re.sub(r"&(\d+)", shift_old_ref, line_base)

        if index in normalized_target_indices:
            line_to_append = shifted_line
            for param_name, replacement_token in param_to_replacement.items():
                pattern = re.compile(f"({re.escape(param_name)}=)([^ ]+)")
                line_to_append = pattern.sub(
                    lambda match, token=replacement_token: f"{match.group(1)}{token}",
                    line_to_append,
                )
        else:
            line_to_append = shifted_line

        rewritten_lines.append(line_to_append)

    logger.info("Successfully shifted original recipe references and patched target line.")
    return "version 0.1.0\n" + "\n".join(new_node_strings) + "\n" + "\n".join(rewritten_lines)
