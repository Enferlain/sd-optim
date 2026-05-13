from __future__ import annotations

import logging

from dataclasses import dataclass
from typing import Any

from sd_optim.guide_compiler import (
    BoundValue,
    CompiledBinding,
    build_optimizer_bounds,
    compile_bindings,
)
from sd_optim.guide_nodes import GraphGuide, build_graph_guide_specs

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraphRuntimeSummary:
    source_count: int
    build_count: int
    binding_count: int
    compiled_parameter_count: int
    target_space_counts: dict[str, int]
    grouping_counts: dict[str, int]
    bounds_shape_counts: dict[str, int]


@dataclass(frozen=True)
class GraphRuntimeBundle:
    compiled_bindings: tuple[CompiledBinding, ...]
    optimizer_bounds: dict[str, BoundValue]
    summary: GraphRuntimeSummary


def build_graph_runtime_bundle(
    graph: GraphGuide,
    *,
    base_model_config: Any,
    custom_block_config: Any | None,
) -> GraphRuntimeBundle:
    """Compile a graph-authored guide into a runtime bundle for execution paths."""
    sources, bindings = build_graph_guide_specs(
        graph,
        base_model_config=base_model_config,
        custom_block_config=custom_block_config,
    )
    compiled_bindings = tuple(compile_bindings(sources, bindings))
    summary = summarize_graph_runtime(
        graph,
        binding_count=len(bindings),
        compiled_bindings=compiled_bindings,
    )
    optimizer_bounds = build_optimizer_bounds(compiled_bindings)

    logger.info(
        "Prepared graph runtime bundle with %s compiled parameter(s) from %s source(s), %s build(s), and %s resolved binding(s).",
        summary.compiled_parameter_count,
        summary.source_count,
        summary.build_count,
        summary.binding_count,
    )
    return GraphRuntimeBundle(
        compiled_bindings=compiled_bindings,
        optimizer_bounds=optimizer_bounds,
        summary=summary,
    )


def summarize_graph_runtime(
    graph: GraphGuide,
    *,
    binding_count: int,
    compiled_bindings: tuple[CompiledBinding, ...] | list[CompiledBinding],
) -> GraphRuntimeSummary:
    active_nodes = [
        node
        for node in graph.get("nodes", [])
        if isinstance(node, dict) and node.get("enabled", True) is not False
    ]
    source_count = sum(1 for node in active_nodes if node.get("type") == "source")
    build_count = sum(1 for node in active_nodes if node.get("type") == "build")

    target_space_counts: dict[str, int] = {}
    grouping_counts: dict[str, int] = {}
    bounds_shape_counts = {
        "fixed": 0,
        "categorical": 0,
        "continuous": 0,
        "default_bounds_used": 0,
    }

    for binding in compiled_bindings:
        target_space_counts[binding.target_space] = target_space_counts.get(binding.target_space, 0) + 1
        grouping_counts[binding.grouping] = grouping_counts.get(binding.grouping, 0) + 1
        _record_bound_shape(bounds_shape_counts, binding.bounds)

    return GraphRuntimeSummary(
        source_count=source_count,
        build_count=build_count,
        binding_count=binding_count,
        compiled_parameter_count=len(compiled_bindings),
        target_space_counts=target_space_counts,
        grouping_counts=grouping_counts,
        bounds_shape_counts=bounds_shape_counts,
    )


def materialize_target_space_payloads(
    sampled_values: dict[str, Any],
    compiled_bindings: tuple[CompiledBinding, ...] | list[CompiledBinding],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], set[str]]:
    """Split sampled graph values into block/key payloads without legacy bounds metadata."""
    block_payloads: dict[str, dict[str, Any]] = {}
    key_payloads: dict[str, dict[str, Any]] = {}
    handled_method_params: set[str] = set()

    for binding in compiled_bindings:
        if binding.optimizer_param_name not in sampled_values:
            raise KeyError(
                f"Missing sampled value for optimizer parameter '{binding.optimizer_param_name}'."
            )

        handled_method_params.add(binding.method_param_name)
        payload_map = (
            block_payloads.setdefault(binding.method_param_name, {})
            if binding.target_space == "block"
            else key_payloads.setdefault(binding.method_param_name, {})
        )
        value = sampled_values[binding.optimizer_param_name]

        for target in binding.targets:
            if target in payload_map:
                raise ValueError(
                    f"Target '{target}' was assigned multiple values for method parameter "
                    f"'{binding.method_param_name}' in target space '{binding.target_space}'."
                )
            payload_map[target] = value

    logger.info(
        "Materialized graph payloads for %s method parameter(s): %s block payload(s), %s key payload(s).",
        len(handled_method_params),
        len(block_payloads),
        len(key_payloads),
    )
    return block_payloads, key_payloads, handled_method_params


def _record_bound_shape(
    bound_shape_counts: dict[str, int],
    bounds: BoundValue,
) -> None:
    if bounds == (0.0, 1.0):
        bound_shape_counts["default_bounds_used"] += 1
    if isinstance(bounds, list):
        bound_shape_counts["categorical"] += 1
    elif isinstance(bounds, tuple) or (isinstance(bounds, dict) and "range" in bounds):
        bound_shape_counts["continuous"] += 1
    elif isinstance(bounds, (bool, int, float)):
        bound_shape_counts["fixed"] += 1
