from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from sd_optim.guide_compiler import (
    BindingSpec,
    BoundValue,
    CompiledBinding,
    SelectionSpec,
    TargetSource,
    TargetSpace,
    compile_bindings,
)

GraphGuide = Mapping[str, Any]
GraphNode = Mapping[str, Any]


@dataclass(frozen=True)
class _ResolvedBranch:
    source: TargetSource
    component_name: str
    branch_type: str
    targets: tuple[str, ...]
    param_name: str | None
    bounds: BoundValue
    shared_group_name: str | None
    has_explicit_selection: bool
    path_index: int


def compile_graph_guide_to_bindings(
    graph: GraphGuide,
    *,
    base_model_config: Any,
    custom_block_config: Any | None,
) -> list[CompiledBinding]:
    """Compile build-centered graph guide branches into optimizer-visible bindings."""
    sources, bindings = build_graph_guide_specs(
        graph,
        base_model_config=base_model_config,
        custom_block_config=custom_block_config,
    )
    return compile_bindings(sources, bindings)


def build_graph_guide_specs(
    graph: GraphGuide,
    *,
    base_model_config: Any,
    custom_block_config: Any | None,
) -> tuple[list[TargetSource], list[BindingSpec]]:
    """Build canonical source and binding specs from build-centered guide nodes."""
    node_map = _active_node_map(graph)
    incoming = _active_incoming_edges(graph, node_map)
    sources = _build_sources(
        node_map,
        base_model_config=base_model_config,
        custom_block_config=custom_block_config,
    )

    bindings: list[BindingSpec] = []
    for build_node in node_map.values():
        if build_node.get("type") != "build":
            continue
        build_paths = _walk_paths_to_build(build_node["id"], incoming, node_map)
        resolved_branches = [
            _resolve_branch_path(path, sources)
            for path in build_paths
        ]
        bindings.extend(_compile_build_branches(str(build_node["id"]), resolved_branches))

    return list(sources.values()), bindings


def _active_node_map(graph: GraphGuide) -> dict[str, GraphNode]:
    nodes = graph.get("nodes", [])
    if not isinstance(nodes, list):
        raise ValueError("Graph guide 'nodes' must be a list.")

    active_nodes: dict[str, GraphNode] = {}
    for node in nodes:
        if not isinstance(node, Mapping):
            raise ValueError("Graph guide node entries must be mappings.")
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id:
            raise ValueError("Graph guide nodes must define a non-empty string 'id'.")
        if node.get("enabled", True) is False:
            continue
        if node_id in active_nodes:
            raise ValueError(f"Duplicate graph guide node id '{node_id}'.")
        active_nodes[node_id] = node
    return active_nodes


def _active_incoming_edges(
    graph: GraphGuide,
    node_map: dict[str, GraphNode],
) -> dict[str, list[str]]:
    edges = graph.get("edges", [])
    if not isinstance(edges, list):
        raise ValueError("Graph guide 'edges' must be a list.")

    incoming: dict[str, list[str]] = {node_id: [] for node_id in node_map}
    for edge in edges:
        if not isinstance(edge, Mapping):
            raise ValueError("Graph guide edge entries must be mappings.")
        edge_from = edge.get("from")
        edge_to = edge.get("to")
        if not isinstance(edge_from, str) or not isinstance(edge_to, str):
            raise ValueError("Graph guide edges must define string 'from' and 'to'.")
        if edge_from not in node_map or edge_to not in node_map:
            continue
        incoming.setdefault(edge_to, []).append(edge_from)
    return incoming


def _walk_paths_to_build(
    build_node_id: str,
    incoming: dict[str, list[str]],
    node_map: dict[str, GraphNode],
) -> list[list[GraphNode]]:
    paths: list[list[GraphNode]] = []

    def visit(node_id: str, path: list[GraphNode], seen: set[str]) -> None:
        node = node_map[node_id]
        next_path = [node, *path]
        if node.get("type") == "source":
            paths.append(next_path)
            return

        previous_nodes = incoming.get(node_id, [])
        if not previous_nodes:
            raise ValueError(
                f"Build branch ending at '{build_node_id}' is missing an upstream source path."
            )

        for previous_node_id in previous_nodes:
            if previous_node_id in seen:
                raise ValueError(f"Graph guide contains a cycle through node '{previous_node_id}'.")
            visit(previous_node_id, next_path, {*seen, previous_node_id})

    for previous_node_id in incoming.get(build_node_id, []):
        visit(previous_node_id, [node_map[build_node_id]], {build_node_id, previous_node_id})
    return paths


def _build_sources(
    node_map: dict[str, GraphNode],
    *,
    base_model_config: Any,
    custom_block_config: Any | None,
) -> dict[str, TargetSource]:
    sources: dict[str, TargetSource] = {}
    for node in node_map.values():
        if node.get("type") != "source":
            continue
        source = _build_target_source(
            node,
            base_model_config=base_model_config,
            custom_block_config=custom_block_config,
        )
        sources[source.name] = source
    return sources


def _build_target_source(
    node: GraphNode,
    *,
    base_model_config: Any,
    custom_block_config: Any | None,
) -> TargetSource:
    data = _node_data(node)
    component = data.get("component")
    if not isinstance(component, str) or not component:
        raise ValueError(f"Source node '{node['id']}' must define data.component.")

    source_kind = data.get("source_kind")
    if source_kind == "block":
        target_space: TargetSpace = "block"
        config = custom_block_config
    elif source_kind == "key":
        target_space = "key"
        config = base_model_config
    else:
        raise ValueError(
            f"Source node '{node['id']}' must define data.source_kind as 'block' or 'key'."
        )

    if config is None:
        raise ValueError(
            f"Source node '{node['id']}' requested '{source_kind}' targets but no config was loaded."
        )

    components = config.components()
    if component not in components:
        config_id = getattr(config, "identifier", "unknown")
        raise ValueError(
            f"Source node '{node['id']}' references missing component '{component}' in config '{config_id}'."
        )

    items = tuple(components[component].keys().keys())
    if not items:
        raise ValueError(f"Source node '{node['id']}' resolved no target items.")

    return TargetSource(
        name=str(node["id"]),
        target_space=target_space,
        items=items,
    )


def _resolve_branch_path(
    path: list[GraphNode],
    sources: dict[str, TargetSource],
) -> _ResolvedBranch:
    source_node = _single_node_of_type(path, "source")
    source = sources[str(source_node["id"])]
    component_name = _source_component_name(source_node)
    type_node = _single_node_of_type(path, "type")
    selection_nodes = [node for node in path if node.get("type") == "selection"]
    domain_nodes = [node for node in path if node.get("type") == "domain"]
    param_nodes = [node for node in path if node.get("type") == "param"]

    if len(selection_nodes) > 1:
        raise ValueError("Build branches may define at most one selection node.")
    if len(domain_nodes) > 1:
        raise ValueError("Build branches may define at most one domain node.")
    if len(param_nodes) > 1:
        raise ValueError("Build branches may define at most one param node.")

    selection_node = selection_nodes[0] if selection_nodes else None
    domain_node = domain_nodes[0] if domain_nodes else None
    param_node = param_nodes[0] if param_nodes else None
    type_data = _node_data(type_node)
    branch_type = type_data.get("mode")
    if branch_type not in {"all", "group", "exclude"}:
        raise ValueError(f"Type node '{type_node['id']}' has unsupported mode '{branch_type}'.")

    targets = _branch_targets(source, selection_node)
    if not targets:
        raise ValueError(f"Branch feeding build '{path[-1]['id']}' resolved no targets.")

    param_name = _param_name(param_node) if param_node is not None else None
    if branch_type != "exclude" and param_name is None:
        raise ValueError(
            f"Build branch using type '{branch_type}' must define a param node."
        )

    bounds = _domain_value(domain_node) if domain_node is not None else (0.0, 1.0)
    shared_group_name: str | None = None

    if branch_type == "group":
        if "groups" in type_data:
            raise ValueError(
                f"Type node '{type_node['id']}' cannot define data.groups. "
                "One graph 'group' node may only produce one grouped value."
            )
        shared_group_name = type_data.get("name")
        if not isinstance(shared_group_name, str) or not shared_group_name:
            shared_group_name = str(type_node["id"])

    return _ResolvedBranch(
        source=source,
        component_name=component_name,
        branch_type=branch_type,
        targets=targets,
        param_name=param_name,
        bounds=bounds,
        shared_group_name=shared_group_name,
        has_explicit_selection=selection_node is not None,
        path_index=_path_index(path),
    )


def _compile_build_branches(
    build_name: str,
    branches: list[_ResolvedBranch],
) -> list[BindingSpec]:
    excluded_targets = {
        (branch.source.name, target)
        for branch in branches
        if branch.branch_type == "exclude"
        for target in branch.targets
    }

    grouped_by_param: dict[str, list[_ResolvedBranch]] = {}
    for branch in branches:
        if branch.branch_type == "exclude":
            continue
        if branch.param_name is None:
            continue
        grouped_by_param.setdefault(branch.param_name, []).append(branch)

    bindings: list[BindingSpec] = []
    for param_name, param_branches in grouped_by_param.items():
        assigned_targets: set[tuple[str, str]] = set()
        for branch in sorted(param_branches, key=_branch_priority):
            available_targets = tuple(
                target
                for target in branch.targets
                if (branch.source.name, target) not in excluded_targets
                and (branch.source.name, target) not in assigned_targets
            )
            if not available_targets:
                continue

            if branch.branch_type == "all":
                bindings.append(
                    BindingSpec(
                        method_param_name=param_name,
                        component_name=branch.component_name,
                        strategy_label="graph_all",
                        selection=SelectionSpec(
                            source_name=branch.source.name,
                            include=available_targets,
                        ),
                        grouping="per_target",
                        bounds=branch.bounds,
                    )
                )
                assigned_targets.update((branch.source.name, target) for target in available_targets)
                continue

            bindings.append(
                BindingSpec(
                    method_param_name=param_name,
                    component_name=branch.component_name,
                    strategy_label="graph_group",
                    selection=SelectionSpec(
                        source_name=branch.source.name,
                        include=available_targets,
                    ),
                    grouping="shared",
                    bounds=branch.bounds,
                    group_name=branch.shared_group_name or f"{build_name}_{param_name}_group",
                )
            )
            assigned_targets.update((branch.source.name, target) for target in available_targets)

    return bindings


def _path_index(path: list[GraphNode]) -> int:
    return sum(index for index, _node in enumerate(path))


def _branch_priority(branch: _ResolvedBranch) -> tuple[int, int, int]:
    type_priority = 0 if branch.branch_type == "group" else 1
    selection_priority = 0 if branch.has_explicit_selection else 1
    return (type_priority, selection_priority, branch.path_index)


def _branch_targets(
    source: TargetSource,
    selection_node: GraphNode | None,
) -> tuple[str, ...]:
    if selection_node is None:
        return source.items

    data = _node_data(selection_node)
    mode = data.get("mode")
    if mode in {"block", "layer"}:
        selected = _string_tuple(data.get("items"), f"selection '{selection_node['id']}' items")
        if not selected:
            raise ValueError(
                f"Selection node '{selection_node['id']}' with mode '{mode}' must define items."
            )
        return tuple(target for target in source.items if target in selected)
    if mode == "regex":
        patterns = _string_tuple(data.get("patterns"), f"selection '{selection_node['id']}' patterns")
        if not patterns:
            raise ValueError(f"Selection node '{selection_node['id']}' regex mode needs patterns.")
        return _matching_targets(source.items, patterns, ())
    raise ValueError(f"Selection node '{selection_node['id']}' has unsupported mode '{mode}'.")


def _matching_targets(
    candidates: tuple[str, ...] | list[str],
    include: tuple[str, ...],
    exclude: tuple[str, ...],
) -> tuple[str, ...]:
    selected = []
    for candidate in candidates:
        if include and not any(_matches(candidate, pattern) for pattern in include):
            continue
        if exclude and any(_matches(candidate, pattern) for pattern in exclude):
            continue
        selected.append(candidate)
    return tuple(selected)


def _matches(value: str, pattern: str) -> bool:
    if "*" in pattern or "?" in pattern or "[" in pattern:
        import fnmatch

        return fnmatch.fnmatch(value, pattern)
    return value == pattern


def _single_node_of_type(path: list[GraphNode], node_type: str) -> GraphNode:
    matches = [node for node in path if node.get("type") == node_type]
    if len(matches) != 1:
        raise ValueError(f"Build branches must contain exactly one '{node_type}' node.")
    return matches[0]


def _domain_value(domain_node: GraphNode) -> BoundValue:
    data = _node_data(domain_node)
    mode = data.get("mode")
    if mode == "default":
        return (0.0, 1.0)
    if mode == "range":
        return (float(data["min"]), float(data["max"]))
    if mode == "categorical":
        values = data.get("values")
        if not isinstance(values, list):
            raise ValueError(f"Domain node '{domain_node['id']}' categorical mode needs data.values.")
        return values
    if mode == "fixed":
        return data.get("value")
    raise ValueError(f"Domain node '{domain_node['id']}' has unsupported mode '{mode}'.")


def _param_name(param_node: GraphNode) -> str:
    data = _node_data(param_node)
    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"Param node '{param_node['id']}' must define data.name.")
    return name


def _source_component_name(source_node: GraphNode) -> str:
    component = _node_data(source_node).get("component")
    if not isinstance(component, str) or not component:
        raise ValueError(f"Source node '{source_node['id']}' must define data.component.")
    return component


def _node_data(node: GraphNode) -> Mapping[str, Any]:
    data = node.get("data", {})
    if not isinstance(data, Mapping):
        raise ValueError(f"Graph guide node '{node.get('id')}' data must be a mapping.")
    return data


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if value in (None, ()):
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list.")
    if not all(isinstance(item, str) for item in value):
        raise ValueError(f"{label} must contain only strings.")
    return tuple(value)
