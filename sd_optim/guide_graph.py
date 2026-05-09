import fnmatch
import logging

from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger(__name__)

BoundValue = tuple[float, float] | float | int | bool | list[Any] | dict[str, Any]
GroupingMode = Literal["per_target", "shared", "named_groups"]
TargetSpace = Literal["block", "key"]


@dataclass(frozen=True)
class TargetSource:
    """Ordered target universe for one source/component space."""

    name: str
    target_space: TargetSpace
    items: tuple[str, ...]


@dataclass(frozen=True)
class SelectionSpec:
    """Describe which source items are presently selected."""

    source_name: str
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()


@dataclass(frozen=True)
class NamedGroupSpec:
    """Define one explicit group inside a selected target set."""

    name: str
    include: tuple[str, ...]
    exclude: tuple[str, ...] = ()


@dataclass(frozen=True)
class BindingSpec:
    """Bind one selected or grouped target set to a method parameter."""

    method_param_name: str
    component_name: str
    strategy_label: str
    selection: SelectionSpec
    grouping: GroupingMode
    bounds: BoundValue = (0.0, 1.0)
    group_name: str | None = None
    named_groups: tuple[NamedGroupSpec, ...] = ()


@dataclass(frozen=True)
class CompiledBinding:
    """One optimizer-visible parameter after graph compilation."""

    optimizer_param_name: str
    method_param_name: str
    component_name: str
    strategy_label: str
    target_space: TargetSpace
    source_name: str
    targets: tuple[str, ...]
    bounds: BoundValue
    grouping: GroupingMode
    group_name: str | None = None


def compile_bindings(
    sources: list[TargetSource] | tuple[TargetSource, ...],
    bindings: list[BindingSpec] | tuple[BindingSpec, ...],
) -> list[CompiledBinding]:
    """Compile graph-style binding specs into optimizer-visible bindings."""
    source_map = {source.name: source for source in sources}
    compiled: list[CompiledBinding] = []

    for binding in bindings:
        source = source_map.get(binding.selection.source_name)
        if source is None:
            raise ValueError(f"Unknown target source '{binding.selection.source_name}'.")

        selected_targets = _resolve_selection(source, binding.selection)
        if not selected_targets:
            raise ValueError(
                f"Selection for method param '{binding.method_param_name}' resolved no targets "
                f"in source '{source.name}'."
            )

        if binding.grouping == "per_target":
            for target in selected_targets:
                compiled.append(
                    CompiledBinding(
                        optimizer_param_name=f"{target}_{binding.method_param_name}",
                        method_param_name=binding.method_param_name,
                        component_name=binding.component_name,
                        strategy_label=binding.strategy_label,
                        target_space=source.target_space,
                        source_name=source.name,
                        targets=(target,),
                        bounds=binding.bounds,
                        grouping=binding.grouping,
                    )
                )
            continue

        if binding.grouping == "shared":
            shared_name = binding.group_name or f"{source.name}_shared"
            compiled.append(
                CompiledBinding(
                    optimizer_param_name=f"{shared_name}_{binding.method_param_name}",
                    method_param_name=binding.method_param_name,
                    component_name=binding.component_name,
                    strategy_label=binding.strategy_label,
                    target_space=source.target_space,
                    source_name=source.name,
                    targets=tuple(selected_targets),
                    bounds=binding.bounds,
                    grouping=binding.grouping,
                    group_name=shared_name,
                )
            )
            continue

        if binding.grouping == "named_groups":
            if not binding.named_groups:
                raise ValueError(
                    f"Binding '{binding.method_param_name}' uses named_groups but defines none."
                )

            assigned_targets: set[str] = set()
            for group in binding.named_groups:
                group_targets = _filter_targets(
                    selected_targets,
                    include=group.include,
                    exclude=group.exclude,
                )
                if not group_targets:
                    raise ValueError(
                        f"Named group '{group.name}' for method param '{binding.method_param_name}' "
                        "resolved no targets."
                    )

                overlap = assigned_targets.intersection(group_targets)
                if overlap:
                    raise ValueError(
                        f"Named group '{group.name}' for method param '{binding.method_param_name}' "
                        f"overlaps already-assigned targets: {sorted(overlap)}"
                    )

                assigned_targets.update(group_targets)
                compiled.append(
                    CompiledBinding(
                        optimizer_param_name=f"{group.name}_{binding.method_param_name}",
                        method_param_name=binding.method_param_name,
                        component_name=binding.component_name,
                        strategy_label=binding.strategy_label,
                        target_space=source.target_space,
                        source_name=source.name,
                        targets=tuple(group_targets),
                        bounds=binding.bounds,
                        grouping=binding.grouping,
                        group_name=group.name,
                    )
                )
            continue

        raise ValueError(f"Unsupported grouping mode '{binding.grouping}'.")

    logger.info("Compiled %s graph binding(s) from %s target source(s) into %s optimizer-visible parameter(s).", len(bindings), len(sources), len(compiled))
    return compiled


def build_optimizer_bounds(
    compiled_bindings: list[CompiledBinding] | tuple[CompiledBinding, ...],
) -> dict[str, BoundValue]:
    """Extract optimizer bounds from compiled bindings."""
    return {
        binding.optimizer_param_name: binding.bounds
        for binding in compiled_bindings
    }


def materialize_recipe_payloads(
    sampled_values: dict[str, Any],
    compiled_bindings: list[CompiledBinding] | tuple[CompiledBinding, ...],
) -> dict[str, dict[str, Any]]:
    """Turn sampled optimizer values into method-param -> target payload dicts."""
    payloads: dict[str, dict[str, Any]] = {}

    for binding in compiled_bindings:
        if binding.optimizer_param_name not in sampled_values:
            raise KeyError(
                f"Missing sampled value for optimizer parameter '{binding.optimizer_param_name}'."
            )

        value = sampled_values[binding.optimizer_param_name]
        merge_payload = payloads.setdefault(binding.method_param_name, {})

        for target in binding.targets:
            if target in merge_payload:
                raise ValueError(
                    f"Target '{target}' was assigned multiple values for method parameter "
                    f"'{binding.method_param_name}'."
                )
            merge_payload[target] = value

    logger.info("Materialized %s merge payload dict(s) from %s sampled values.", len(payloads), len(sampled_values))
    return payloads


def _resolve_selection(source: TargetSource, selection: SelectionSpec) -> list[str]:
    return _filter_targets(
        source.items,
        include=selection.include,
        exclude=selection.exclude,
    )


def _filter_targets(
    candidates: tuple[str, ...] | list[str],
    *,
    include: tuple[str, ...] = (),
    exclude: tuple[str, ...] = (),
) -> list[str]:
    ordered_candidates = list(candidates)
    selected = (
        ordered_candidates
        if not include
        else [candidate for candidate in ordered_candidates if _matches_any(candidate, include)]
    )
    if exclude:
        selected = [candidate for candidate in selected if not _matches_any(candidate, exclude)]
    return selected


def _matches_any(value: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(value, pattern) for pattern in patterns)
