import logging

import sd_mecha

from typing import Any
from omegaconf import DictConfig, ListConfig, OmegaConf

from sd_optim.guide_graph import (
    BindingSpec,
    CompiledBinding,
    SelectionSpec,
    TargetSource,
    build_optimizer_bounds,
    compile_bindings,
    materialize_recipe_payloads,
)


logger = logging.getLogger(__name__)
BoundsInfo = dict[str, dict[str, Any]]


def compile_legacy_guide_to_bindings(
    cfg: DictConfig,
    base_model_config: sd_mecha.extensions.model_configs.ModelConfig,
    custom_block_config: sd_mecha.extensions.model_configs.ModelConfig | None,
) -> list[CompiledBinding]:
    """Compile the current guide format into graph-backed bindings with legacy semantics."""
    guide_components = cfg.optimization_guide.get("components", [])
    if not guide_components or not isinstance(guide_components, (list, ListConfig)):
        return []

    compiled_bindings: list[CompiledBinding] = []
    assigned_targets: dict[tuple[str, str], str] = {}

    for component_index, component_config_raw in enumerate(guide_components):
        if not isinstance(component_config_raw, (dict, DictConfig)):
            logger.warning("Skipping component entry at index %s: Not a dictionary.", component_index)
            continue

        component_config = _resolve_config_mapping(component_config_raw)
        component_name = component_config.get("name")
        if not component_name:
            logger.warning(
                "Skipping component entry at index %s due to missing 'name'.",
                component_index,
            )
            continue

        component_params = component_config.get("optimize_params", [])
        if not isinstance(component_params, list):
            component_params = []

        strategies = component_config.get("strategies")
        if not strategies or not isinstance(strategies, list):
            logger.warning(
                "Component '%s' is missing a valid 'strategies' list. Skipping this component.",
                component_name,
            )
            continue

        for strategy_index, strategy_raw in enumerate(strategies):
            if not isinstance(strategy_raw, (dict, DictConfig)):
                logger.warning(
                    "Invalid strategy entry format at index %s in '%s'. Skipping.",
                    strategy_index,
                    component_name,
                )
                continue

            strategy = _resolve_config_mapping(strategy_raw)
            strategy_type = strategy.get("type")
            if strategy_type not in {"all", "select", "group", "single", "none"}:
                logger.warning(
                    "Missing or invalid strategy 'type' at index %s in '%s'. Skipping.",
                    strategy_index,
                    component_name,
                )
                continue
            if strategy_type == "none":
                continue

            source = _determine_target_source(
                component_name,
                strategy,
                base_model_config,
                custom_block_config,
            )
            if source is None:
                continue

            current_params = strategy.get("optimize_params", component_params)
            if not isinstance(current_params, list):
                current_params = component_params
            if not current_params:
                continue

            if strategy_type == "group":
                groups = strategy.get("groups", [])
                if not groups or not isinstance(groups, list):
                    logger.warning(
                        "'group' strategy needs a valid 'groups' list in '%s'. Skipping for all params.",
                        component_name,
                    )
                    continue

                for method_param_name in current_params:
                    for group_index, group_raw in enumerate(groups):
                        if not isinstance(group_raw, (dict, DictConfig)):
                            logger.warning(
                                "Invalid group format at index %s in '%s'. Skipping.",
                                group_index,
                                component_name,
                            )
                            continue

                        group = _resolve_config_mapping(group_raw)
                        group_name = group.get("name")
                        group_patterns = group.get("keys", [])
                        if not group_name or not isinstance(group_patterns, list):
                            logger.warning(
                                "Invalid group format (missing name or keys list) for group at index %s in '%s'. Skipping group.",
                                group_index,
                                component_name,
                            )
                            continue

                        binding = BindingSpec(
                            method_param_name=method_param_name,
                            component_name=component_name,
                            strategy_label="group",
                            selection=SelectionSpec(
                                source_name=source.name,
                                include=tuple(group_patterns),
                            ),
                            grouping="shared",
                            group_name=group_name,
                        )
                        compiled_bindings.extend(
                            _compile_binding_with_legacy_conflict_handling(
                                source=source,
                                binding=binding,
                                assigned_targets=assigned_targets,
                                assignment_label=f"group:{group_name}",
                                whole_binding_conflict=True,
                            )
                        )
                continue

            for method_param_name in current_params:
                binding = _build_legacy_binding(
                    component_name=component_name,
                    strategy_type=strategy_type,
                    strategy=strategy,
                    source=source,
                    method_param_name=method_param_name,
                )
                if binding is None:
                    continue

                whole_binding_conflict = strategy_type in {"single"}
                compiled_bindings.extend(
                    _compile_binding_with_legacy_conflict_handling(
                        source=source,
                        binding=binding,
                        assigned_targets=assigned_targets,
                        assignment_label=_assignment_label(strategy_type, binding),
                        whole_binding_conflict=whole_binding_conflict,
                    )
                )

    return compiled_bindings


def compile_legacy_guide_to_bounds(
    cfg: DictConfig,
    base_model_config: sd_mecha.extensions.model_configs.ModelConfig,
    custom_block_config: sd_mecha.extensions.model_configs.ModelConfig | None,
) -> dict[str, Any]:
    """Return optimizer bounds for the legacy guide through the graph-backed path."""
    return build_optimizer_bounds(
        compile_legacy_guide_to_bindings(
            cfg,
            base_model_config,
            custom_block_config,
        )
    )


def materialize_legacy_guide_payloads(
    cfg: DictConfig,
    base_model_config: sd_mecha.extensions.model_configs.ModelConfig,
    custom_block_config: sd_mecha.extensions.model_configs.ModelConfig | None,
    sampled_values: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], set[str]]:
    """Return block/key payload maps for the legacy guide through the graph-backed path."""
    compiled_bindings = compile_legacy_guide_to_bindings(
        cfg,
        base_model_config,
        custom_block_config,
    )
    active_bindings: list[CompiledBinding] = []
    for binding in compiled_bindings:
        if binding.optimizer_param_name not in sampled_values:
            logger.warning(
                "Optimizer did not provide value for parameter '%s'. Skipping.",
                binding.optimizer_param_name,
            )
            continue
        active_bindings.append(binding)

    method_payloads = materialize_recipe_payloads(sampled_values, active_bindings)

    block_payloads: dict[str, dict[str, Any]] = {}
    key_payloads: dict[str, dict[str, Any]] = {}
    handled_method_params = {binding.method_param_name for binding in active_bindings}

    for binding in active_bindings:
        payload = method_payloads.get(binding.method_param_name, {})
        scoped_payload = {target: payload[target] for target in binding.targets if target in payload}
        if binding.target_space == "block":
            block_payloads.setdefault(binding.method_param_name, {}).update(scoped_payload)
        elif binding.target_space == "key":
            key_payloads.setdefault(binding.method_param_name, {}).update(scoped_payload)

    return block_payloads, key_payloads, handled_method_params


def materialize_payloads_from_legacy_bounds_info(
    sampled_values: dict[str, Any],
    param_info: BoundsInfo,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], set[str]]:
    """Materialize block/key payloads from already-compiled legacy bounds metadata."""
    block_payloads: dict[str, dict[str, Any]] = {}
    key_payloads: dict[str, dict[str, Any]] = {}
    handled_method_params: set[str] = set()

    for optimizer_param_name, info in param_info.items():
        if optimizer_param_name not in sampled_values:
            continue

        method_param_name = info.get("base_param")
        target_space = info.get("target_type")
        if not isinstance(method_param_name, str) or target_space not in {"block", "key"}:
            logger.warning(
                "Skipping malformed legacy bounds entry for '%s' during payload materialization.",
                optimizer_param_name,
            )
            continue

        handled_method_params.add(method_param_name)
        payload_map = (
            block_payloads.setdefault(method_param_name, {})
            if target_space == "block"
            else key_payloads.setdefault(method_param_name, {})
        )
        value = sampled_values[optimizer_param_name]

        item_name = info.get("item_name")
        if isinstance(item_name, str):
            payload_map[item_name] = value
            continue

        items_covered = info.get("items_covered")
        if isinstance(items_covered, list):
            for item_name in items_covered:
                if isinstance(item_name, str):
                    payload_map[item_name] = value
            continue

        logger.warning(
            "Skipping payload materialization for '%s' because no target items were recorded.",
            optimizer_param_name,
        )

    return block_payloads, key_payloads, handled_method_params


def compile_legacy_guide_to_bounds_info(
    cfg: DictConfig,
    base_model_config: sd_mecha.extensions.model_configs.ModelConfig,
    custom_block_config: sd_mecha.extensions.model_configs.ModelConfig | None,
) -> BoundsInfo:
    """Return legacy-style bounds metadata through the graph-backed path."""
    compiled_bindings = compile_legacy_guide_to_bindings(
        cfg,
        base_model_config,
        custom_block_config,
    )
    bounds_info: BoundsInfo = {}

    for binding in compiled_bindings:
        entry: dict[str, Any] = {
            "strategy": binding.strategy_label,
            "target_type": binding.target_space,
            "component_name": binding.component_name,
            "base_param": binding.method_param_name,
            "bounds": binding.bounds,
        }
        if binding.grouping == "per_target":
            entry["item_name"] = binding.targets[0]
        else:
            if binding.group_name is not None:
                entry["group_name"] = binding.group_name
            entry["items_covered"] = list(binding.targets)
        bounds_info[binding.optimizer_param_name] = entry

    return bounds_info


def _build_legacy_binding(
    *,
    component_name: str,
    strategy_type: str,
    strategy: dict[str, Any],
    source: TargetSource,
    method_param_name: str,
) -> BindingSpec | None:
    if strategy_type == "all":
        return BindingSpec(
            method_param_name=method_param_name,
            component_name=component_name,
            strategy_label="all",
            selection=SelectionSpec(source_name=source.name),
            grouping="per_target",
        )

    if strategy_type == "select":
        patterns = strategy.get("keys", [])
        if not patterns or not isinstance(patterns, list):
            logger.warning(
                "'select' strategy needs a valid 'keys' list in '%s'. Skipping for param '%s'.",
                component_name,
                method_param_name,
            )
            return None
        return BindingSpec(
            method_param_name=method_param_name,
            component_name=component_name,
            strategy_label="select",
            selection=SelectionSpec(
                source_name=source.name,
                include=tuple(patterns),
            ),
            grouping="per_target",
        )

    if strategy_type == "single":
        return BindingSpec(
            method_param_name=method_param_name,
            component_name=component_name,
            strategy_label="single",
            selection=SelectionSpec(source_name=source.name),
            grouping="shared",
            group_name=f"{component_name}_single",
        )

    raise ValueError(f"Unsupported legacy strategy type '{strategy_type}'.")


def _compile_binding_with_legacy_conflict_handling(
    *,
    source: TargetSource,
    binding: BindingSpec,
    assigned_targets: dict[tuple[str, str], str],
    assignment_label: str,
    whole_binding_conflict: bool,
) -> list[CompiledBinding]:
    try:
        candidates = compile_bindings([source], [binding])
    except ValueError as error:
        if _is_legacy_empty_selection_error(error):
            _log_legacy_empty_selection(binding)
            return []
        raise
    accepted: list[CompiledBinding] = []

    for candidate in candidates:
        conflict_targets = [
            target
            for target in candidate.targets
            if (candidate.method_param_name, target) in assigned_targets
        ]
        if conflict_targets:
            if whole_binding_conflict:
                logger.error(
                    "Conflict for targets %s (param '%s')! Assigned by existing strategy, cannot assign by '%s'. Skipping binding.",
                    conflict_targets,
                    candidate.method_param_name,
                    assignment_label,
                )
                continue

            if candidate.grouping == "per_target":
                logger.error(
                    "Conflict for item '%s' (param '%s')! Assigned by existing strategy, cannot assign by '%s'. Skipping.",
                    conflict_targets[0],
                    candidate.method_param_name,
                    assignment_label,
                )
                continue

        for target in candidate.targets:
            assigned_targets[(candidate.method_param_name, target)] = assignment_label
        accepted.append(candidate)

    return accepted


def _is_legacy_empty_selection_error(error: ValueError) -> bool:
    return "resolved no targets" in str(error)


def _log_legacy_empty_selection(binding: BindingSpec) -> None:
    if binding.strategy_label == "group":
        logger.warning(
            "Group '%s' for param '%s' did not match any items in component '%s'. Group will be skipped.",
            binding.group_name,
            binding.method_param_name,
            binding.component_name,
        )
        return

    if binding.strategy_label == "select":
        logger.warning(
            "'select' strategy pattern(s) %s for param '%s' did not match any items in component '%s'. Skipping.",
            list(binding.selection.include),
            binding.method_param_name,
            binding.component_name,
        )
        return

    logger.warning(
        "Selection for strategy '%s' and param '%s' did not match any items in component '%s'. Skipping.",
        binding.strategy_label,
        binding.method_param_name,
        binding.component_name,
    )


def _assignment_label(strategy_type: str, binding: BindingSpec) -> str:
    if strategy_type == "single":
        return f"single:{binding.group_name}"
    return strategy_type


def _determine_target_source(
    component_name: str,
    strategy_config: dict[str, Any],
    base_model_config: sd_mecha.extensions.model_configs.ModelConfig,
    custom_block_config: sd_mecha.extensions.model_configs.ModelConfig | None,
) -> TargetSource | None:
    strategy_target_type = strategy_config.get("target_type")

    if strategy_target_type == "block":
        config_to_iterate = custom_block_config
        target_space = "block"
    elif strategy_target_type == "key":
        config_to_iterate = base_model_config
        target_space = "key"
    elif custom_block_config and component_name in custom_block_config.components():
        config_to_iterate = custom_block_config
        target_space = "block"
    elif base_model_config and component_name in base_model_config.components():
        config_to_iterate = base_model_config
        target_space = "key"
    else:
        logger.warning("Component '%s' not found in known configs. Skipping component.", component_name)
        return None

    if not config_to_iterate or component_name not in config_to_iterate.components():
        logger.warning(
            "Strategy specifies target_type='%s' but component '%s' not found in target config. Skipping strategy.",
            strategy_target_type,
            component_name,
        )
        return None

    items = tuple(config_to_iterate.components()[component_name].keys().keys())
    if not items:
        logger.warning(
            "Component '%s' has no items in config '%s'. Skipping.",
            component_name,
            getattr(config_to_iterate, "identifier", "unknown"),
        )
        return None

    return TargetSource(
        name=f"{component_name}:{target_space}:{getattr(config_to_iterate, 'identifier', 'unknown')}",
        target_space=target_space,
        items=items,
    )


def _resolve_config_mapping(config_raw: dict[str, Any] | DictConfig) -> dict[str, Any]:
    if isinstance(config_raw, DictConfig):
        return OmegaConf.to_container(config_raw, resolve=True)
    return config_raw
