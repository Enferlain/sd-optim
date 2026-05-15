from collections.abc import Iterable

from sd_optim.guide_compiler import CompiledBinding
from sd_optim.guide_runtime import (
    GraphRuntimeBundle,
    GraphRuntimeSummary,
    build_graph_runtime_bundle,
    materialize_target_space_payloads,
    validate_graph_dependencies,
)


class _FakeComponent:
    def __init__(self, item_names: Iterable[str]) -> None:
        self._items = tuple(item_names)

    def keys(self) -> dict[str, None]:
        return dict.fromkeys(self._items)


class _FakeModelConfig:
    def __init__(self, identifier: str, components: dict[str, Iterable[str]]) -> None:
        self.identifier = identifier
        self._components = {
            component_name: _FakeComponent(item_names)
            for component_name, item_names in components.items()
        }

    def components(self) -> dict[str, _FakeComponent]:
        return self._components


def test_build_graph_runtime_bundle_summarizes_compiled_graph_output() -> None:
    graph = {
        "version": 1,
        "nodes": [
            {
                "id": "unet_blocks",
                "type": "source",
                "data": {
                    "source_kind": "block",
                    "component": "unet",
                },
            },
            {
                "id": "selected_blocks",
                "type": "selection",
                "data": {
                    "mode": "block",
                    "items": ["UNET_IN05_1", "UNET_OUT05_1"],
                },
            },
            {
                "id": "decoder_surface",
                "type": "type",
                "data": {
                    "mode": "group",
                    "name": "decoder_surface",
                },
            },
            {
                "id": "alpha_domain",
                "type": "domain",
                "data": {
                    "mode": "range",
                    "min": 0.25,
                    "max": 0.75,
                },
            },
            {
                "id": "unet_keys",
                "type": "source",
                "data": {
                    "source_kind": "key",
                    "component": "unet",
                },
            },
            {
                "id": "key_targets",
                "type": "selection",
                "data": {
                    "mode": "regex",
                    "patterns": ["model.diffusion_model.out.*"],
                },
            },
            {
                "id": "all_keys",
                "type": "type",
                "data": {"mode": "all"},
            },
            {
                "id": "alpha",
                "type": "param",
                "data": {"name": "alpha"},
            },
            {
                "id": "build_alpha",
                "type": "build",
                "data": {},
            },
        ],
        "edges": [
            {"from": "unet_blocks", "to": "selected_blocks"},
            {"from": "selected_blocks", "to": "decoder_surface"},
            {"from": "decoder_surface", "to": "alpha_domain"},
            {"from": "alpha_domain", "to": "alpha"},
            {"from": "alpha", "to": "build_alpha"},
            {"from": "unet_keys", "to": "key_targets"},
            {"from": "key_targets", "to": "all_keys"},
            {"from": "all_keys", "to": "alpha"},
        ],
    }

    bundle = build_graph_runtime_bundle(
        graph,
        base_model_config=_FakeModelConfig(
            "sdxl-sgm",
            {
                "unet": [
                    "model.diffusion_model.out.0.weight",
                    "model.diffusion_model.out.2.weight",
                    "model.diffusion_model.time_embed.weight",
                ],
            },
        ),
        custom_block_config=_FakeModelConfig(
            "sdxl-optim_blocks_sub",
            {
                "unet": [
                    "UNET_IN05_1",
                    "UNET_OUT05_1",
                    "UNET_MID00",
                ],
            },
        ),
    )

    assert [binding.optimizer_param_name for binding in bundle.compiled_bindings] == [
        "decoder_surface_alpha",
        "model.diffusion_model.out.0.weight_alpha",
        "model.diffusion_model.out.2.weight_alpha",
    ]
    assert bundle.optimizer_bounds == {
        "decoder_surface_alpha": (0.25, 0.75),
        "model.diffusion_model.out.0.weight_alpha": (0.0, 1.0),
        "model.diffusion_model.out.2.weight_alpha": (0.0, 1.0),
    }
    assert bundle.summary.source_count == 2
    assert bundle.summary.build_count == 1
    assert bundle.summary.binding_count == 2
    assert bundle.summary.compiled_parameter_count == 3
    assert bundle.summary.target_space_counts == {
        "block": 1,
        "key": 2,
    }
    assert bundle.summary.grouping_counts == {
        "shared": 1,
        "per_target": 2,
    }
    assert bundle.summary.bounds_shape_counts == {
        "fixed": 0,
        "categorical": 0,
        "continuous": 3,
        "default_bounds_used": 2,
    }


def test_materialize_target_space_payloads_splits_block_and_key_targets() -> None:
    graph = {
        "version": 1,
        "nodes": [
            {
                "id": "unet_blocks",
                "type": "source",
                "data": {
                    "source_kind": "block",
                    "component": "unet",
                },
            },
            {
                "id": "selected_blocks",
                "type": "selection",
                "data": {
                    "mode": "block",
                    "items": ["UNET_IN05_1", "UNET_OUT05_1"],
                },
            },
            {
                "id": "decoder_surface",
                "type": "type",
                "data": {
                    "mode": "group",
                    "name": "decoder_surface",
                },
            },
            {
                "id": "unet_keys",
                "type": "source",
                "data": {
                    "source_kind": "key",
                    "component": "unet",
                },
            },
            {
                "id": "key_targets",
                "type": "selection",
                "data": {
                    "mode": "regex",
                    "patterns": ["model.diffusion_model.out.*"],
                },
            },
            {
                "id": "all_keys",
                "type": "type",
                "data": {"mode": "all"},
            },
            {
                "id": "alpha",
                "type": "param",
                "data": {"name": "alpha"},
            },
            {
                "id": "build_alpha",
                "type": "build",
                "data": {},
            },
        ],
        "edges": [
            {"from": "unet_blocks", "to": "selected_blocks"},
            {"from": "selected_blocks", "to": "decoder_surface"},
            {"from": "decoder_surface", "to": "alpha"},
            {"from": "alpha", "to": "build_alpha"},
            {"from": "unet_keys", "to": "key_targets"},
            {"from": "key_targets", "to": "all_keys"},
            {"from": "all_keys", "to": "alpha"},
        ],
    }

    bundle = build_graph_runtime_bundle(
        graph,
        base_model_config=_FakeModelConfig(
            "sdxl-sgm",
            {
                "unet": [
                    "model.diffusion_model.out.0.weight",
                    "model.diffusion_model.out.2.weight",
                    "model.diffusion_model.time_embed.weight",
                ],
            },
        ),
        custom_block_config=_FakeModelConfig(
            "sdxl-optim_blocks_sub",
            {
                "unet": [
                    "UNET_IN05_1",
                    "UNET_OUT05_1",
                    "UNET_MID00",
                ],
            },
        ),
    )

    block_payloads, key_payloads, handled_method_params = materialize_target_space_payloads(
        {
            "decoder_surface_alpha": 0.6,
            "model.diffusion_model.out.0.weight_alpha": 0.2,
            "model.diffusion_model.out.2.weight_alpha": 0.4,
        },
        bundle.compiled_bindings,
    )

    assert block_payloads == {
        "alpha": {
            "UNET_IN05_1": 0.6,
            "UNET_OUT05_1": 0.6,
        }
    }
    assert key_payloads == {
        "alpha": {
            "model.diffusion_model.out.0.weight": 0.2,
            "model.diffusion_model.out.2.weight": 0.4,
        }
    }
    assert handled_method_params == {"alpha"}


def test_validate_graph_dependencies_maps_matching_binding_scopes() -> None:
    graph = {
        "version": 1,
        "nodes": [
            {
                "id": "unet_keys",
                "type": "source",
                "data": {
                    "source_kind": "key",
                    "component": "unet",
                },
            },
            {
                "id": "key_targets",
                "type": "selection",
                "data": {
                    "mode": "regex",
                    "patterns": ["model.diffusion_model.out.*"],
                },
            },
            {
                "id": "all_keys",
                "type": "type",
                "data": {"mode": "all"},
            },
            {
                "id": "alpha",
                "type": "param",
                "data": {"name": "alpha"},
            },
            {
                "id": "beta",
                "type": "param",
                "data": {"name": "beta"},
            },
            {
                "id": "build_alpha",
                "type": "build",
                "data": {},
            },
            {
                "id": "build_beta",
                "type": "build",
                "data": {},
            },
        ],
        "edges": [
            {"from": "unet_keys", "to": "key_targets"},
            {"from": "key_targets", "to": "all_keys"},
            {"from": "all_keys", "to": "alpha"},
            {"from": "alpha", "to": "build_alpha"},
            {"from": "all_keys", "to": "beta"},
            {"from": "beta", "to": "build_beta"},
        ],
    }

    bundle = build_graph_runtime_bundle(
        graph,
        base_model_config=_FakeModelConfig(
            "sdxl-sgm",
            {
                "unet": [
                    "model.diffusion_model.out.0.weight",
                    "model.diffusion_model.out.2.weight",
                    "model.diffusion_model.time_embed.weight",
                ],
            },
        ),
        custom_block_config=None,
    )

    dependencies = validate_graph_dependencies(
        bundle,
        [{"parent": "alpha", "child": "beta", "condition": "> 0", "default": 0.25}],
    )

    assert dependencies == {
        "model.diffusion_model.out.0.weight_beta": {
            "parent": "model.diffusion_model.out.0.weight_alpha",
            "condition": "> 0",
            "default": 0.25,
        },
        "model.diffusion_model.out.2.weight_beta": {
            "parent": "model.diffusion_model.out.2.weight_alpha",
            "condition": "> 0",
            "default": 0.25,
        },
    }


def test_validate_graph_dependencies_ignores_unmatched_scopes() -> None:
    bundle = GraphRuntimeBundle(
        compiled_bindings=(
            CompiledBinding(
                optimizer_param_name="first_alpha",
                method_param_name="alpha",
                component_name="unet",
                strategy_label="graph_all",
                target_space="key",
                source_name="first_source",
                targets=("model.diffusion_model.out.0.weight",),
                bounds=(0.0, 1.0),
                grouping="per_target",
            ),
            CompiledBinding(
                optimizer_param_name="second_beta",
                method_param_name="beta",
                component_name="unet",
                strategy_label="graph_all",
                target_space="key",
                source_name="second_source",
                targets=("model.diffusion_model.out.0.weight",),
                bounds=(0.0, 1.0),
                grouping="per_target",
            ),
        ),
        optimizer_bounds={
            "first_alpha": (0.0, 1.0),
            "second_beta": (0.0, 1.0),
        },
        summary=GraphRuntimeSummary(
            source_count=2,
            build_count=2,
            binding_count=2,
            compiled_parameter_count=2,
            target_space_counts={"key": 2},
            grouping_counts={"per_target": 2},
            bounds_shape_counts={
                "fixed": 0,
                "categorical": 0,
                "continuous": 2,
                "default_bounds_used": 2,
            },
        ),
    )

    assert validate_graph_dependencies(bundle, [{"parent": "alpha", "child": "beta"}]) == {}
