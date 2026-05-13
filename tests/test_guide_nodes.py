from collections.abc import Iterable

from sd_optim.guide_compiler import build_optimizer_bounds, materialize_recipe_payloads
from sd_optim.guide_nodes import compile_graph_guide_to_bindings


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


def test_build_all_branch_uses_default_selection_and_default_domain() -> None:
    graph = {
        "version": 1,
        "nodes": [
            {
                "id": "unet",
                "type": "source",
                "data": {
                    "source_kind": "block",
                    "component": "unet",
                },
            },
            {
                "id": "all_targets",
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
            {"from": "unet", "to": "all_targets"},
            {"from": "all_targets", "to": "alpha"},
            {"from": "alpha", "to": "build_alpha"},
        ],
    }

    compiled = compile_graph_guide_to_bindings(
        graph,
        base_model_config=_FakeModelConfig("sdxl-sgm", {"unet": ["TEXT_A"]}),
        custom_block_config=_FakeModelConfig(
            "sdxl-optim_blocks_sub",
            {"unet": ["UNET_IN05_1", "UNET_OUT05_1"]},
        ),
    )

    assert [binding.optimizer_param_name for binding in compiled] == [
        "UNET_IN05_1_alpha",
        "UNET_OUT05_1_alpha",
    ]
    assert build_optimizer_bounds(compiled) == {
        "UNET_IN05_1_alpha": (0.0, 1.0),
        "UNET_OUT05_1_alpha": (0.0, 1.0),
    }


def test_build_group_branch_overrides_subset_of_broad_all_branch() -> None:
    graph = {
        "version": 1,
        "nodes": [
            {
                "id": "unet",
                "type": "source",
                "data": {
                    "source_kind": "block",
                    "component": "unet",
                },
            },
            {
                "id": "all_targets",
                "type": "type",
                "data": {"mode": "all"},
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
                "id": "alpha_bound",
                "type": "domain",
                "data": {
                    "mode": "range",
                    "min": 0.25,
                    "max": 0.75,
                },
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
            {"from": "unet", "to": "all_targets"},
            {"from": "all_targets", "to": "alpha"},
            {"from": "alpha", "to": "build_alpha"},
            {"from": "unet", "to": "selected_blocks"},
            {"from": "selected_blocks", "to": "decoder_surface"},
            {"from": "decoder_surface", "to": "alpha_bound"},
            {"from": "alpha_bound", "to": "alpha"},
        ],
    }

    compiled = compile_graph_guide_to_bindings(
        graph,
        base_model_config=_FakeModelConfig("sdxl-sgm", {"unet": ["TEXT_A"]}),
        custom_block_config=_FakeModelConfig(
            "sdxl-optim_blocks_sub",
            {"unet": ["UNET_IN05_1", "UNET_OUT05_1", "UNET_MID00"]},
        ),
    )

    assert [binding.optimizer_param_name for binding in compiled] == [
        "decoder_surface_alpha",
        "UNET_MID00_alpha",
    ]
    assert compiled[0].targets == ("UNET_IN05_1", "UNET_OUT05_1")
    assert compiled[1].targets == ("UNET_MID00",)
    assert build_optimizer_bounds(compiled) == {
        "decoder_surface_alpha": (0.25, 0.75),
        "UNET_MID00_alpha": (0.0, 1.0),
    }


def test_build_exclude_branch_removes_targets_from_other_branches() -> None:
    graph = {
        "version": 1,
        "nodes": [
            {
                "id": "unet",
                "type": "source",
                "data": {
                    "source_kind": "block",
                    "component": "unet",
                },
            },
            {
                "id": "all_targets",
                "type": "type",
                "data": {"mode": "all"},
            },
            {
                "id": "block_69",
                "type": "selection",
                "data": {
                    "mode": "block",
                    "items": ["UNET_OUT05_1"],
                },
            },
            {
                "id": "exclude_target",
                "type": "type",
                "data": {"mode": "exclude"},
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
            {"from": "unet", "to": "all_targets"},
            {"from": "all_targets", "to": "alpha"},
            {"from": "alpha", "to": "build_alpha"},
            {"from": "unet", "to": "block_69"},
            {"from": "block_69", "to": "exclude_target"},
            {"from": "exclude_target", "to": "build_alpha"},
        ],
    }

    compiled = compile_graph_guide_to_bindings(
        graph,
        base_model_config=_FakeModelConfig("sdxl-sgm", {"unet": ["TEXT_A"]}),
        custom_block_config=_FakeModelConfig(
            "sdxl-optim_blocks_sub",
            {"unet": ["UNET_IN05_1", "UNET_OUT05_1", "UNET_MID00"]},
        ),
    )

    assert [binding.optimizer_param_name for binding in compiled] == [
        "UNET_IN05_1_alpha",
        "UNET_MID00_alpha",
    ]
    assert materialize_recipe_payloads(
        {
            "UNET_IN05_1_alpha": 0.1,
            "UNET_MID00_alpha": 0.9,
        },
        compiled,
    ) == {
        "alpha": {
            "UNET_IN05_1": 0.1,
            "UNET_MID00": 0.9,
        }
    }


def test_build_combines_all_group_and_exclude_branches_like_sketch() -> None:
    graph = {
        "version": 1,
        "nodes": [
            {
                "id": "unet",
                "type": "source",
                "data": {
                    "source_kind": "block",
                    "component": "unet",
                },
            },
            {
                "id": "all_targets",
                "type": "type",
                "data": {"mode": "all"},
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
                "id": "alpha_bound",
                "type": "domain",
                "data": {
                    "mode": "range",
                    "min": 0.25,
                    "max": 0.75,
                },
            },
            {
                "id": "excluded_block",
                "type": "selection",
                "data": {
                    "mode": "block",
                    "items": ["UNET_OUT05_1"],
                },
            },
            {
                "id": "exclude_target",
                "type": "type",
                "data": {"mode": "exclude"},
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
            {"from": "unet", "to": "all_targets"},
            {"from": "all_targets", "to": "alpha"},
            {"from": "alpha", "to": "build_alpha"},
            {"from": "unet", "to": "selected_blocks"},
            {"from": "selected_blocks", "to": "decoder_surface"},
            {"from": "decoder_surface", "to": "alpha_bound"},
            {"from": "alpha_bound", "to": "alpha"},
            {"from": "unet", "to": "excluded_block"},
            {"from": "excluded_block", "to": "exclude_target"},
            {"from": "exclude_target", "to": "build_alpha"},
        ],
    }

    compiled = compile_graph_guide_to_bindings(
        graph,
        base_model_config=_FakeModelConfig("sdxl-sgm", {"unet": ["TEXT_A"]}),
        custom_block_config=_FakeModelConfig(
            "sdxl-optim_blocks_sub",
            {"unet": ["UNET_IN05_1", "UNET_OUT05_1", "UNET_MID00"]},
        ),
    )

    assert [binding.optimizer_param_name for binding in compiled] == [
        "decoder_surface_alpha",
        "UNET_MID00_alpha",
    ]
    assert compiled[0].targets == ("UNET_IN05_1",)
    assert compiled[1].targets == ("UNET_MID00",)
    assert build_optimizer_bounds(compiled) == {
        "decoder_surface_alpha": (0.25, 0.75),
        "UNET_MID00_alpha": (0.0, 1.0),
    }
    assert materialize_recipe_payloads(
        {
            "decoder_surface_alpha": 0.5,
            "UNET_MID00_alpha": 0.9,
        },
        compiled,
    ) == {
        "alpha": {
            "UNET_IN05_1": 0.5,
            "UNET_MID00": 0.9,
        }
    }


def test_build_named_group_type_compiles_multiple_shared_bindings() -> None:
    graph = {
        "version": 1,
        "nodes": [
            {
                "id": "unet",
                "type": "source",
                "data": {
                    "source_kind": "key",
                    "component": "unet",
                },
            },
            {
                "id": "selected_keys",
                "type": "selection",
                "data": {
                    "mode": "regex",
                    "patterns": ["model.diffusion_model.*"],
                },
            },
            {
                "id": "key_groups",
                "type": "type",
                "data": {
                    "mode": "group",
                    "groups": [
                        {"name": "out_0", "patterns": ["*.out.0.*"]},
                        {"name": "time_embed", "patterns": ["*.time_embed.*"]},
                    ],
                },
            },
            {
                "id": "rank_ratio",
                "type": "param",
                "data": {"name": "rank_ratio"},
            },
            {
                "id": "build_rank",
                "type": "build",
                "data": {},
            },
        ],
        "edges": [
            {"from": "unet", "to": "selected_keys"},
            {"from": "selected_keys", "to": "key_groups"},
            {"from": "key_groups", "to": "rank_ratio"},
            {"from": "rank_ratio", "to": "build_rank"},
        ],
    }

    compiled = compile_graph_guide_to_bindings(
        graph,
        base_model_config=_FakeModelConfig(
            "sdxl-sgm",
            {
                "unet": [
                    "model.diffusion_model.out.0.weight",
                    "model.diffusion_model.out.2.weight",
                    "model.diffusion_model.time_embed.weight",
                ]
            },
        ),
        custom_block_config=None,
    )

    assert [binding.optimizer_param_name for binding in compiled] == [
        "out_0_rank_ratio",
        "time_embed_rank_ratio",
    ]
    assert materialize_recipe_payloads(
        {
            "out_0_rank_ratio": 0.5,
            "time_embed_rank_ratio": 0.75,
        },
        compiled,
    ) == {
        "rank_ratio": {
            "model.diffusion_model.out.0.weight": 0.5,
            "model.diffusion_model.time_embed.weight": 0.75,
        }
    }
