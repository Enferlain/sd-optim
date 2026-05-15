from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace

import pytest
import sd_mecha
from omegaconf import OmegaConf
from torch import Tensor

import sd_optim.merge.recipe_builder as recipe_builder
from sd_optim.merge import recipe_rewrite
from sd_optim.bounds import ParameterHandler
from sd_optim.guide_legacy import materialize_legacy_guide_payloads
from sd_optim.guide_runtime import build_graph_runtime_bundle


@sd_mecha.merge_method(identifier="recipe_bounds_demo_for_tests")
def recipe_bounds_demo_for_tests(
    a: sd_mecha.Parameter(Tensor, "weight"),
    b: sd_mecha.Parameter(Tensor, "weight"),
    *,
    alpha: float = 0.5,
    beta: float = 0.25,
) -> sd_mecha.Return(Tensor, "weight"):
    return a


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


def _old_payload_maps(
    params: dict[str, float],
    param_info: dict[str, dict[str, object]],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], set[str]]:
    block_based_values_per_param: dict[str, dict[str, float]] = {}
    key_based_values_per_param: dict[str, dict[str, float]] = {}
    handled_base_params: set[str] = set()

    for opt_param_name, info in param_info.items():
        if opt_param_name not in params:
            continue

        value = params[opt_param_name]
        strategy = info.get("strategy")
        target_type = info.get("target_type")
        base_param = info.get("base_param")
        item_name = info.get("item_name")
        items_covered = info.get("items_covered", [])

        if not isinstance(base_param, str):
            continue

        handled_base_params.add(base_param)

        if target_type == "block":
            block_based_values_per_param.setdefault(base_param, {})
        elif target_type == "key":
            key_based_values_per_param.setdefault(base_param, {})
        else:
            continue

        if strategy in ["all", "select"]:
            if not isinstance(item_name, str):
                continue
            if target_type == "block":
                block_based_values_per_param[base_param][item_name] = value
            else:
                key_based_values_per_param[base_param][item_name] = value
        elif strategy in ["group", "single"]:
            if not isinstance(items_covered, list):
                continue
            for item in items_covered:
                if target_type == "block":
                    block_based_values_per_param[base_param][item] = value
                else:
                    key_based_values_per_param[base_param][item] = value

    return block_based_values_per_param, key_based_values_per_param, handled_base_params


def test_recipe_rewrite_shows_bounds_assembled_into_recipe_payloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original_recipe = "\n".join(
        [
            "version 0.1.0",
            'model "a.safetensors" model_config="sdxl-sgm" merge_space="weight"',
            'model "b.safetensors" model_config="sdxl-sgm" merge_space="weight"',
            'merge "recipe_bounds_demo_for_tests" &0 &1 alpha=0.5 beta=0.1',
        ]
    )
    recipe_path = tmp_path / "recipe.mecha"
    recipe_path.write_text(original_recipe, encoding="utf-8")

    handler = ParameterHandler.__new__(ParameterHandler)
    handler.cfg = OmegaConf.create(
        {
            "optimization_mode": "recipe",
            "merge": {"merge_method": "recipe_bounds_demo_for_tests"},
            "recipe_optimization": {
                "recipe_path": str(recipe_path),
                "target_params": ["alpha", "beta"],
            },
            "optimization_guide": {
                "components": [
                    {
                        "name": "unet",
                        "strategies": [
                            {
                                "type": "select",
                                "target_type": "key",
                                "optimize_params": ["alpha"],
                                "keys": [
                                    "model.diffusion_model.out.0.weight",
                                    "model.diffusion_model.out.2.weight",
                                ],
                            },
                            {
                                "type": "single",
                                "target_type": "key",
                                "optimize_params": ["beta"],
                            },
                        ],
                    }
                ]
            },
        }
    )
    handler.base_model_config = _FakeModelConfig(
        "sdxl-sgm",
        {
            "unet": [
                "model.diffusion_model.out.0.weight",
                "model.diffusion_model.out.2.weight",
                "model.diffusion_model.out.4.weight",
            ]
        },
    )
    handler.custom_block_config = None
    handler._guide_processing_summary = ParameterHandler._new_guide_processing_summary()

    param_info, _ = handler.get_bounds({})

    merger = SimpleNamespace(
        cfg=handler.cfg,
        base_model_config=handler.base_model_config,
        custom_block_config=None,
        models_dir=tmp_path,
        models=[],
        _model_config_candidates_cache={},
    )

    # Key-only parameter nodes do not require real config conversion.
    monkeypatch.setattr(recipe_builder, "get_conversion_context_node", lambda _: object())

    new_param_nodes = recipe_builder.prepare_param_recipe_args(
        merger,
        {
            "model.diffusion_model.out.0.weight_alpha": 0.75,
            "model.diffusion_model.out.2.weight_alpha": 0.25,
            "unet_single_beta": 0.9,
        },
        param_info,
        recipe_bounds_demo_for_tests,
    )

    assert set(new_param_nodes) == {"alpha", "beta"}

    new_node_strings, param_to_replacement = recipe_rewrite.serialize_nodes_for_rewrite(new_param_nodes)
    rewritten = recipe_rewrite.rewrite_recipe_text(
        original_recipe_text=original_recipe,
        target_node_idx=2,
        new_node_strings=new_node_strings,
        param_to_replacement=param_to_replacement,
    )

    assert 'merge "recipe_bounds_demo_for_tests"' in rewritten
    assert "alpha=&" in rewritten
    assert "beta=&" in rewritten
    assert "model.diffusion_model.out.0.weight=0.75" in rewritten
    assert "model.diffusion_model.out.2.weight=0.25" in rewritten
    assert "model.diffusion_model.out.4.weight=0.9" in rewritten

    rewritten_root = sd_mecha.deserialize(rewritten)
    alpha_node = rewritten_root.bound_args.arguments["alpha"]
    beta_node = rewritten_root.bound_args.arguments["beta"]

    assert alpha_node.value_dict == {
        "model.diffusion_model.out.0.weight": 0.75,
        "model.diffusion_model.out.2.weight": 0.25,
    }
    assert beta_node.value_dict == {
        "model.diffusion_model.out.0.weight": 0.9,
        "model.diffusion_model.out.2.weight": 0.9,
        "model.diffusion_model.out.4.weight": 0.9,
    }


def test_graph_backed_payload_maps_match_legacy_recipe_payload_assembly() -> None:
    handler = ParameterHandler.__new__(ParameterHandler)
    handler.cfg = OmegaConf.create(
        {
            "optimization_mode": "recipe",
            "merge": {"merge_method": "recipe_bounds_demo_for_tests"},
            "recipe_optimization": {
                "target_params": ["alpha", "beta"],
            },
            "optimization_guide": {
                "components": [
                    {
                        "name": "unet",
                        "strategies": [
                            {
                                "type": "all",
                                "target_type": "block",
                                "optimize_params": ["alpha"],
                                "keys": ["UNET_IN00"],
                            },
                            {
                                "type": "select",
                                "target_type": "key",
                                "optimize_params": ["alpha"],
                                "keys": ["TEXT_A"],
                            },
                            {
                                "type": "single",
                                "target_type": "key",
                                "optimize_params": ["beta"],
                            },
                        ],
                    }
                ]
            },
        }
    )
    handler.base_model_config = _FakeModelConfig(
        "sdxl-sgm",
        {
            "unet": ["TEXT_A", "TEXT_B"],
        },
    )
    handler.custom_block_config = _FakeModelConfig(
        "sdxl-optim_blocks_sub",
        {
            "unet": ["UNET_IN00", "UNET_IN01"],
        },
    )
    handler._guide_processing_summary = ParameterHandler._new_guide_processing_summary()

    param_info, _ = handler.get_bounds({})
    sampled_values = {
        "UNET_IN00_alpha": 0.1,
        "UNET_IN01_alpha": 0.2,
        "TEXT_A_alpha": 0.3,
        "unet_single_beta": 0.9,
    }

    old_block, old_key, old_handled = _old_payload_maps(sampled_values, param_info)
    new_block, new_key, new_handled = materialize_legacy_guide_payloads(
        handler.cfg,
        handler.base_model_config,
        handler.custom_block_config,
        sampled_values,
    )

    assert new_block == old_block == {
        "alpha": {
            "UNET_IN00": 0.1,
            "UNET_IN01": 0.2,
        }
    }
    assert new_key == old_key == {
        "alpha": {
            "TEXT_A": 0.3,
        },
        "beta": {
            "TEXT_A": 0.9,
            "TEXT_B": 0.9,
        },
    }
    assert new_handled == old_handled == {"alpha", "beta"}


def test_prepare_param_recipe_args_uses_supplied_param_info(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    handler = ParameterHandler.__new__(ParameterHandler)
    handler.cfg = OmegaConf.create(
        {
            "optimization_mode": "recipe",
            "merge": {"merge_method": "recipe_bounds_demo_for_tests"},
            "recipe_optimization": {
                "target_params": ["alpha"],
            },
            "optimization_guide": {
                "components": [
                    {
                        "name": "unet",
                        "strategies": [
                            {
                                "type": "all",
                                "target_type": "key",
                                "optimize_params": ["alpha"],
                            }
                        ],
                    }
                ]
            },
        }
    )
    handler.base_model_config = _FakeModelConfig(
        "sdxl-sgm",
        {
            "unet": ["TEXT_A", "TEXT_B"],
        },
    )
    handler.custom_block_config = None
    handler._guide_processing_summary = ParameterHandler._new_guide_processing_summary()

    full_param_info, _ = handler.get_bounds({})
    trimmed_param_info = {
        "TEXT_A_alpha": full_param_info["TEXT_A_alpha"],
    }

    merger = SimpleNamespace(
        cfg=handler.cfg,
        base_model_config=handler.base_model_config,
        custom_block_config=None,
        models_dir=tmp_path,
        models=[],
        _model_config_candidates_cache={},
    )
    monkeypatch.setattr(recipe_builder, "get_conversion_context_node", lambda _: object())

    new_param_nodes = recipe_builder.prepare_param_recipe_args(
        merger,
        {
            "TEXT_A_alpha": 0.3,
            "TEXT_B_alpha": 0.8,
        },
        trimmed_param_info,
        recipe_bounds_demo_for_tests,
    )

    assert new_param_nodes["alpha"].value_dict == {
        "TEXT_A": 0.3,
    }


def test_prepare_param_recipe_args_accepts_graph_runtime_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
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
                "id": "selected_keys",
                "type": "selection",
                "data": {
                    "mode": "regex",
                    "patterns": ["TEXT_*"],
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
            {"from": "unet_keys", "to": "selected_keys"},
            {"from": "selected_keys", "to": "all_keys"},
            {"from": "all_keys", "to": "alpha"},
            {"from": "alpha", "to": "build_alpha"},
        ],
    }
    graph_bundle = build_graph_runtime_bundle(
        graph,
        base_model_config=_FakeModelConfig(
            "sdxl-sgm",
            {
                "unet": ["TEXT_A", "TEXT_B", "TEXT_C"],
            },
        ),
        custom_block_config=None,
    )

    merger = SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "optimization_mode": "recipe",
                "merge": {"merge_method": "recipe_bounds_demo_for_tests"},
                "recipe_optimization": {
                    "target_params": ["alpha"],
                },
                "optimization_guide": {},
            }
        ),
        base_model_config=_FakeModelConfig(
            "sdxl-sgm",
            {
                "unet": ["TEXT_A", "TEXT_B", "TEXT_C"],
            },
        ),
        custom_block_config=None,
        models_dir=tmp_path,
        models=[],
        _model_config_candidates_cache={},
    )
    monkeypatch.setattr(recipe_builder, "get_conversion_context_node", lambda _: object())

    new_param_nodes = recipe_builder.prepare_param_recipe_args(
        merger,
        {
            "TEXT_A_alpha": 0.2,
            "TEXT_B_alpha": 0.4,
            "TEXT_C_alpha": 0.8,
        },
        graph_bundle,
        recipe_bounds_demo_for_tests,
    )

    assert new_param_nodes["alpha"].value_dict == {
        "TEXT_A": 0.2,
        "TEXT_B": 0.4,
        "TEXT_C": 0.8,
    }


def test_prepare_param_recipe_args_rejects_graph_params_unknown_to_merge_method(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
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
                "id": "all_keys",
                "type": "type",
                "data": {"mode": "all"},
            },
            {
                "id": "unknown_param",
                "type": "param",
                "data": {"name": "not_a_merge_kwarg"},
            },
            {
                "id": "build_unknown",
                "type": "build",
                "data": {},
            },
        ],
        "edges": [
            {"from": "unet_keys", "to": "all_keys"},
            {"from": "all_keys", "to": "unknown_param"},
            {"from": "unknown_param", "to": "build_unknown"},
        ],
    }
    graph_bundle = build_graph_runtime_bundle(
        graph,
        base_model_config=_FakeModelConfig(
            "sdxl-sgm",
            {
                "unet": ["TEXT_A"],
            },
        ),
        custom_block_config=None,
    )

    merger = SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "optimization_mode": "merge",
                "merge": {"merge_method": "recipe_bounds_demo_for_tests"},
                "optimization_guide": {},
            }
        ),
        base_model_config=_FakeModelConfig(
            "sdxl-sgm",
            {
                "unet": ["TEXT_A"],
            },
        ),
        custom_block_config=None,
        models_dir=tmp_path,
        models=[],
        _model_config_candidates_cache={},
    )
    monkeypatch.setattr(recipe_builder, "get_conversion_context_node", lambda _: object())

    with pytest.raises(ValueError, match="Graph guide parameter.*not_a_merge_kwarg.*recipe_bounds_demo_for_tests"):
        recipe_builder.prepare_param_recipe_args(
            merger,
            {
                "TEXT_A_not_a_merge_kwarg": 0.2,
            },
            graph_bundle,
            recipe_bounds_demo_for_tests,
        )


def test_prepare_param_recipe_args_rejects_custom_bounds_for_graph_runtime_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
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
            {"from": "unet_keys", "to": "all_keys"},
            {"from": "all_keys", "to": "alpha"},
            {"from": "alpha", "to": "build_alpha"},
        ],
    }
    graph_bundle = build_graph_runtime_bundle(
        graph,
        base_model_config=_FakeModelConfig(
            "sdxl-sgm",
            {
                "unet": ["TEXT_A"],
            },
        ),
        custom_block_config=None,
    )

    merger = SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "optimization_mode": "recipe",
                "merge": {"merge_method": "recipe_bounds_demo_for_tests"},
                "recipe_optimization": {
                    "target_params": ["alpha", "beta"],
                },
                "optimization_guide": {
                    "custom_bounds": {
                        "beta": 0.9,
                    }
                },
            }
        ),
        base_model_config=_FakeModelConfig(
            "sdxl-sgm",
            {
                "unet": ["TEXT_A"],
            },
        ),
        custom_block_config=None,
        models_dir=tmp_path,
        models=[],
        _model_config_candidates_cache={},
    )
    monkeypatch.setattr(recipe_builder, "get_conversion_context_node", lambda _: object())

    with pytest.raises(ValueError, match="custom_bounds is a legacy-guide feature"):
        recipe_builder.prepare_param_recipe_args(
            merger,
            {
                "TEXT_A_alpha": 0.2,
            },
            graph_bundle,
            recipe_bounds_demo_for_tests,
        )
