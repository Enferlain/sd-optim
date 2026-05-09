from __future__ import annotations

from collections.abc import Iterable

import pytest
from omegaconf import OmegaConf


pytest.importorskip("sd_mecha")

from sd_optim.bounds import ParameterHandler  # noqa: E402


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


def _make_handler(
    guide_components: list[dict],
    *,
    base_components: dict[str, Iterable[str]] | None = None,
    block_components: dict[str, Iterable[str]] | None = None,
) -> ParameterHandler:
    handler = ParameterHandler.__new__(ParameterHandler)
    handler.cfg = OmegaConf.create(
        {
            "optimization_mode": "merge",
            "merge_method": "weighted_sum",
            "optimization_guide": {"components": guide_components},
        }
    )
    handler.base_model_config = _FakeModelConfig("base-config", base_components or {})
    handler.custom_block_config = (
        _FakeModelConfig("block-config", block_components)
        if block_components is not None
        else None
    )
    handler._guide_processing_summary = ParameterHandler._new_guide_processing_summary()
    return handler


def test_parameter_handler_init_sets_loaded_configs(caplog: pytest.LogCaptureFixture) -> None:
    cfg = OmegaConf.create({"optimization_guide": {}})
    base_model_config = _FakeModelConfig("base-config", {})
    custom_block_config = _FakeModelConfig("block-config", {})

    caplog.set_level("INFO")
    handler = ParameterHandler(cfg, base_model_config, custom_block_config)

    assert handler.cfg is cfg
    assert handler.base_model_config is base_model_config
    assert handler.custom_block_config is custom_block_config
    assert handler._guide_processing_summary == {
        "components_read": 0,
        "components_used": 0,
        "skipped_components": [],
    }
    assert "ParameterHandler initialized with pre-loaded model configs." in caplog.text


def test_create_parameter_bounds_metadata_supports_all_strategy_types() -> None:
    handler = _make_handler(
        [
            {
                "name": "unet",
                "strategies": [
                    {"type": "all", "target_type": "block", "optimize_params": ["alpha"]},
                    {
                        "type": "select",
                        "target_type": "block",
                        "optimize_params": ["beta"],
                        "keys": ["UNET_IN*"],
                    },
                    {
                        "type": "group",
                        "target_type": "block",
                        "optimize_params": ["gamma"],
                        "groups": [
                            {"name": "encoder", "keys": ["UNET_IN*"]},
                            {"name": "bridge", "keys": ["UNET_MID"]},
                        ],
                    },
                ],
            },
            {
                "name": "text",
                "strategies": [
                    {"type": "single", "target_type": "key", "optimize_params": ["delta"]},
                ],
            },
        ],
        base_components={"text": ["TEXT_A", "TEXT_B"]},
        block_components={"unet": ["UNET_IN00", "UNET_IN01", "UNET_MID"]},
    )

    params_info = handler.create_parameter_bounds_metadata()

    assert set(params_info) == {
        "UNET_IN00_alpha",
        "UNET_IN01_alpha",
        "UNET_MID_alpha",
        "UNET_IN00_beta",
        "UNET_IN01_beta",
        "encoder_gamma",
        "bridge_gamma",
        "text_single_delta",
    }
    assert params_info["UNET_IN00_alpha"]["target_type"] == "block"
    assert params_info["UNET_IN00_beta"]["item_name"] == "UNET_IN00"
    assert set(params_info["encoder_gamma"]["items_covered"]) == {"UNET_IN00", "UNET_IN01"}
    assert params_info["bridge_gamma"]["items_covered"] == ["UNET_MID"]
    assert params_info["text_single_delta"]["target_type"] == "key"
    assert params_info["text_single_delta"]["items_covered"] == ["TEXT_A", "TEXT_B"]


def test_all_strategy_currently_ignores_keys_filter_and_expands_full_component() -> None:
    handler = _make_handler(
        [
            {
                "name": "unet",
                "strategies": [
                    {
                        "type": "all",
                        "target_type": "block",
                        "optimize_params": ["alpha"],
                        "keys": ["UNET_IN00"],
                    }
                ],
            }
        ],
        block_components={"unet": ["UNET_IN00", "UNET_IN01", "UNET_MID"]},
    )

    params_info = handler.create_parameter_bounds_metadata()

    assert set(params_info) == {
        "UNET_IN00_alpha",
        "UNET_IN01_alpha",
        "UNET_MID_alpha",
    }


def test_create_parameter_bounds_metadata_skips_conflicting_assignments(
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = _make_handler(
        [
            {
                "name": "unet",
                "strategies": [
                    {
                        "type": "select",
                        "target_type": "block",
                        "optimize_params": ["alpha"],
                        "keys": ["UNET_IN00"],
                    },
                    {"type": "all", "target_type": "block", "optimize_params": ["alpha"]},
                ],
            }
        ],
        block_components={"unet": ["UNET_IN00", "UNET_IN01"]},
    )

    caplog.set_level("ERROR")
    params_info = handler.create_parameter_bounds_metadata()

    assert set(params_info) == {"UNET_IN00_alpha", "UNET_IN01_alpha"}
    assert "Conflict for item 'UNET_IN00' (param 'alpha')" in caplog.text


def test_create_parameter_bounds_metadata_skips_select_entries_that_match_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = _make_handler(
        [
            {
                "name": "unet",
                "strategies": [
                    {
                        "type": "select",
                        "target_type": "block",
                        "optimize_params": ["alpha"],
                        "keys": ["TEXT_*"],
                    }
                ],
            }
        ],
        block_components={"unet": ["UNET_IN00", "UNET_IN01"]},
    )

    caplog.set_level("WARNING")
    assert handler.create_parameter_bounds_metadata() == {}
    assert "did not match any items in component 'unet'" in caplog.text


def test_create_parameter_bounds_metadata_skips_groups_that_match_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = _make_handler(
        [
            {
                "name": "unet",
                "strategies": [
                    {
                        "type": "group",
                        "target_type": "block",
                        "optimize_params": ["alpha"],
                        "groups": [
                            {"name": "missing", "keys": ["TEXT_*"]},
                        ],
                    }
                ],
            }
        ],
        block_components={"unet": ["UNET_IN00", "UNET_IN01"]},
    )

    caplog.set_level("WARNING")
    assert handler.create_parameter_bounds_metadata() == {}
    assert "Group 'missing' for param 'alpha' did not match any items" in caplog.text
    assert "Group will be skipped" in caplog.text


def test_validate_dependencies_maps_item_and_group_pairs() -> None:
    handler = ParameterHandler.__new__(ParameterHandler)
    params_info = {
        "UNET_IN00_alpha": {"item_name": "UNET_IN00", "base_param": "alpha"},
        "UNET_IN00_beta": {"item_name": "UNET_IN00", "base_param": "beta"},
        "UNET_IN01_alpha": {"item_name": "UNET_IN01", "base_param": "alpha"},
        "UNET_IN01_beta": {"item_name": "UNET_IN01", "base_param": "beta"},
        "encoder_alpha": {"group_name": "encoder", "base_param": "alpha"},
        "encoder_beta": {"group_name": "encoder", "base_param": "beta"},
    }

    dependency_map = handler.validate_dependencies(
        params_info,
        [{"parent": "alpha", "child": "beta", "condition": "> 0.5", "default": 0.0}],
    )

    assert dependency_map == {
        "UNET_IN00_beta": {
            "parent": "UNET_IN00_alpha",
            "condition": "> 0.5",
            "default": 0.0,
        },
        "UNET_IN01_beta": {
            "parent": "UNET_IN01_alpha",
            "condition": "> 0.5",
            "default": 0.0,
        },
        "encoder_beta": {
            "parent": "encoder_alpha",
            "condition": "> 0.5",
            "default": 0.0,
        },
    }


def test_validate_custom_bounds_accepts_omegaconf_range_and_warns_for_log_step_combo(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("WARNING")

    validated = ParameterHandler.validate_custom_bounds(
        OmegaConf.create(
            {
                "alpha": {
                    "range": [1, 3],
                    "log": True,
                    "step": 1,
                }
            }
        )
    )

    assert validated == {
        "alpha": {
            "range": (1.0, 3.0),
            "log": True,
            "step": 1.0,
        }
    }
    assert "both 'log' and 'step' are specified" in caplog.text


def test_create_parameter_bounds_metadata_returns_empty_for_missing_components(
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = _make_handler([])

    caplog.set_level("WARNING")
    assert handler.create_parameter_bounds_metadata() == {}
    assert "No 'components' list found or invalid format" in caplog.text


def test_create_parameter_bounds_metadata_skips_invalid_component_shapes(
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = _make_handler(
        [
            "bad",
            {"strategies": []},
            {"name": "unet", "optimize_params": "bad", "strategies": "bad"},
        ]
    )

    caplog.set_level("WARNING")
    assert handler.create_parameter_bounds_metadata() == {}
    assert "Not a dictionary" in caplog.text
    assert "missing 'name'" in caplog.text
    assert "Invalid 'optimize_params' format for component 'unet'" in caplog.text
    assert "missing a valid 'strategies' list" in caplog.text
    assert handler._guide_processing_summary["skipped_components"] == [
        "component[0]: not a dictionary",
        "component[1]: missing 'name'",
        "component[2]: unet: missing valid 'strategies' list",
    ]


def test_process_strategy_skips_invalid_strategy_inputs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = _make_handler([], block_components={"unet": ["UNET_IN00"]})

    caplog.set_level("WARNING")
    assert handler._process_strategy("unet", ["alpha"], 0, "bad", {}) == {}
    assert handler._process_strategy("unet", ["alpha"], 1, {"type": "bogus"}, {}) == {}
    assert handler._process_strategy("unet", ["alpha"], 2, {"type": "none"}, {}) == {}
    assert "Invalid strategy entry format" in caplog.text
    assert "Missing or invalid strategy 'type'" in caplog.text


def test_process_strategy_skips_when_target_config_or_items_or_params_are_invalid(
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = _make_handler([], block_components={"unet": ["UNET_IN00"]})
    empty_handler = _make_handler([], block_components={"unet": []})

    caplog.set_level("WARNING")
    assert (
        handler._process_strategy(
            "unet",
            ["alpha"],
            0,
            {"type": "all", "target_type": "key"},
            {},
        )
        == {}
    )
    assert (
        empty_handler._process_strategy(
            "unet",
            ["alpha"],
            1,
            {"type": "all", "target_type": "block"},
            {},
        )
        == {}
    )
    assert (
        handler._process_strategy(
            "unet",
            ["alpha"],
            2,
            {"type": "all", "target_type": "block", "optimize_params": []},
            {},
        )
        == {}
    )
    assert "target_type='key' but component 'unet' not found in base_model_config" in caplog.text
    assert "has no items in config 'block-config'" in caplog.text


def test_process_strategy_returns_empty_when_no_optimize_params_after_fallback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = _make_handler([], block_components={"unet": ["UNET_IN00"]})

    caplog.set_level("DEBUG")
    out = handler._process_strategy(
        "unet",
        [],
        0,
        {"type": "all", "target_type": "block", "optimize_params": "bad"},
        {},
    )

    assert out == {}
    assert "No 'optimize_params' specified for strategy 'all'" in caplog.text


def test_determine_target_config_warns_when_explicit_block_target_is_missing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = _make_handler([], base_components={"unet": ["UNET_IN00"]}, block_components={})

    caplog.set_level("WARNING")
    assert handler._determine_target_config("unet", {"target_type": "block"}) == (None, False, "")
    assert "target_type='block' but component 'unet' not found in custom_block_config" in caplog.text


def test_determine_target_config_auto_and_explicit_paths(
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = _make_handler(
        [],
        base_components={"text": ["TEXT_A"]},
        block_components={"unet": ["UNET_IN00"]},
    )
    missing = _make_handler([], base_components={}, block_components={})

    caplog.set_level("WARNING")
    assert handler._determine_target_config("unet", {"target_type": "block"}) == (
        handler.custom_block_config,
        True,
        "block",
    )
    assert handler._determine_target_config("text", {"target_type": "key"}) == (
        handler.base_model_config,
        False,
        "key",
    )
    assert handler._determine_target_config("unet", {}) == (
        handler.custom_block_config,
        True,
        "block",
    )
    assert handler._determine_target_config("text", {}) == (
        handler.base_model_config,
        False,
        "key",
    )
    assert missing._determine_target_config("missing", {}) == (None, False, "")
    assert "Component 'missing' not found in known configs" in caplog.text


def test_get_component_items_returns_empty_for_missing_component_or_bad_shape(
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = _make_handler([], block_components={"unet": ["UNET_IN00"]})

    class _BadConfig:
        identifier = "bad-config"

        def components(self) -> dict[str, object]:
            return {"unet": object()}

    caplog.set_level("WARNING")
    assert handler._get_component_items(handler.custom_block_config, "missing") == []
    assert handler._get_component_items(_BadConfig(), "unet") == []
    assert "Component name 'missing' not found within config 'block-config'" in caplog.text
    assert "Error accessing items for component 'unet' using config 'bad-config'" in caplog.text


def test_process_select_strategy_handles_invalid_keys_and_no_matches(
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = _make_handler([])
    base_metadata = {"strategy": "select", "target_type": "block", "component_name": "unet"}

    caplog.set_level("WARNING")
    assert handler._process_select_strategy(["alpha"], ["UNET_IN00"], base_metadata, {}, "unet", {}) == {}
    assert (
        handler._process_select_strategy(
            ["alpha"],
            ["UNET_IN00"],
            base_metadata,
            {"keys": ["TEXT_*"]},
            "unet",
            {},
        )
        == {}
    )
    assert "'select' strategy needs a valid 'keys' list" in caplog.text
    assert "did not match any items in component 'unet'" in caplog.text


def test_process_select_strategy_skips_conflicting_item_matches(
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = _make_handler([])
    base_metadata = {"strategy": "select", "target_type": "block", "component_name": "unet"}

    caplog.set_level("ERROR")
    out = handler._process_select_strategy(
        ["alpha"],
        ["UNET_IN00"],
        base_metadata,
        {"keys": ["UNET_*"]},
        "unet",
        {("alpha", "UNET_IN00"): "taken"},
    )

    assert out == {}
    assert "cannot assign by 'select' pattern 'UNET_*'" in caplog.text


def test_process_group_strategy_handles_invalid_group_shapes_and_conflicts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = _make_handler([])
    base_metadata = {"strategy": "group", "target_type": "block", "component_name": "unet"}

    caplog.set_level("WARNING")
    assert handler._process_group_strategy(["alpha"], ["UNET_IN00"], base_metadata, {}, "unet", {}) == {}
    assert (
        handler._process_group_strategy(
            ["alpha"],
            ["UNET_IN00"],
            base_metadata,
            {"groups": ["bad", {"name": "g1", "keys": "bad"}, {"name": "g2", "keys": ["TEXT_*"]}]},
            "unet",
            {},
        )
        == {}
    )
    assert (
        handler._process_group_strategy(
            ["alpha"],
            ["UNET_IN00"],
            base_metadata,
            {"groups": [{"name": "g3", "keys": ["UNET_*"]}]},
            "unet",
            {("alpha", "UNET_IN00"): "taken"},
        )
        == {}
    )
    assert "'group' strategy needs a valid 'groups' list" in caplog.text
    assert "Invalid group format at index 0" in caplog.text
    assert "Invalid group format (missing name or keys list)" in caplog.text
    assert "did not match any items in component 'unet'" in caplog.text
    assert "Group will be skipped" in caplog.text


def test_process_single_strategy_handles_empty_items_and_conflicts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = _make_handler([])
    base_metadata = {"strategy": "single", "target_type": "block", "component_name": "unet"}

    caplog.set_level("WARNING")
    assert handler._process_single_strategy(["alpha"], [], base_metadata, {}, "unet", {}) == {}
    assert (
        handler._process_single_strategy(
            ["alpha"],
            ["UNET_IN00"],
            base_metadata,
            {},
            "unet",
            {("alpha", "UNET_IN00"): "taken"},
        )
        == {}
    )
    assert "No items found for component 'unet'" in caplog.text
    assert "cannot assign by 'single:unet_single'" in caplog.text


def test_get_bounds_returns_empty_when_no_params_generated() -> None:
    handler = ParameterHandler.__new__(ParameterHandler)
    handler.create_parameter_bounds_metadata = lambda: {}

    assert handler.get_bounds({}) == ({}, {})


def test_build_parameter_space_summary_counts_bound_shapes() -> None:
    handler = ParameterHandler.__new__(ParameterHandler)
    handler.cfg = OmegaConf.create({"optimization_mode": "merge", "merge_method": "weighted_sum"})
    handler.base_model_config = _FakeModelConfig("base", {})
    handler.custom_block_config = None
    handler._guide_processing_summary = {
        "components_read": 1,
        "components_used": 1,
        "skipped_components": [],
    }

    summary = handler._build_parameter_space_summary(
        params_info={
            "fixed": {"strategy": "single", "target_type": "key", "bounds": 1.0},
            "categorical": {"strategy": "single", "target_type": "key", "bounds": [0, 1]},
            "continuous": {"strategy": "all", "target_type": "block", "bounds": {"range": (0.0, 1.0)}},
        },
        optimizer_pbounds={"fixed": 1.0, "categorical": [0, 1], "continuous": {"range": (0.0, 1.0)}},
        exact_override_count=0,
        base_override_count=0,
        unmatched_custom_bounds=[],
    )

    assert summary["fixed_count"] == 1
    assert summary["categorical_count"] == 1
    assert summary["continuous_count"] == 1


def test_validate_dependencies_handles_empty_and_invalid_dependency_entries(
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = ParameterHandler.__new__(ParameterHandler)
    params_info = {"UNET_IN00_alpha": {"item_name": "UNET_IN00", "base_param": "alpha"}}

    assert handler.validate_dependencies(params_info, None) == {}

    caplog.set_level("DEBUG")
    out = handler.validate_dependencies(
        params_info,
        [
            {"child": "beta"},
            {"parent": "missing", "child": "beta"},
            {"parent": "alpha", "child": "beta"},
        ],
    )

    assert out == {}
    assert "missing 'parent' or 'child'" in caplog.text
    assert "parent base parameter 'missing' not found" in caplog.text
    assert "No pairs found for dependency 'alpha' -> 'beta'" in caplog.text


def test_validate_custom_bounds_returns_empty_for_none() -> None:
    assert ParameterHandler.validate_custom_bounds(None) == {}


def test_validate_custom_bounds_drops_invalid_range_shapes_and_types(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("ERROR")

    out = ParameterHandler.validate_custom_bounds(
        {
            "missing_range": {"step": 1},
            "bad_range_length": {"range": [0, 1, 2]},
            "bad_tuple": (0, 1, 2),
            "bad_type": object(),
            "float_string": "(1.5, 2)",
        }
    )

    assert out["float_string"] == (1.5, 2)
    assert "missing_range" not in out
    assert "bad_range_length" not in out
    assert "bad_tuple" not in out
    assert "bad_type" not in out
    assert "requires a 'range' key" in caplog.text
    assert "'range' must have 2 values" in caplog.text
    assert "Range tuple must have 2 values" in caplog.text
    assert "Bound must be a tuple (range), list (categorical), dict (advanced), int, or float." in caplog.text
