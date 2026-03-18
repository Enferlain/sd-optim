"""Characterization tests to protect behavior during optimizer reorganization.

These tests are intentionally focused on stable contracts:
- bounds parsing semantics
- Optuna sampler/config wiring
- known issues documented via xfail markers
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf


pytest.importorskip("optuna")
pytest.importorskip("sd_mecha")

from optuna.samplers import CmaEsSampler, TPESampler  # noqa: E402

from sd_optim.bounds import ParameterHandler  # noqa: E402
from sd_optim.optimizers.optuna.optimizer import OptunaOptimizer  # noqa: E402


def _make_optuna_cfg(*, sampler_type: str, extra_sampler: dict | None = None):
    sampler = {"type": sampler_type}
    if extra_sampler:
        sampler.update(extra_sampler)
    return OmegaConf.create(
        {
            "optimizer": {
                "n_iters": 10,
                "init_points": 5,
                "random_state": 218,
                "optuna_config": {
                    "sampler": sampler,
                    "pruner_type": "median",
                    "use_pruning": False,
                },
            }
        }
    )


def _make_optimizer_stub(cfg):
    # Avoid full __init__ (which wires merger/generator/scorer); we only need cfg.
    obj = OptunaOptimizer.__new__(OptunaOptimizer)
    obj.cfg = cfg
    return obj


def test_validate_custom_bounds_parses_supported_formats():
    custom_bounds = {
        "from_str": "(0.25, 1.0)",
        "categorical": [0.0, 1.0],
        "advanced": {"range": [0, 1], "log": False, "step": 0.1},
        "tuple_range": (0.0, 1.0),
        "fixed": 1.0,
    }

    out = ParameterHandler.validate_custom_bounds(custom_bounds)

    assert out["from_str"] == (0.25, 1.0)
    assert out["categorical"] == [0.0, 1.0]
    assert out["advanced"] == {"range": (0.0, 1.0), "log": False, "step": 0.1}
    assert out["tuple_range"] == (0.0, 1.0)
    assert out["fixed"] == 1.0


def test_validate_custom_bounds_drops_invalid_entries():
    custom_bounds = {
        "bad_str": "not_a_tuple",
        "bad_tuple": (0, 1, 2),
        "ok": [0.0, 1.0],
    }

    out = ParameterHandler.validate_custom_bounds(custom_bounds)

    assert "ok" in out
    assert "bad_str" not in out
    assert "bad_tuple" not in out


def test_configure_sampler_tpe_initializes():
    cfg = _make_optuna_cfg(
        sampler_type="tpe",
        extra_sampler={"multivariate": True, "group": True, "constant_liar": True},
    )
    opt = _make_optimizer_stub(cfg)
    sampler = opt._configure_sampler()
    assert isinstance(sampler, TPESampler)


def test_configure_sampler_cmaes_initializes():
    cfg = _make_optuna_cfg(
        sampler_type="cmaes",
        extra_sampler={"sigma0": 0.35, "use_separable_cma": True},
    )
    opt = _make_optimizer_stub(cfg)
    sampler = opt._configure_sampler()
    assert isinstance(sampler, CmaEsSampler)


def test_validate_optimizer_config_rejects_grid_without_search_space():
    cfg = _make_optuna_cfg(sampler_type="grid")
    opt = _make_optimizer_stub(cfg)
    assert opt.validate_optimizer_config() is False


def test_configure_sampler_qmc_initializes():
    cfg = _make_optuna_cfg(
        sampler_type="qmc",
        extra_sampler={"qmc_type": "sobol", "scramble": True},
    )
    opt = _make_optimizer_stub(cfg)
    sampler = opt._configure_sampler()
    # If typo is fixed, this should pass and return a QMCSampler.
    assert sampler.__class__.__name__ == "QMCSampler"


def test_validate_optimizer_config_warns_on_unknown_pruner_from_optuna_config(caplog):
    cfg = _make_optuna_cfg(sampler_type="tpe")
    cfg.optimizer.optuna_config.use_pruning = True
    cfg.optimizer.optuna_config.pruner_type = "not_a_real_pruner"

    opt = _make_optimizer_stub(cfg)
    caplog.set_level("WARNING")
    opt.validate_optimizer_config()
    assert "Unknown pruner_type" in caplog.text


def test_get_bounds_logs_compact_summary_at_info(caplog: pytest.LogCaptureFixture) -> None:
    handler = ParameterHandler.__new__(ParameterHandler)
    handler.cfg = OmegaConf.create(
        {
            "optimization_mode": "merge",
            "merge_method": "weighted_sum",
            "optimization_guide": {"custom_block_config_id": "sdxl-optim_blocks_sub"},
        }
    )
    handler.base_model_config = SimpleNamespace(identifier="sdxl-sgm")
    handler.custom_block_config = SimpleNamespace(identifier="sdxl-optim_blocks_sub")
    handler._guide_processing_summary = {
        "components_read": 2,
        "components_used": 1,
        "skipped_components": ["component[1]: missing 'name'"],
    }
    handler.create_parameter_bounds_metadata = lambda: {
        "UNET_IN00_alpha": {
            "strategy": "all",
            "target_type": "block",
            "component_name": "unet",
            "item_name": "UNET_IN00",
            "base_param": "alpha",
            "bounds": (0.0, 1.0),
        },
        "UNET_IN01_alpha": {
            "strategy": "all",
            "target_type": "block",
            "component_name": "unet",
            "item_name": "UNET_IN01",
            "base_param": "alpha",
            "bounds": (0.0, 1.0),
        },
        "UNET_MID_beta": {
            "strategy": "single",
            "target_type": "key",
            "component_name": "diffuser",
            "group_name": "diffuser_single",
            "base_param": "beta",
            "bounds": 1.0,
        },
    }

    caplog.set_level("INFO")
    handler.get_bounds(
        {
            "UNET_IN00_alpha": (0.2, 0.8),
            "beta": [0, 1],
            "unused": 5.0,
        }
    )

    assert "Guide / Parameter Space Summary" in caplog.text
    assert "components read: 2" in caplog.text
    assert "components used: 1" in caplog.text
    assert "component[1]: missing 'name'" in caplog.text
    assert "total generated parameters: 3" in caplog.text
    assert "parameters updated by exact custom bounds: 1" in caplog.text
    assert "parameters updated by base-name custom bounds: 1" in caplog.text
    assert "unused custom_bounds:" in caplog.text
    assert "unused" in caplog.text
    assert "full parameter list: available at DEBUG" in caplog.text
    assert "UNET_IN00_alpha: {" not in caplog.text


def test_create_parameter_bounds_metadata_moves_count_log_to_debug(caplog: pytest.LogCaptureFixture) -> None:
    handler = ParameterHandler.__new__(ParameterHandler)
    handler.cfg = OmegaConf.create(
        {
            "optimization_guide": {
                "components": [
                    {"name": "unet"},
                    {"name": "te"},
                ]
            }
        }
    )

    def _fake_process_component(component_index: int, component_config_raw: dict, assigned_items: dict) -> dict:
        if component_index == 0:
            return {"UNET_alpha": {"bounds": (0.0, 1.0)}}
        return {}

    handler._process_component = _fake_process_component

    caplog.set_level("INFO")
    handler.create_parameter_bounds_metadata()
    assert "Generated metadata for 1 optimization parameters based on guide." not in caplog.text

    caplog.clear()
    caplog.set_level("DEBUG")
    handler.create_parameter_bounds_metadata()
    assert "Generated metadata for 1 optimization parameters based on guide." in caplog.text


def test_get_bounds_summarizes_unused_custom_bounds_without_per_key_debug_lines(
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = ParameterHandler.__new__(ParameterHandler)
    handler.cfg = OmegaConf.create(
        {
            "optimization_mode": "merge",
            "merge_method": "weighted_sum",
            "optimization_guide": {},
        }
    )
    handler.base_model_config = SimpleNamespace(identifier="sdxl-sgm")
    handler.custom_block_config = None
    handler._guide_processing_summary = {
        "components_read": 1,
        "components_used": 1,
        "skipped_components": [],
    }
    handler.create_parameter_bounds_metadata = lambda: {
        "UNET_IN00_alpha": {
            "strategy": "all",
            "target_type": "block",
            "component_name": "unet",
            "item_name": "UNET_IN00",
            "base_param": "alpha",
            "bounds": (0.0, 1.0),
        }
    }

    caplog.set_level("DEBUG")
    handler.get_bounds({"unused": 5.0})

    assert "unused custom_bounds:" in caplog.text
    assert "  - unused" in caplog.text
    assert "Custom bound key 'unused' did not match any generated optimizer parameter or base_param." not in caplog.text


def test_get_bounds_logs_full_parameter_list_at_debug(caplog: pytest.LogCaptureFixture) -> None:
    handler = ParameterHandler.__new__(ParameterHandler)
    handler.cfg = OmegaConf.create(
        {
            "optimization_mode": "merge",
            "merge_method": "weighted_sum",
            "optimization_guide": {},
        }
    )
    handler.base_model_config = SimpleNamespace(identifier="sdxl-sgm")
    handler.custom_block_config = None
    handler._guide_processing_summary = {
        "components_read": 1,
        "components_used": 1,
        "skipped_components": [],
    }
    handler.create_parameter_bounds_metadata = lambda: {
        "UNET_IN00_alpha": {
            "strategy": "all",
            "target_type": "block",
            "component_name": "unet",
            "item_name": "UNET_IN00",
            "base_param": "alpha",
            "bounds": (0.0, 1.0),
        }
    }

    caplog.set_level("DEBUG")
    handler.get_bounds({})

    assert "--- Final 1 Optimization Parameter Details" in caplog.text
    assert "UNET_IN00_alpha: {" in caplog.text
