"""Characterization tests to protect behavior during optimizer reorganization.

These tests are intentionally focused on stable contracts:
- bounds parsing semantics
- Optuna sampler/config wiring
- known issues documented via xfail markers
"""

from __future__ import annotations

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
