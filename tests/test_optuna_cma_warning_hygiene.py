from __future__ import annotations

import importlib
import sys
import types
import warnings

from omegaconf import OmegaConf


def _load_optuna_optimizer_module(monkeypatch):
    dummy_optimizer_mod = types.ModuleType("sd_optim.optimizer")

    class DummyOptimizer:
        pass

    dummy_optimizer_mod.Optimizer = DummyOptimizer
    monkeypatch.setitem(sys.modules, "sd_optim.optimizer", dummy_optimizer_mod)
    sys.modules.pop("sd_optim.optuna_optimizer", None)

    return importlib.import_module("sd_optim.optuna_optimizer")


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


def _make_optimizer_stub(module, cfg):
    obj = module.OptunaOptimizer.__new__(module.OptunaOptimizer)
    obj.cfg = cfg
    return obj


def test_cmaes_omits_deprecated_restart_kwargs(monkeypatch):
    module = _load_optuna_optimizer_module(monkeypatch)
    cfg = _make_optuna_cfg(
        sampler_type="cmaes",
        extra_sampler={"restart_strategy": "ipop", "inc_popsize": 2, "sigma0": 0.35},
    )
    opt = _make_optimizer_stub(module, cfg)

    captured_kwargs = {}

    class DummyCma:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(module, "CmaEsSampler", DummyCma)

    opt._configure_sampler()

    assert "restart_strategy" not in captured_kwargs
    assert "inc_popsize" not in captured_kwargs


def test_cmaes_suppresses_use_separable_warning(monkeypatch):
    module = _load_optuna_optimizer_module(monkeypatch)
    cfg = _make_optuna_cfg(
        sampler_type="cmaes",
        extra_sampler={"use_separable_cma": True, "sigma0": 0.35},
    )
    opt = _make_optimizer_stub(module, cfg)

    class DummyCma:
        def __init__(self, **kwargs):
            warnings.warn(
                "Argument ``use_separable_cma`` is an experimental feature. The interface can change in the future.",
                UserWarning,
                stacklevel=1,
            )

    monkeypatch.setattr(module, "CmaEsSampler", DummyCma)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        opt._configure_sampler()

    assert not any("use_separable_cma" in str(w.message) for w in caught)
