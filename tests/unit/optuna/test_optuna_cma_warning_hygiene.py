from __future__ import annotations

import importlib
import warnings

from omegaconf import OmegaConf


def _load_sampler_factory_module():
    return importlib.import_module("sd_optim.optimizers.optuna.sampler_factory")


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


def test_cmaes_omits_deprecated_restart_kwargs(monkeypatch):
    module = _load_sampler_factory_module()
    cfg = _make_optuna_cfg(
        sampler_type="cmaes",
        extra_sampler={"restart_strategy": "ipop", "inc_popsize": 2, "sigma0": 0.35},
    )

    captured_kwargs = {}

    class DummyCma:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(module, "CmaEsSampler", DummyCma)

    module.configure_sampler(cfg)

    assert "restart_strategy" not in captured_kwargs
    assert "inc_popsize" not in captured_kwargs


def test_cmaes_suppresses_use_separable_warning(monkeypatch):
    module = _load_sampler_factory_module()
    cfg = _make_optuna_cfg(
        sampler_type="cmaes",
        extra_sampler={"use_separable_cma": True, "sigma0": 0.35},
    )

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
        module.configure_sampler(cfg)

    assert not any("use_separable_cma" in str(w.message) for w in caught)
