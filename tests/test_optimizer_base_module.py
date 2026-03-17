from __future__ import annotations

import importlib
import inspect
import sys
import types

import pytest


def test_optimizer_base_module_exports_optimizer_class() -> None:
    module = importlib.import_module("sd_optim.core.optimizer_base")

    assert inspect.isclass(module.Optimizer)


def test_optuna_optimizer_uses_core_optimizer_base() -> None:
    base_module = importlib.import_module("sd_optim.core.optimizer_base")
    optuna_module = importlib.import_module("sd_optim.optimizers.optuna.optimizer")

    assert issubclass(optuna_module.OptunaOptimizer, base_module.Optimizer)


def test_top_level_optuna_optimizer_module_is_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("sd_optim.optuna_optimizer")


def test_bayes_optimizer_uses_core_optimizer_base(monkeypatch) -> None:
    base_module = importlib.import_module("sd_optim.core.optimizer_base")

    bayes_opt_module = types.ModuleType("bayes_opt")
    bayes_opt_module.BayesianOptimization = type("BayesianOptimization", (), {})
    bayes_opt_module.Events = type("Events", (), {})
    bayes_opt_module.UtilityFunction = type("UtilityFunction", (), {})

    bayes_opt_logger_module = types.ModuleType("bayes_opt.logger")
    bayes_opt_logger_module.JSONLogger = type("JSONLogger", (), {})

    bayes_opt_domain_module = types.ModuleType("bayes_opt.domain_reduction")
    bayes_opt_domain_module.SequentialDomainReductionTransformer = type(
        "SequentialDomainReductionTransformer",
        (),
        {},
    )

    monkeypatch.setitem(sys.modules, "bayes_opt", bayes_opt_module)
    monkeypatch.setitem(sys.modules, "bayes_opt.logger", bayes_opt_logger_module)
    monkeypatch.setitem(sys.modules, "bayes_opt.domain_reduction", bayes_opt_domain_module)
    sys.modules.pop("sd_optim.optimizers.bayes.optimizer", None)

    bayes_module = importlib.import_module("sd_optim.optimizers.bayes.optimizer")

    assert issubclass(bayes_module.BayesOptimizer, base_module.Optimizer)


def test_top_level_bayes_optimizer_module_is_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("sd_optim.bayes_optimizer")
