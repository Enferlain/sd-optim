from __future__ import annotations

import importlib
import inspect
import sys
import types
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf


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


def test_setup_parameter_space_keeps_parameter_count_log_at_debug(caplog: pytest.LogCaptureFixture) -> None:
    module = importlib.import_module("sd_optim.core.optimizer_base")

    class _TestOptimizer(module.Optimizer):
        def optimize(self) -> None:
            return None

        def validate_optimizer_config(self) -> bool:
            return True

        def get_optimization_history(self) -> list:
            return []

        def get_best_parameters(self) -> dict:
            return {}

        def postprocess(self) -> None:
            return None

    optimizer = object.__new__(_TestOptimizer)
    optimizer.cfg = OmegaConf.create({"optimization_guide": {"custom_bounds": {}}})
    optimizer.bounds_initializer = SimpleNamespace(
        get_bounds=lambda custom_bounds: (  # noqa: ARG005
            {"alpha": {"bounds": (0.0, 1.0)}},
            {"alpha": (0.0, 1.0)},
        )
    )

    caplog.set_level("INFO")
    optimizer.setup_parameter_space()
    assert "Prepared 1 parameters for the optimizer with specific bounds." not in caplog.text

    caplog.clear()
    caplog.set_level("DEBUG")
    optimizer.setup_parameter_space()
    assert "Prepared 1 parameters for the optimizer with specific bounds." in caplog.text
