from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from sd_optim.config.dataclasses.optimizer import OptimizerConfig, OptunaConfig
from sd_optim.config.dataclasses.run import SdOptimConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_entry_module(module_name: str):
    module_path = PROJECT_ROOT / "sd_optim.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_entry_selects_optimizer_from_narrow_optimizer_config() -> None:
    module = _load_entry_module("sd_optim_entry_runtime_access_test")
    optimizer_config = OmegaConf.structured(OptimizerConfig(bayes=False, optuna=True))

    optimizer_class, optimizer_name, optimizer_kind = module._select_optimizer_class(optimizer_config)

    assert optimizer_class.__name__ == "OptunaOptimizer"
    assert optimizer_name == "Optuna"
    assert optimizer_kind == "optuna"


def test_entry_extension_paths_use_typed_root_config(tmp_path: Path) -> None:
    module = _load_entry_module("sd_optim_entry_path_access_test")
    config_dir = tmp_path / "configs"
    conversion_dir = tmp_path / "conversions"
    cfg = OmegaConf.structured(SdOptimConfig)
    cfg.paths.configs_dir = str(config_dir)
    cfg.paths.conversion_dir = str(conversion_dir)

    resolved_config_dir, resolved_conversion_dir = module._determine_extension_paths(cfg)

    assert resolved_config_dir == config_dir.resolve()
    assert resolved_conversion_dir == conversion_dir.resolve()


def test_entry_rejects_invalid_optimizer_selection_before_importing_optimizer() -> None:
    module = _load_entry_module("sd_optim_entry_invalid_optimizer_test")
    optimizer_config = OmegaConf.structured(OptimizerConfig(bayes=False, optuna=False))

    with pytest.raises(SystemExit):
        module._select_optimizer_class(optimizer_config)


def test_runtime_log_iteration_uses_structured_optimizer_config(caplog: pytest.LogCaptureFixture) -> None:
    runtime = __import__("sd_optim.core.optimizer_runtime", fromlist=["log_iteration_start"])
    cfg = OmegaConf.structured(SdOptimConfig(optimizer=OptimizerConfig(init_points=2, optuna_config=OptunaConfig())))
    optimizer = type("OptimizerStub", (), {"cfg": cfg})()

    caplog.set_level("INFO")
    runtime.log_iteration_start(optimizer, {"alpha": 0.1}, effective_iteration=3)

    assert "Starting optimization Phase" in caplog.text


def test_runtime_uses_nested_generation_config_for_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = __import__("sd_optim.core.optimizer_runtime", fromlist=["_build_generation_client_settings"])
    cfg = OmegaConf.structured(SdOptimConfig)
    cfg.generation.generator_concurrency_limit = 3
    cfg.generation.generator_keepalive_interval = 15
    cfg.generation.generator_total_timeout = 42

    connector_kwargs, timeout_settings = runtime._build_generation_client_settings(cfg.generation)

    assert connector_kwargs == {"limit": 3, "keepalive_timeout": 15}
    assert timeout_settings.total == 42
