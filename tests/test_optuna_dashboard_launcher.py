from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf


def _load_dashboard_module():
    return importlib.import_module("sd_optim.optimizers.optuna.dashboard")


def test_dashboard_launcher_uses_current_python_interpreter(monkeypatch):
    module = _load_dashboard_module()
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    captured = {}

    class DummyProcess:
        pid = 4321

        def poll(self):
            return None

    def fake_popen(cmd, stdout, stderr, text, creationflags):
        captured["cmd"] = cmd
        captured["stdout"] = stdout
        captured["stderr"] = stderr
        captured["text"] = text
        captured["creationflags"] = creationflags
        return DummyProcess()

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)

    process = module.run_dashboard_in_background("sqlite:///tmp/test.db", 8080)

    assert process is not None
    assert captured["cmd"][:3] == [sys.executable, "-m", "optuna_dashboard"]


def test_dashboard_launcher_returns_none_when_process_exits_early(monkeypatch):
    module = _load_dashboard_module()
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    class ExitingProcess:
        pid = 123

        def poll(self):
            return 1

        def communicate(self, timeout):
            return ("stdout message", "stderr message")

    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: ExitingProcess())

    process = module.run_dashboard_in_background("sqlite:///tmp/test.db", 8080)

    assert process is None


def test_start_dashboard_for_optimizer_uses_study_storage_path(monkeypatch, tmp_path: Path):
    module = _load_dashboard_module()
    optimizer = SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "optimization_mode": "merge",
                "merge_method": "weighted_sum",
                "scorer_method": ["manual"],
                "optimizer": {"optuna_config": {"storage_dir": str(tmp_path)}},
            }
        ),
        optuna_storage_dir=tmp_path,
    )
    captured = {}

    def fake_runner(storage_uri: str, port: int):
        captured["storage_uri"] = storage_uri
        captured["port"] = port
        return "process"

    monkeypatch.setattr(module, "run_dashboard_in_background", fake_runner)

    process = module.start_dashboard_for_optimizer(optimizer, port=9090)

    assert process == "process"
    assert captured["port"] == 9090
    assert captured["storage_uri"] == f"sqlite:///{(tmp_path / 'optuna_merge_weighted_sum_manual.db').resolve()}"
