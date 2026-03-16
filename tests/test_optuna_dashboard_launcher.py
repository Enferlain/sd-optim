from __future__ import annotations

import importlib
import sys
import types


def _load_optuna_optimizer_module(monkeypatch):
    dummy_optimizer_mod = types.ModuleType("sd_optim.optimizer")

    class DummyOptimizer:
        pass

    dummy_optimizer_mod.Optimizer = DummyOptimizer
    dummy_optimizer_mod.fail_on_error_enabled = lambda cfg: True
    monkeypatch.setitem(sys.modules, "sd_optim.optimizer", dummy_optimizer_mod)
    sys.modules.pop("sd_optim.optuna_optimizer", None)

    return importlib.import_module("sd_optim.optuna_optimizer")


def test_dashboard_launcher_uses_current_python_interpreter(monkeypatch):
    module = _load_optuna_optimizer_module(monkeypatch)
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
    module = _load_optuna_optimizer_module(monkeypatch)
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
