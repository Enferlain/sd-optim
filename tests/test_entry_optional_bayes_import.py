from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path


def test_entry_module_imports_without_bayes_opt_installed(monkeypatch) -> None:
    # Avoid importing full utils stack (pynput/X11) for this import guard test.
    monkeypatch.setitem(sys.modules, "sd_optim.utils", types.ModuleType("sd_optim.utils"))

    module_path = Path(__file__).resolve().parents[1] / "sd_optim.py"
    spec = importlib.util.spec_from_file_location("sd_optim_entry_test", module_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def test_entry_module_sets_httpcore_http11_logger_to_warning(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "sd_optim.utils", types.ModuleType("sd_optim.utils"))
    logging.getLogger("httpcore.http11").setLevel(logging.NOTSET)

    module_path = Path(__file__).resolve().parents[1] / "sd_optim.py"
    spec = importlib.util.spec_from_file_location("sd_optim_entry_logging_test", module_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert logging.getLogger("httpcore.http11").level == logging.WARNING
