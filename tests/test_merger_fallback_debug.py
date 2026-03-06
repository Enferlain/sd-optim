from __future__ import annotations

import importlib
import logging
import sys
import types

import sd_mecha
from omegaconf import OmegaConf
from sd_mecha.recipe_nodes import MergeRecipeNode
from sd_mecha.streaming import StateDictKeyError


def _import_merger_with_pynput_stub():
    pynput_mod = types.ModuleType("pynput")
    keyboard_mod = types.ModuleType("pynput.keyboard")
    keyboard_mod.Key = types.SimpleNamespace(ctrl="ctrl", shift="shift", alt="alt")
    pynput_mod.keyboard = keyboard_mod
    sys.modules.setdefault("pynput", pynput_mod)
    sys.modules.setdefault("pynput.keyboard", keyboard_mod)
    return importlib.import_module("sd_optim.merger")


merger_mod = _import_merger_with_pynput_stub()


def test_fallback_debug_logged_logs_key_at_debug(caplog) -> None:
    class MissingStateDict(dict):
        def __getitem__(self, key):
            raise StateDictKeyError(key)

    caplog.set_level(logging.DEBUG, logger="sd_optim.merger")
    value = merger_mod.fallback_debug_logged.merge_key(
        [MissingStateDict(), {"x.key": 123}],
        {},
        "x.key",
        None,
    )

    assert value == 123
    assert any("Using fallback for key: x.key" in rec.message for rec in caplog.records)


def test_fallback_debug_logged_suppresses_key_log_above_debug(caplog) -> None:
    class MissingStateDict(dict):
        def __getitem__(self, key):
            raise StateDictKeyError(key)

    caplog.set_level(logging.INFO, logger="sd_optim.merger")
    value = merger_mod.fallback_debug_logged.merge_key(
        [MissingStateDict(), {"x.key": 456}],
        {},
        "x.key",
        None,
    )

    assert value == 456
    assert not any("Using fallback for key: x.key" in rec.message for rec in caplog.records)


def test_execute_recipe_wraps_fallback_in_recipe_instead_of_sd_mecha_kwarg(monkeypatch, tmp_path) -> None:
    merger = object.__new__(merger_mod.Merger)
    merger.cfg = OmegaConf.create(
        {
            "optimization_mode": "merge",
            "fallback_model_index": 0,
            "device": "cpu",
            "merge_dtype": "fp32",
            "save_dtype": "fp32",
            "threads": 1,
        }
    )
    merger.models = [sd_mecha.model("fallback.safetensors")]
    merger.models_dir = tmp_path

    captured: dict = {}

    def _fake_merge(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(sd_mecha, "merge", _fake_merge)

    merger._execute_recipe(
        final_recipe_node=sd_mecha.model("main.safetensors"),
        model_path=tmp_path / "out.safetensors",
    )

    assert captured["fallback_model"] is None
    assert isinstance(captured["recipe"], MergeRecipeNode)
    assert captured["recipe"].merge_method.identifier == "fallback_debug_logged"
