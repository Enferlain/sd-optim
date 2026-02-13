from __future__ import annotations

import types

import pytest
from omegaconf import OmegaConf

import sd_optim.scorer as scorer_mod


def test_import_attr_returns_none_on_missing_module(monkeypatch):
    def fake_import_module(_name):
        raise ImportError("missing optional dependency")

    monkeypatch.setattr(scorer_mod.importlib, "import_module", fake_import_module)
    assert scorer_mod._import_attr("sd_optim.models.NotReal", "Nope") is None


def test_get_scorer_class_imports_only_requested_module(monkeypatch):
    calls: list[str] = []

    class DummyScorer:
        pass

    def fake_import_module(name):
        calls.append(name)
        return types.SimpleNamespace(DummyScorer=DummyScorer)

    monkeypatch.setattr(scorer_mod.importlib, "import_module", fake_import_module)
    monkeypatch.setitem(
        scorer_mod.SCORER_CLASS_PATHS,
        "dummy",
        ("sd_optim.models.DummyScorer", "DummyScorer"),
    )
    scorer_mod._SCORER_CLASS_CACHE.pop("dummy", None)

    resolved = scorer_mod._get_scorer_class("dummy")
    assert resolved is DummyScorer
    assert calls == ["sd_optim.models.DummyScorer"]


def test_get_scorer_class_returns_none_for_unknown():
    assert scorer_mod._get_scorer_class("this_is_not_a_scorer") is None


def test_rembg_required_scorer_raises_clear_error_without_rembg(monkeypatch):
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "rembg":
            raise ImportError("No module named rembg")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    cfg = OmegaConf.create({"scorer_method": ["textureclean"]})

    with pytest.raises(ImportError, match="requires 'rembg'"):
        scorer_mod.AestheticScorer(cfg)
