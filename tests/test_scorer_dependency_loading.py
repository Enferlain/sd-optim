from __future__ import annotations

import pytest
from omegaconf import OmegaConf

import sd_optim.scorer as scorer_mod


def test_get_scorer_class_returns_known_class():
    resolved = scorer_mod._get_scorer_class("cityaes")
    assert resolved is not None
    assert resolved.__name__ == "CityAestheticsScorer"


def test_get_scorer_class_alias_resolves_same_class():
    assert scorer_mod._get_scorer_class("laion") is scorer_mod._get_scorer_class("chad")


def test_get_scorer_class_returns_none_for_unknown():
    assert scorer_mod._get_scorer_class("this_is_not_a_scorer") is None


def test_rembg_required_scorer_raises_clear_error_without_rembg(monkeypatch):
    monkeypatch.setattr(scorer_mod, "new_session", lambda **_: (_ for _ in ()).throw(ImportError("no rembg")))
    monkeypatch.setattr(scorer_mod.AestheticScorer, "setup_evaluator_paths", lambda self: None)
    monkeypatch.setattr(scorer_mod.AestheticScorer, "get_models", lambda self: None)
    monkeypatch.setattr(scorer_mod.AestheticScorer, "_load_all_models", lambda self: None)
    cfg = OmegaConf.create({"scorer_method": ["textureclean"], "scorer_weight": {}, "scorer_device": {}})
    with pytest.raises(ImportError, match="requires 'rembg'"):
        scorer_mod.AestheticScorer(cfg)
