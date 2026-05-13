from __future__ import annotations

import importlib
import sys

import pytest
from omegaconf import OmegaConf


def _reload_scorer_module():
    for module_name in [
        "sd_optim.scorer",
        "sd_optim.scoring.registry",
        "sd_optim.extensions.bundled.scorers.models.Laion",
        "sd_optim.extensions.bundled.scorers.models.CityAesthetics",
    ]:
        sys.modules.pop(module_name, None)

    return importlib.import_module("sd_optim.scorer")


def test_get_scorer_class_returns_known_class():
    scorer_mod = _reload_scorer_module()
    resolved = scorer_mod.get_scorer_class("cityaes")
    assert resolved is not None
    assert resolved.__name__ == "CityAestheticsScorer"


def test_get_scorer_class_alias_resolves_same_class():
    scorer_mod = _reload_scorer_module()
    assert scorer_mod.get_scorer_class("laion") is scorer_mod.get_scorer_class("chad")


def test_get_scorer_class_returns_none_for_unknown():
    scorer_mod = _reload_scorer_module()
    assert scorer_mod.get_scorer_class("this_is_not_a_scorer") is None


def test_rembg_required_scorer_raises_clear_error_without_rembg(monkeypatch):
    scorer_mod = _reload_scorer_module()
    monkeypatch.setattr(scorer_mod, "new_session", lambda **_: (_ for _ in ()).throw(ImportError("no rembg")))
    monkeypatch.setattr(scorer_mod, "setup_evaluator_paths", lambda self: None)
    monkeypatch.setattr(scorer_mod, "get_models", lambda self: None)
    monkeypatch.setattr(scorer_mod, "load_all_models", lambda self: None)
    cfg = OmegaConf.create(
        {
            "generation": {"save_imgs": False},
            "scoring": {"scorer_method": ["textureclean"], "scorer_weight": {}, "scorer_device": {}},
        }
    )
    with pytest.raises(ImportError, match="requires 'rembg'"):
        scorer_mod.Scorer(cfg)


def test_scorer_module_import_does_not_eagerly_import_all_builtin_scorers():
    scorer_mod = _reload_scorer_module()

    assert "sd_optim.extensions.bundled.scorers.models.Laion" not in sys.modules

    resolved = scorer_mod.get_scorer_class("cityaes")

    assert resolved is not None
    assert "sd_optim.extensions.bundled.scorers.models.CityAesthetics" in sys.modules
    assert "sd_optim.extensions.bundled.scorers.models.Laion" not in sys.modules


def test_scorer_registry_points_at_models_subpackage():
    registry_mod = importlib.import_module("sd_optim.scoring.catalog")

    assert registry_mod.SCORER_CLASS_PATHS["cityaes"][0].startswith(
        "sd_optim.extensions.bundled.scorers.models."
    )
