from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf


def test_assets_helper_sets_paths_and_defaults() -> None:
    from sd_optim.scoring import assets as assets_mod

    scorer = SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "paths": {"scorer_model_dir": "C:/models"},
                "scoring": {
                    "scorer_method": ["cityaes"],
                    "scorer_device": {},
                    "scorer_weight": {},
                    "scorer_default_device": "cpu",
                    "scorer_alt_location": {},
                },
            }
        ),
        model_path={},
    )

    assets_mod.setup_evaluator_paths(scorer)

    assert scorer.model_path["cityaes"] == Path("C:/models") / "CityAesthetics-Anime-v1.8.safetensors"
    assert scorer.cfg.scoring.scorer_device["cityaes"] == "cpu"
    assert scorer.cfg.scoring.scorer_weight["cityaes"] == 1.0


def test_loading_helper_builds_factory_entries() -> None:
    from sd_optim.scoring import loading as loading_mod

    factory = loading_mod.build_scorer_factory(
        Path("clip-l.pt"),
        Path("clip-b.safetensors"),
    )

    assert factory["laion"]["extra_args"]["clip_model_path"] == "clip-l.pt"
    assert factory["textureclean"]["extra_args"]["rembg_session"] == "self.rembg_session"
