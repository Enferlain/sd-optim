from __future__ import annotations

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf


def test_runtime_average_calc_uses_default_weights_when_lengths_mismatch() -> None:
    from sd_optim.scoring import runtime as runtime_mod

    result = runtime_mod.average_calc([2.0, 4.0], [1.0], "arithmetic")

    assert result == 3.0


def test_runtime_setup_img_saving_enables_manual_mode_without_hydra(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from sd_optim.scoring import runtime as runtime_mod

    cfg = OmegaConf.create(
        {
            "scorer_method": ["manual"],
            "save_imgs": False,
        }
    )
    scorer = SimpleNamespace(cfg=cfg, imgs_dir=None)

    monkeypatch.setattr(runtime_mod.HydraConfig, "get", lambda: (_ for _ in ()).throw(ValueError("no hydra")))
    monkeypatch.chdir(tmp_path)

    runtime_mod.setup_img_saving(scorer)

    assert scorer.imgs_dir == (tmp_path / "imgs_fallback").resolve()
    assert scorer.cfg.save_imgs is True
    assert scorer.imgs_dir.is_dir()


def test_runtime_ensure_rembg_session_requires_dependency() -> None:
    from sd_optim.scoring import runtime as runtime_mod

    scorer = SimpleNamespace(rembg_session=None, _rembg_required=True)

    with pytest.raises(ImportError, match="requires 'rembg'"):
        runtime_mod.ensure_rembg_session(scorer, session_factory=None)
