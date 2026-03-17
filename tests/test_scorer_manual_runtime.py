from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from PIL import Image


@pytest.mark.asyncio
async def test_manual_score_uses_saved_preview_and_records_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    import sd_optim.scoring.runtime as scorer_runtime_mod
    from sd_optim.scorer import AestheticScorer

    monkeypatch.setattr(
        scorer_runtime_mod.HydraConfig,
        "get",
        lambda: SimpleNamespace(runtime=SimpleNamespace(output_dir=str(tmp_path))),
    )

    cfg = OmegaConf.create(
        {
            "scorer_method": ["manual"],
            "scorer_model_dir": str(tmp_path / "models"),
            "scorer_default_device": "cpu",
            "save_imgs": False,
            "scorer_print_individual": False,
            "scorer_average_type": "arithmetic",
            "scorer_weight": {},
            "scorer_device": {},
            "scorer_lazy_load_list": [],
        }
    )

    scorer = AestheticScorer(cfg)
    opened_paths = []

    monkeypatch.setattr(Image.Image, "show", lambda self: None)
    monkeypatch.setattr(scorer, "open_image", lambda path: opened_paths.append(path))
    monkeypatch.setattr(scorer, "get_user_score", lambda: 6.25)

    score = await scorer.score(Image.new("RGB", (8, 8), "red"), prompt="test", name="payload-one")

    assert score == 6.25
    assert scorer.last_scorer_results == {"manual": 6.25}
    deadline = time.time() + 1.0
    while not opened_paths and time.time() < deadline:
        time.sleep(0.01)

    assert len(opened_paths) == 1

    preview_path = opened_paths[0]
    assert preview_path.exists()
    assert preview_path.parent == tmp_path / "imgs"
    assert preview_path.name.startswith("manual-")
    assert "payload-one" in preview_path.name
    assert preview_path.suffix == ".png"
