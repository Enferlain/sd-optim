from __future__ import annotations

from types import SimpleNamespace

import pytest
import sd_mecha
import torch
from omegaconf import OmegaConf


def test_sanitize_recipe_text_for_deserialize_drops_blank_lines() -> None:
    from sd_optim.merge.recipe_optimization import sanitize_recipe_text_for_deserialize

    recipe_text = "version 0.1.0\n\nmodel \"a.safetensors\"\n   \nmerge \"weighted_sum\" &0 alpha=0.5\n"

    assert sanitize_recipe_text_for_deserialize(recipe_text) == (
        "version 0.1.0\nmodel \"a.safetensors\"\nmerge \"weighted_sum\" &0 alpha=0.5"
    )


def test_resolve_layer_adjust_model_path_falls_back_to_models_dir(tmp_path) -> None:
    from sd_optim.merge.layer_adjust import resolve_layer_adjust_model_path

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model_path = models_dir / "base.safetensors"
    model_path.write_text("fake", encoding="utf-8")

    resolved = resolve_layer_adjust_model_path(
        models_dir=models_dir,
        model_path_str="base.safetensors",
    )

    assert resolved == model_path


def test_detect_sdxl_model_looks_for_conditioner_embedder_key() -> None:
    from sd_optim.merge.layer_adjust import detect_is_xl_model

    assert detect_is_xl_model({"conditioner.embedders.1.model.weight": torch.tensor(1.0)}) is True
    assert detect_is_xl_model({"model.diffusion_model.out.0.weight": torch.tensor(1.0)}) is False


def test_recipe_optimization_rejects_non_merge_target(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from sd_optim.merge import recipe_optimization as recipe_mod

    recipe_path = tmp_path / "test.mecha"
    recipe_path.write_text(
        "\n".join(
            [
                "version 0.1.0",
                'model "a.safetensors"',
            ]
        ),
        encoding="utf-8",
    )

    merger = SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "recipe_optimization": {
                    "recipe_path": str(recipe_path),
                    "target_nodes": "&0",
                    "target_params": ["alpha"],
                }
            }
        )
    )

    monkeypatch.setattr(recipe_mod.sd_mecha, "deserialize", lambda lines: sd_mecha.model("a.safetensors"))

    with pytest.raises(TypeError, match="is not a merge node"):
        recipe_mod.load_validated_target_node(merger, recipe_path.read_text(encoding="utf-8"))
