from __future__ import annotations

from pathlib import Path

import sd_mecha

from omegaconf import OmegaConf

from sd_optim.merger import Merger
from sd_optim.merge.model_nodes import create_model_nodes


def test_create_model_nodes_keeps_paths_relative_to_models_dir(tmp_path) -> None:
    merger = Merger.__new__(Merger)
    merger.cfg = OmegaConf.create({"model_paths": ["subdir/model-a.safetensors"]})
    merger.models_dir = tmp_path

    model_file = tmp_path / "subdir" / "model-a.safetensors"
    model_file.parent.mkdir(parents=True)
    model_file.write_text("fake")

    nodes = create_model_nodes(merger)

    assert len(nodes) == 1
    assert nodes[0].path == Path("subdir/model-a.safetensors")


def test_create_model_nodes_relativizes_absolute_paths_within_models_dir(tmp_path) -> None:
    merger = Merger.__new__(Merger)
    model_file = tmp_path / "nested" / "model-b.safetensors"
    model_file.parent.mkdir(parents=True)
    model_file.write_text("fake")

    merger.cfg = OmegaConf.create({"model_paths": [str(model_file)]})
    merger.models_dir = tmp_path

    nodes = create_model_nodes(merger)

    assert len(nodes) == 1
    assert nodes[0].path == Path("nested/model-b.safetensors")
    assert isinstance(nodes[0], sd_mecha.recipe_nodes.ModelRecipeNode)
