from __future__ import annotations

from pathlib import Path

import sd_mecha

from sd_optim.merger import Merger


def test_slice_models_does_not_treat_weighted_sum_alpha_as_model_input() -> None:
    merger = Merger.__new__(Merger)

    prepared_model_nodes = ["model-a", "model-b", "model-c"]

    sliced = merger._slice_models(prepared_model_nodes, sd_mecha.weighted_sum)

    assert sliced == ["model-a", "model-b"]


def test_prepare_model_recipe_args_skips_unused_models_for_fixed_arity_methods() -> None:
    merger = Merger.__new__(Merger)

    checked_paths: list[Path] = []

    def fake_get_adapter_candidate_ids(node):
        checked_paths.append(node.path)
        if node.path == Path("model-c.safetensors"):
            raise AssertionError("unused extra model should not be preprocessed")
        return ()

    merger._get_adapter_candidate_ids = fake_get_adapter_candidate_ids

    model_nodes = [
        sd_mecha.model("model-a.safetensors"),
        sd_mecha.model("model-b.safetensors"),
        sd_mecha.model("model-c.safetensors"),
    ]

    prepared = merger._prepare_model_recipe_args(model_nodes, None, sd_mecha.weighted_sum)

    assert [node.path for node in prepared] == [
        Path("model-a.safetensors"),
        Path("model-b.safetensors"),
    ]
    assert checked_paths == [
        Path("model-a.safetensors"),
        Path("model-b.safetensors"),
    ]
