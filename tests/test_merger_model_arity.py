from __future__ import annotations

from pathlib import Path

import sd_mecha

from sd_optim.merger import Merger
from sd_optim.merge.recipe_builder import prepare_model_recipe_args, slice_models


def test_slice_models_does_not_treat_weighted_sum_alpha_as_model_input() -> None:
    merger = Merger.__new__(Merger)

    prepared_model_nodes = ["model-a", "model-b", "model-c"]

    sliced = slice_models(merger, prepared_model_nodes, sd_mecha.weighted_sum)

    assert sliced == ["model-a", "model-b"]


def test_prepare_model_recipe_args_skips_unused_models_for_fixed_arity_methods() -> None:
    merger = Merger.__new__(Merger)

    checked_paths: list[Path] = []

    def fake_get_adapter_candidate_ids(node):
        checked_paths.append(node.path)
        if node.path == Path("model-c.safetensors"):
            raise AssertionError("unused extra model should not be preprocessed")
        return ()

    import sd_optim.merge.recipe_builder as recipe_builder_mod

    original_get_adapter_candidate_ids = recipe_builder_mod.get_adapter_candidate_ids
    recipe_builder_mod.get_adapter_candidate_ids = lambda merger_arg, node: fake_get_adapter_candidate_ids(node)  # noqa: ARG005

    model_nodes = [
        sd_mecha.model("model-a.safetensors"),
        sd_mecha.model("model-b.safetensors"),
        sd_mecha.model("model-c.safetensors"),
    ]

    try:
        prepared = prepare_model_recipe_args(merger, model_nodes, None, sd_mecha.weighted_sum)
    finally:
        recipe_builder_mod.get_adapter_candidate_ids = original_get_adapter_candidate_ids

    assert [node.path for node in prepared] == [
        Path("model-a.safetensors"),
        Path("model-b.safetensors"),
    ]
    assert checked_paths == [
        Path("model-a.safetensors"),
        Path("model-b.safetensors"),
    ]
