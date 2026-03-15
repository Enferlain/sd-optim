from __future__ import annotations

import sd_mecha

from sd_optim.merger import Merger


def test_slice_models_does_not_treat_weighted_sum_alpha_as_model_input() -> None:
    merger = Merger.__new__(Merger)

    prepared_model_nodes = ["model-a", "model-b", "model-c"]

    sliced = merger._slice_models(prepared_model_nodes, sd_mecha.weighted_sum)

    assert sliced == ["model-a", "model-b"]
