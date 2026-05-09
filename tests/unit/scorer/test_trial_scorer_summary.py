from __future__ import annotations

import importlib

import pytest

from sd_optim.core.trial_scorer_summary import build_trial_scorer_summary


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    return sum(v * w for v, w in zip(values, weights)) / sum(weights)


def test_build_trial_scorer_summary_aggregates_across_payloads() -> None:
    payload_entries = [
        {
            "name": "a",
            "weight": 1.0,
            "scores": {"cityaes": 2.0, "textureclean": 4.0},
            "combined": 3.0,
        },
        {
            "name": "b",
            "weight": 3.0,
            "scores": {"cityaes": 6.0},
            "combined": 5.0,
        },
    ]

    summary = build_trial_scorer_summary(payload_entries, final_score=4.5, combine_scores=_weighted_mean)

    assert summary["aggregate"]["cityaes"] == 5.0
    assert summary["aggregate"]["textureclean"] == 4.0
    assert summary["aggregate"]["combined"] == 4.5


def test_build_trial_scorer_summary_preserves_duplicate_payload_names() -> None:
    payload_entries = [
        {"name": "same_name", "weight": 1.0, "scores": {"cityaes": 1.0}, "combined": 1.0},
        {"name": "same_name", "weight": 1.0, "scores": {"cityaes": 3.0}, "combined": 3.0},
    ]

    summary = build_trial_scorer_summary(payload_entries, final_score=2.0, combine_scores=_weighted_mean)

    assert len(summary["payloads"]) == 2
    assert summary["payloads"][0]["name"] == "same_name"
    assert summary["payloads"][1]["name"] == "same_name"
    assert summary["aggregate"]["cityaes"] == 2.0


def test_top_level_trial_scorer_summary_module_is_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("sd_optim.trial_scorer_summary")
