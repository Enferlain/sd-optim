from __future__ import annotations

from types import SimpleNamespace

import optuna
import pytest
from omegaconf import OmegaConf

from sd_optim.optimizers.optuna.objective import evaluate_condition, run_objective, suggest_parameters


class _DummyTrial:
    number = 3

    def __init__(self) -> None:
        self.user_attrs: dict[str, object] = {}
        self.calls: list[tuple] = []

    def suggest_int(self, name: str, low: int, high: int, step: int = 1, log: bool = False) -> int:
        self.calls.append(("int", name, low, high, step, log))
        return low

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        step: float | None = None,
        log: bool = False,
    ) -> float:
        self.calls.append(("float", name, low, high, step, log))
        return low

    def suggest_categorical(self, name: str, choices: list[object]) -> object:
        self.calls.append(("categorical", name, tuple(choices)))
        return choices[0]

    def set_user_attr(self, key: str, value: object) -> None:
        self.user_attrs[key] = value


def test_evaluate_condition_handles_supported_and_invalid_inputs(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("WARNING")

    assert evaluate_condition(0.5, ">= 0.4") is True
    assert evaluate_condition(0.5, "< 0.4") is False
    assert evaluate_condition(0.5, "??? 0.4") is True
    assert evaluate_condition("bad", ">= 0.4") is True
    assert "Error evaluating condition" in caplog.text


def test_suggest_parameters_supports_mixed_bounds_and_dependency_defaults() -> None:
    optimizer = SimpleNamespace(
        optimizer_pbounds={
            "parent": {"range": (0, 1), "step": 1},
            "child": [10, 20],
            "float_range": {"range": (0.2, 0.8), "step": 0.1, "log": False},
            "tuple_int": (1, 3),
            "tuple_float": (0.5, 1.5),
            "fixed": 7,
        },
        child_to_parent={"child": {"parent": "parent", "condition": "> 0", "default": 99}},
    )
    trial = _DummyTrial()

    params = suggest_parameters(optimizer, trial)

    assert params == {
        "parent": 0,
        "child": 99,
        "float_range": 0.2,
        "tuple_int": 1,
        "tuple_float": 0.5,
        "fixed": 7,
    }
    assert ("categorical", "child", (10, 20)) not in trial.calls
    assert ("int", "parent", 0, 1, 1, False) in trial.calls
    assert ("float", "float_range", 0.2, 0.8, 0.1, False) in trial.calls


def test_suggest_parameters_rejects_unsupported_bounds_format() -> None:
    optimizer = SimpleNamespace(
        optimizer_pbounds={"broken": {"oops": True}},
        child_to_parent={},
    )

    with pytest.raises(ValueError, match="Unsupported bounds format"):
        suggest_parameters(optimizer, _DummyTrial())


def test_suggest_parameters_without_bounds_prunes_trial() -> None:
    optimizer = SimpleNamespace(optimizer_pbounds={}, child_to_parent={})

    with pytest.raises(optuna.exceptions.TrialPruned, match="Bounds not available"):
        suggest_parameters(optimizer, _DummyTrial())


def test_run_objective_falls_back_to_last_scorer_results() -> None:
    optimizer = SimpleNamespace(
        cfg=OmegaConf.create({}),
        optimizer_pbounds={"alpha": 0.5},
        child_to_parent={},
        scorer=SimpleNamespace(last_scorer_results={"manual": 0.7}),
        last_trial_scorer_summary={},
        trial_scores=[],
        early_stopping=False,
        patience=2,
        min_improvement=0.0,
    )

    async def target(params):  # noqa: ARG001
        return 0.7

    optimizer.sd_target_function = target
    trial = _DummyTrial()

    assert run_objective(optimizer, trial) == 0.7
    assert trial.user_attrs["scorer_results"] == {"manual": 0.7}
    assert optimizer.trial_scores == [0.7]


def test_run_objective_logs_iteration_start_before_parameter_suggestion(monkeypatch) -> None:
    objective_mod = __import__("sd_optim.optimizers.optuna.objective", fromlist=["run_objective"])
    banner_calls: list[tuple[object, int]] = []

    monkeypatch.setattr(
        objective_mod,
        "log_iteration_start",
        lambda optimizer, params, *, effective_iteration: banner_calls.append((params, effective_iteration)),
    )

    optimizer = SimpleNamespace(
        cfg=OmegaConf.create({"optimizer": {"init_points": 2}}),
        iteration=0,
        completed_trials=0,
        optimizer_pbounds={"alpha": 0.5},
        child_to_parent={},
        scorer=SimpleNamespace(last_scorer_results={"manual": 0.7}),
        last_trial_scorer_summary={},
        trial_scores=[],
        early_stopping=False,
        patience=2,
        min_improvement=0.0,
    )

    async def target(params):  # noqa: ARG001
        return 0.7

    optimizer.sd_target_function = target

    assert run_objective(optimizer, _DummyTrial()) == 0.7
    assert banner_calls == [(None, 1)]
    assert optimizer._iteration_start_logged is True


def test_run_objective_prunes_when_early_stopping_threshold_is_hit() -> None:
    optimizer = SimpleNamespace(
        cfg=OmegaConf.create({}),
        optimizer_pbounds={"alpha": 0.5},
        child_to_parent={},
        scorer=SimpleNamespace(last_scorer_results={}),
        last_trial_scorer_summary={"aggregate": {}, "payloads": []},
        trial_scores=[0.8],
        early_stopping=True,
        patience=1,
        min_improvement=0.1,
        no_improvement_count=0,
    )

    async def target(params):  # noqa: ARG001
        return 0.81

    optimizer.sd_target_function = target

    with pytest.raises(optuna.exceptions.TrialPruned):
        run_objective(optimizer, _DummyTrial())
