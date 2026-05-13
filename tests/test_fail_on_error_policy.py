from __future__ import annotations

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from sd_optim.core.optimizer_cache import fail_on_error_enabled
from sd_optim.optimizers.optuna.objective import run_objective


def _objective_cfg(**overrides: object):
    data = {"optimizer": {"init_points": 0}}
    data.update(overrides)
    return OmegaConf.create(data)


class _DummyTrial:
    number = 0

    def __init__(self) -> None:
        self.user_attrs: dict[str, object] = {}

    def suggest_float(self, name: str, low: float, high: float, step=None, log: bool = False) -> float:  # noqa: ARG002
        return low

    def set_user_attr(self, key: str, value: object) -> None:
        self.user_attrs[key] = value


def test_fail_on_error_enabled_defaults_to_true() -> None:
    assert fail_on_error_enabled(OmegaConf.create({})) is True


def test_fail_on_error_enabled_respects_explicit_false() -> None:
    assert fail_on_error_enabled(OmegaConf.create({"fail_on_error": False})) is False


def test_optuna_objective_raises_by_default_on_runtime_error() -> None:
    optimizer = SimpleNamespace()
    optimizer.cfg = _objective_cfg()
    optimizer.optimizer_pbounds = {"alpha": (0.0, 1.0)}
    optimizer.child_to_parent = {}
    optimizer.last_trial_scorer_summary = {}
    optimizer.scorer = SimpleNamespace(last_scorer_results={})
    optimizer.trial_scores = []
    optimizer.early_stopping = False
    optimizer.patience = 1
    optimizer.min_improvement = 0.0

    async def boom(params):  # noqa: ARG001
        raise RuntimeError("boom")

    optimizer.sd_target_function = boom

    with pytest.raises(RuntimeError, match="boom"):
        run_objective(optimizer, _DummyTrial())


def test_optuna_objective_can_continue_when_fail_on_error_disabled() -> None:
    optimizer = SimpleNamespace()
    optimizer.cfg = _objective_cfg(fail_on_error=False)
    optimizer.optimizer_pbounds = {"alpha": (0.0, 1.0)}
    optimizer.child_to_parent = {}
    optimizer.last_trial_scorer_summary = {}
    optimizer.scorer = SimpleNamespace(last_scorer_results={})
    optimizer.trial_scores = []
    optimizer.early_stopping = False
    optimizer.patience = 1
    optimizer.min_improvement = 0.0

    async def boom(params):  # noqa: ARG001
        raise RuntimeError("boom")

    optimizer.sd_target_function = boom

    assert run_objective(optimizer, _DummyTrial()) == float("-inf")
