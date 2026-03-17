from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import optuna
import pytest
from omegaconf import OmegaConf

from sd_optim.optimizers.optuna import study_manager


def _make_cfg(
    *,
    scorer_method: list[str] | None = None,
    optuna_overrides: dict | None = None,
    optimization_mode: str = "merge",
) -> object:
    optuna_config = {
        "sampler": {"type": "tpe"},
        "use_pruning": False,
        "pruner_type": "median",
        "storage_dir": "./tmp_optuna_db",
        "resume_from_study": None,
        "fork_study": False,
        "n_jobs": 1,
    }
    if optuna_overrides:
        optuna_config.update(optuna_overrides)
    return OmegaConf.create(
        {
            "optimization_mode": optimization_mode,
            "merge_method": "weighted_sum",
            "recipe_optimization": {"recipe_path": "recipes/demo_recipe.yaml"},
            "model_paths": ["a.safetensors", "b.safetensors"],
            "base_model_index": 1,
            "scorer_method": scorer_method or ["manual"],
            "optimization_guide": {"dependencies": []},
            "optimizer": {
                "init_points": 2,
                "n_iters": 3,
                "optuna_config": optuna_config,
            },
        }
    )


class _DummyStudy:
    def __init__(self, *, trials: list | None = None, user_attrs: dict | None = None) -> None:
        self.trials = trials or []
        self.user_attrs = user_attrs or {}
        self.attrs_set: dict[str, object] = {}
        self.enqueued: list[dict] = []
        self.best_value = 0.9
        self.best_trial = object() if self.trials else None
        self.optimize_calls: list[dict] = []

    def set_user_attr(self, key: str, value: object) -> None:
        self.attrs_set[key] = value

    def add_trial(self, trial: object) -> None:
        self.trials.append(trial)

    def enqueue_trial(self, params: dict) -> None:
        self.enqueued.append(params)

    def optimize(self, func, n_trials: int, n_jobs: int, show_progress_bar: bool, callbacks: list) -> None:
        self.optimize_calls.append(
            {
                "func": func,
                "n_trials": n_trials,
                "n_jobs": n_jobs,
                "show_progress_bar": show_progress_bar,
                "callbacks": callbacks,
            }
        )


class _DummyTrialLogger:
    def __init__(self, data: list[dict] | None = None) -> None:
        self.data = data or []
        self.path: Path | None = None

    def set_path(self, path: Path) -> None:
        self.path = path

    def load_trials(self) -> list[dict]:
        return list(self.data)


def test_initialize_optuna_state_wires_storage_logger_and_dependencies(monkeypatch, tmp_path: Path) -> None:
    cfg = _make_cfg(optuna_overrides={"storage_dir": str(tmp_path)})
    logger_instance = _DummyTrialLogger()
    monkeypatch.setattr(study_manager, "TrialLogger", lambda: logger_instance)
    monkeypatch.setattr(study_manager, "initialize_storage_dir", lambda cfg: tmp_path)  # noqa: ARG005

    optimizer = SimpleNamespace(
        cfg=cfg,
        param_info={"alpha": {"bounds": (0.0, 1.0)}},
        bounds_initializer=SimpleNamespace(validate_dependencies=lambda param_info, deps: {"beta": {"parent": "alpha", "condition": "> 0", "default": 0.0}}),
    )

    study_manager.initialize_optuna_state(optimizer)

    assert optimizer.study is None
    assert optimizer.optuna_storage_dir == tmp_path
    assert optimizer.logger is logger_instance
    assert optimizer.child_to_parent["beta"]["parent"] == "alpha"


def test_initialize_storage_dir_and_storage_uri_fallback(monkeypatch, tmp_path: Path) -> None:
    cfg = _make_cfg(optuna_overrides={"storage_dir": str(tmp_path / "nested" / "dbs")})

    storage_dir = study_manager.initialize_storage_dir(cfg)
    assert storage_dir.is_dir()

    monkeypatch.setattr(study_manager, "flatten_scorers", lambda scorers: (_ for _ in ()).throw(RuntimeError("boom")))  # noqa: ARG005
    storage_uri, filename = study_manager.build_storage_uri_for_new_study(cfg, tmp_path)

    assert filename == "optuna_fallback.db"
    assert storage_uri.endswith(filename)


def test_flatten_scorers_and_storage_uri_handle_nested_and_recipe_modes(tmp_path: Path) -> None:
    merge_cfg = _make_cfg(scorer_method=["manual", ["clip"]])
    recipe_cfg = _make_cfg(optimization_mode="recipe", scorer_method=["manual"])

    assert study_manager.flatten_scorers(merge_cfg.scorer_method) == ["manual", "clip"]

    merge_uri, merge_filename = study_manager.build_storage_uri_for_new_study(merge_cfg, tmp_path)
    recipe_uri, recipe_filename = study_manager.build_storage_uri_for_new_study(recipe_cfg, tmp_path, is_fork=True)

    assert merge_filename == "optuna_merge_weighted_sum_clip_manual.db"
    assert merge_uri.endswith(merge_filename)
    assert recipe_filename == "optuna_fork_recipe_demo_recipe_manual.db"
    assert recipe_uri.endswith(recipe_filename)


def test_configure_pruner_handles_supported_and_unknown_values(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("WARNING")

    assert study_manager.configure_pruner(_make_cfg(optuna_overrides={"use_pruning": False})) is None
    assert isinstance(study_manager.configure_pruner(_make_cfg(optuna_overrides={"use_pruning": True, "pruner_type": "median"})), optuna.pruners.MedianPruner)
    assert isinstance(
        study_manager.configure_pruner(_make_cfg(optuna_overrides={"use_pruning": True, "pruner_type": "successive_halving"})),
        optuna.pruners.SuccessiveHalvingPruner,
    )
    assert study_manager.configure_pruner(_make_cfg(optuna_overrides={"use_pruning": True, "pruner_type": "mystery"})) is None
    assert "Unknown pruner_type" in caplog.text


def test_find_db_for_study_and_set_initial_attributes(monkeypatch, tmp_path: Path) -> None:
    first_db = tmp_path / "first.db"
    second_db = tmp_path / "second.db"
    first_db.write_text("", encoding="utf-8")
    second_db.write_text("", encoding="utf-8")

    def fake_summaries(storage: str):
        if storage.endswith("second.db"):
            return [SimpleNamespace(study_name="target")]
        return [SimpleNamespace(study_name="other")]

    monkeypatch.setattr(study_manager.optuna.study, "get_all_study_summaries", fake_summaries)

    storage_uri, db_name = study_manager.find_db_for_study(tmp_path, "target")
    assert db_name == "second.db"
    assert storage_uri is not None and storage_uri.endswith("second.db")

    study = _DummyStudy()
    study_manager.set_initial_study_attributes(study, _make_cfg(), "run_1", parent_name="parent_run")

    assert study.attrs_set["config_merge_method"] == "weighted_sum"
    assert study.attrs_set["forked_from"] == "parent_run"
    assert study.attrs_set["config_scorers"] == ["manual"]


def test_find_db_for_study_handles_scan_errors_and_missing_target(
    monkeypatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("WARNING")
    broken_db = tmp_path / "broken.db"
    broken_db.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        study_manager.optuna.study,
        "get_all_study_summaries",
        lambda storage: (_ for _ in ()).throw(RuntimeError("db boom")),
    )

    assert study_manager.find_db_for_study(tmp_path, "missing") == (None, None)
    assert "Could not inspect database" in caplog.text
    assert "was not found in any database" in caplog.text


def test_set_initial_study_attributes_warns_on_write_error(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("WARNING")
    study = SimpleNamespace(set_user_attr=lambda key, value: (_ for _ in ()).throw(RuntimeError("write failed")))  # noqa: ARG005

    study_manager.set_initial_study_attributes(study, _make_cfg(), "run_warn")

    assert "Could not store config as study attributes" in caplog.text


def test_setup_trial_logger_path_and_restore_trials(monkeypatch, tmp_path: Path) -> None:
    logger_instance = _DummyTrialLogger(
        [
            {"params": {"alpha": 0.1}, "target": 0.7},
            {"params": {"alpha": 0.2}, "target": None},
            {"target": 0.3},
        ]
    )
    monkeypatch.setattr(study_manager.HydraConfig, "get", lambda: SimpleNamespace(runtime=SimpleNamespace(output_dir=str(tmp_path))))
    monkeypatch.setattr(
        study_manager.optuna.trial,
        "create_trial",
        lambda params, value, state: {"params": params, "value": value, "state": state},
    )
    study = _DummyStudy()

    study_manager.setup_trial_logger_path(logger_instance, "run_abc")
    study_manager.restore_trials_from_log(study, logger_instance)

    assert logger_instance.path == tmp_path / "run_abc_trials.jsonl"
    assert study.trials == [{"params": {"alpha": 0.1}, "value": 0.7, "state": optuna.trial.TrialState.COMPLETE}]


def test_setup_trial_logger_path_and_restore_trials_warn_on_missing_state(
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("WARNING")
    logger_instance = _DummyTrialLogger([])
    study = _DummyStudy()
    monkeypatch.setattr(study_manager.HydraConfig, "get", lambda: (_ for _ in ()).throw(ValueError("no hydra")))

    study_manager.setup_trial_logger_path(logger_instance, "run_missing")
    study_manager.restore_trials_from_log(study, logger_instance)

    assert logger_instance.path is None
    assert study.trials == []
    assert "Failed to set trial logger path" in caplog.text
    assert "No trials found in log file" in caplog.text


def test_optimize_study_creates_new_study_and_skips_when_no_trials_remaining(monkeypatch, tmp_path: Path) -> None:
    cfg = _make_cfg(optuna_overrides={"storage_dir": str(tmp_path)}, scorer_method=["manual"])
    cfg.optimizer.init_points = 0
    cfg.optimizer.n_iters = 0
    created_study = _DummyStudy()
    optimizer = SimpleNamespace(
        cfg=cfg,
        optimizer_pbounds={"alpha": (0.0, 1.0)},
        optuna_storage_dir=tmp_path,
        logger=_DummyTrialLogger(),
        best_rolling_score=0.0,
        completed_trials=0,
    )
    recorded = {"initial_attrs": 0, "analyzed": 0}

    monkeypatch.setattr(study_manager, "configure_sampler", lambda cfg: "sampler")  # noqa: ARG005
    monkeypatch.setattr(study_manager, "configure_pruner", lambda cfg: None)  # noqa: ARG005
    monkeypatch.setattr(study_manager, "build_storage_uri_for_new_study", lambda cfg, storage_dir, is_fork=False: ("sqlite:///new.db", "new.db"))  # noqa: ARG005
    monkeypatch.setattr(study_manager.optuna, "create_study", lambda **kwargs: created_study)
    monkeypatch.setattr(study_manager, "set_initial_study_attributes", lambda *args, **kwargs: recorded.__setitem__("initial_attrs", recorded["initial_attrs"] + 1))
    monkeypatch.setattr(study_manager, "analyze_parameter_importance", lambda optimizer: recorded.__setitem__("analyzed", recorded["analyzed"] + 1))
    monkeypatch.setattr(study_manager.time, "strftime", lambda fmt: "20260317_120000")  # noqa: ARG005

    asyncio.run(study_manager.optimize_study(optimizer))

    assert optimizer.study is created_study
    assert optimizer.study_name == "run_20260317_120000_tpe"
    assert recorded == {"initial_attrs": 1, "analyzed": 1}
    assert created_study.optimize_calls == []


def test_optimize_study_forks_completed_trials(monkeypatch, tmp_path: Path) -> None:
    cfg = _make_cfg(optuna_overrides={"storage_dir": str(tmp_path), "resume_from_study": "parent", "fork_study": True})
    cfg.optimizer.init_points = 0
    cfg.optimizer.n_iters = 0
    parent_trials = [
        SimpleNamespace(state=optuna.trial.TrialState.COMPLETE, params={"alpha": 0.1}),
        SimpleNamespace(state=optuna.trial.TrialState.FAIL, params={"alpha": 0.2}),
    ]
    parent_study = _DummyStudy(trials=parent_trials, user_attrs={"config_scorers": ["manual"]})
    forked_study = _DummyStudy()
    optimizer = SimpleNamespace(
        cfg=cfg,
        optimizer_pbounds={"alpha": (0.0, 1.0)},
        optuna_storage_dir=tmp_path,
        logger=_DummyTrialLogger(),
        best_rolling_score=0.0,
        completed_trials=0,
    )

    monkeypatch.setattr(study_manager, "configure_sampler", lambda cfg: "sampler")  # noqa: ARG005
    monkeypatch.setattr(study_manager, "configure_pruner", lambda cfg: None)  # noqa: ARG005
    monkeypatch.setattr(study_manager, "find_db_for_study", lambda storage_dir, name: ("sqlite:///parent.db", "parent.db"))  # noqa: ARG005
    monkeypatch.setattr(study_manager.optuna, "load_study", lambda study_name, storage: parent_study)  # noqa: ARG005
    monkeypatch.setattr(
        study_manager,
        "build_storage_uri_for_new_study",
        lambda cfg, storage_dir, is_fork=False: ("sqlite:///fork.db", "fork.db" if is_fork else "new.db"),  # noqa: ARG005
    )
    monkeypatch.setattr(study_manager.optuna, "create_study", lambda **kwargs: forked_study)
    monkeypatch.setattr(study_manager, "set_initial_study_attributes", lambda *args, **kwargs: None)
    monkeypatch.setattr(study_manager, "analyze_parameter_importance", lambda optimizer: None)  # noqa: ARG005
    monkeypatch.setattr(study_manager.time, "strftime", lambda fmt: "20260317_120500")  # noqa: ARG005

    asyncio.run(study_manager.optimize_study(optimizer))

    assert optimizer.study is forked_study
    assert forked_study.enqueued == [{"alpha": 0.1}]
    assert optimizer.study_name == "fork_parent_20260317_120500"


def test_optimize_study_rejects_resume_when_scorers_change(monkeypatch, tmp_path: Path) -> None:
    cfg = _make_cfg(
        optuna_overrides={"storage_dir": str(tmp_path), "resume_from_study": "parent", "fork_study": False},
        scorer_method=["manual", "clip"],
    )
    parent_study = _DummyStudy(trials=[], user_attrs={"config_scorers": ["manual"]})
    optimizer = SimpleNamespace(
        cfg=cfg,
        optimizer_pbounds={"alpha": (0.0, 1.0)},
        optuna_storage_dir=tmp_path,
        logger=_DummyTrialLogger(),
        best_rolling_score=0.0,
        completed_trials=0,
    )

    monkeypatch.setattr(study_manager, "configure_sampler", lambda cfg: "sampler")  # noqa: ARG005
    monkeypatch.setattr(study_manager, "configure_pruner", lambda cfg: None)  # noqa: ARG005
    monkeypatch.setattr(study_manager, "find_db_for_study", lambda storage_dir, name: ("sqlite:///parent.db", "parent.db"))  # noqa: ARG005
    monkeypatch.setattr(study_manager.optuna, "load_study", lambda study_name, storage: parent_study)  # noqa: ARG005

    with pytest.raises(ValueError, match="scorers have changed"):
        asyncio.run(study_manager.optimize_study(optimizer))


def test_optimize_study_raises_when_parent_study_cannot_be_found(monkeypatch, tmp_path: Path) -> None:
    cfg = _make_cfg(optuna_overrides={"storage_dir": str(tmp_path), "resume_from_study": "missing"})
    optimizer = SimpleNamespace(
        cfg=cfg,
        optimizer_pbounds={"alpha": (0.0, 1.0)},
        optuna_storage_dir=tmp_path,
        logger=_DummyTrialLogger(),
        best_rolling_score=0.0,
        completed_trials=0,
    )

    monkeypatch.setattr(study_manager, "configure_sampler", lambda cfg: "sampler")  # noqa: ARG005
    monkeypatch.setattr(study_manager, "configure_pruner", lambda cfg: None)  # noqa: ARG005
    monkeypatch.setattr(study_manager, "find_db_for_study", lambda storage_dir, name: (None, None))  # noqa: ARG005

    with pytest.raises(ValueError, match="Could not find study"):
        asyncio.run(study_manager.optimize_study(optimizer))


def test_optimize_study_resume_path_runs_existing_trials_and_optimize(monkeypatch, tmp_path: Path) -> None:
    cfg = _make_cfg(optuna_overrides={"storage_dir": str(tmp_path), "resume_from_study": "parent", "n_jobs": 3})
    cfg.optimizer.init_points = 2
    cfg.optimizer.n_iters = 1
    existing_trial = SimpleNamespace(number=0, state=optuna.trial.TrialState.COMPLETE, params={"alpha": 0.2})
    parent_study = _DummyStudy(trials=[existing_trial], user_attrs={"config_scorers": ["manual"]})
    parent_study.sampler = SimpleNamespace()
    callback_calls: list[int] = []
    analyzed: list[bool] = []
    to_thread_calls: list[tuple] = []
    optimizer = SimpleNamespace(
        cfg=cfg,
        optimizer_pbounds={"alpha": (0.0, 1.0)},
        optuna_storage_dir=tmp_path,
        logger=_DummyTrialLogger(),
        best_rolling_score=0.1,
        completed_trials=0,
    )

    async def fake_to_thread(callable_obj, *args, **kwargs):
        to_thread_calls.append((callable_obj, kwargs))
        callable_obj(*args, **kwargs)

    monkeypatch.setattr(study_manager, "configure_sampler", lambda cfg: "sampler")  # noqa: ARG005
    monkeypatch.setattr(study_manager, "configure_pruner", lambda cfg: None)  # noqa: ARG005
    monkeypatch.setattr(study_manager, "find_db_for_study", lambda storage_dir, name: ("sqlite:///parent.db", "parent.db"))  # noqa: ARG005
    monkeypatch.setattr(study_manager.optuna, "load_study", lambda study_name, storage: parent_study)  # noqa: ARG005
    monkeypatch.setattr(study_manager, "record_trial_callback", lambda optimizer, study, trial: callback_calls.append(trial.number))  # noqa: ARG005
    monkeypatch.setattr(study_manager, "analyze_parameter_importance", lambda optimizer: analyzed.append(True))  # noqa: ARG005
    monkeypatch.setattr(study_manager.asyncio, "to_thread", fake_to_thread)

    asyncio.run(study_manager.optimize_study(optimizer))

    assert optimizer.study is parent_study
    assert optimizer.completed_trials == 1
    assert optimizer.best_rolling_score == 0.9
    assert callback_calls == [0]
    assert to_thread_calls
    assert analyzed == [True]


def test_optimize_study_re_raises_non_keyboard_interrupt_from_optimize(monkeypatch, tmp_path: Path) -> None:
    cfg = _make_cfg(optuna_overrides={"storage_dir": str(tmp_path)})
    cfg.optimizer.init_points = 1
    cfg.optimizer.n_iters = 0
    optimizer = SimpleNamespace(
        cfg=cfg,
        optimizer_pbounds={"alpha": (0.0, 1.0)},
        optuna_storage_dir=tmp_path,
        logger=_DummyTrialLogger(),
        best_rolling_score=0.0,
        completed_trials=0,
    )

    monkeypatch.setattr(study_manager, "configure_sampler", lambda cfg: "sampler")  # noqa: ARG005
    monkeypatch.setattr(study_manager, "configure_pruner", lambda cfg: None)  # noqa: ARG005
    monkeypatch.setattr(study_manager, "build_storage_uri_for_new_study", lambda cfg, storage_dir, is_fork=False: ("sqlite:///new.db", "new.db"))  # noqa: ARG005
    monkeypatch.setattr(study_manager.optuna, "create_study", lambda **kwargs: _DummyStudy())
    monkeypatch.setattr(study_manager, "set_initial_study_attributes", lambda *args, **kwargs: None)
    monkeypatch.setattr(study_manager, "analyze_parameter_importance", lambda optimizer: None)  # noqa: ARG005

    async def fake_to_thread(callable_obj, *args, **kwargs):  # noqa: ARG001
        raise RuntimeError("optimize boom")

    monkeypatch.setattr(study_manager.asyncio, "to_thread", fake_to_thread)

    with pytest.raises(RuntimeError, match="optimize boom"):
        asyncio.run(study_manager.optimize_study(optimizer))
