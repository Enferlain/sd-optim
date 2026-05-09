from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def test_optuna_split_modules_export_expected_helpers() -> None:
    sampler_module = importlib.import_module("sd_optim.optimizers.optuna.sampler_factory")
    logger_module = importlib.import_module("sd_optim.optimizers.optuna.trial_logger")
    dashboard_module = importlib.import_module("sd_optim.optimizers.optuna.dashboard")
    objective_module = importlib.import_module("sd_optim.optimizers.optuna.objective")
    reporting_module = importlib.import_module("sd_optim.optimizers.optuna.reporting")
    study_manager_module = importlib.import_module("sd_optim.optimizers.optuna.study_manager")

    assert callable(sampler_module.configure_sampler)
    assert hasattr(logger_module, "TrialLogger")
    assert callable(dashboard_module.run_dashboard_in_background)
    assert callable(dashboard_module.start_dashboard_for_optimizer)
    assert callable(objective_module.run_objective)
    assert callable(reporting_module.postprocess_study)
    assert callable(study_manager_module.optimize_study)


def test_trial_logger_round_trip(tmp_path: Path) -> None:
    logger_module = importlib.import_module("sd_optim.optimizers.optuna.trial_logger")
    trial_logger = logger_module.TrialLogger()
    log_path = tmp_path / "trials.jsonl"

    trial_logger.set_path(log_path)
    trial_logger.log({"trial_number": 1, "target": 0.5})
    trial_logger.log({"trial_number": 2, "target": 0.7})

    assert trial_logger.load_trials() == [
        {"trial_number": 1, "target": 0.5},
        {"trial_number": 2, "target": 0.7},
    ]


def test_trial_logger_skips_invalid_json_lines(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    logger_module = importlib.import_module("sd_optim.optimizers.optuna.trial_logger")
    trial_logger = logger_module.TrialLogger()
    log_path = tmp_path / "trials.jsonl"
    log_path.write_text('{"ok": 1}\nnot-json\n', encoding="utf-8")

    trial_logger.set_path(log_path)
    caplog.set_level("WARNING")

    assert trial_logger.load_trials() == [{"ok": 1}]
    assert "Skipping invalid line in trial log" in caplog.text


def test_collect_optimization_history_filters_failed_trials() -> None:
    reporting_module = importlib.import_module("sd_optim.optimizers.optuna.reporting")
    finished_trial = type(
        "FinishedTrial",
        (),
        {
            "number": 1,
            "value": 0.9,
            "params": {"alpha": 0.2},
            "datetime_start": None,
        },
    )()
    failed_trial = type(
        "FailedTrial",
        (),
        {
            "number": 2,
            "value": None,
            "params": {"alpha": 0.8},
            "datetime_start": None,
        },
    )()
    study = type("Study", (), {"trials": [finished_trial, failed_trial]})()

    assert reporting_module.collect_optimization_history(study) == [
        {
            "trial_number": 1,
            "value": 0.9,
            "params": {"alpha": 0.2},
            "datetime": None,
        }
    ]
