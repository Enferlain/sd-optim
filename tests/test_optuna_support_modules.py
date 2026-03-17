from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from sd_optim.optimizers.optuna import dashboard, sampler_factory, trial_logger


def _make_cfg(*, sampler_type: str = "tpe", extra_sampler: dict | None = None, random_state: int = 218):
    sampler = {"type": sampler_type}
    if extra_sampler:
        sampler.update(extra_sampler)
    return OmegaConf.create(
        {
            "optimizer": {
                "n_iters": 4,
                "init_points": 2,
                "random_state": random_state,
                "optuna_config": {
                    "sampler": sampler,
                    "pruner_type": "median",
                    "use_pruning": False,
                },
            }
        }
    )


def test_validate_optimizer_config_reports_missing_fields(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("ERROR")
    cfg = OmegaConf.create({"optimizer": {"init_points": 2, "optuna_config": {"sampler": {"type": "tpe"}}}})

    assert sampler_factory.validate_optimizer_config(cfg) is False
    assert "Missing required configuration fields" in caplog.text


def test_configure_sampler_random_and_gp_cover_generated_seed(monkeypatch) -> None:
    captured_random = {}
    captured_gp = {}

    class DummyRandom:
        def __init__(self, **kwargs):
            captured_random.update(kwargs)

    class DummyGp:
        def __init__(self, **kwargs):
            captured_gp.update(kwargs)

    monkeypatch.setattr(sampler_factory.os, "urandom", lambda n: b"\x00\x00\x00\x02")  # noqa: ARG005
    monkeypatch.setattr(sampler_factory, "RandomSampler", DummyRandom)
    monkeypatch.setattr(sampler_factory, "GPSampler", DummyGp)

    sampler_factory.configure_sampler(_make_cfg(sampler_type="random", random_state=-1))
    sampler_factory.configure_sampler(_make_cfg(sampler_type="gp"))

    assert captured_random["seed"] == 2
    assert captured_gp["n_startup_trials"] == 2
    assert captured_gp["seed"] == 218


def test_configure_sampler_grid_and_unknown_fallback(monkeypatch) -> None:
    captured_grid = {}
    captured_tpe = {}

    class DummyGrid:
        def __init__(self, search_space):
            captured_grid["search_space"] = search_space

    class DummyTpe:
        def __init__(self, **kwargs):
            captured_tpe.update(kwargs)

    monkeypatch.setattr(sampler_factory, "GridSampler", DummyGrid)
    monkeypatch.setattr(sampler_factory, "TPESampler", DummyTpe)

    sampler_factory.configure_sampler(
        _make_cfg(
            sampler_type="grid",
            extra_sampler={"search_space": {"alpha": [0.1, 0.2], "beta": ["x", "y"]}},
        )
    )
    sampler_factory.configure_sampler(_make_cfg(sampler_type="mystery"))

    assert captured_grid["search_space"] == {"alpha": [0.1, 0.2], "beta": ["x", "y"]}
    assert captured_tpe["n_startup_trials"] == 2
    assert captured_tpe["multivariate"] is True


def test_configure_sampler_grid_requires_search_space() -> None:
    with pytest.raises(ValueError, match="requires a 'search_space'"):
        sampler_factory.configure_sampler(_make_cfg(sampler_type="grid"))


def test_configure_sampler_cmaes_warns_for_categorical_heavy_space(
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    captured_cmaes = {}

    class DummyCma:
        def __init__(self, **kwargs):
            captured_cmaes.update(kwargs)

    monkeypatch.setattr(sampler_factory, "CmaEsSampler", DummyCma)
    caplog.set_level("INFO")

    sampler_factory.configure_sampler(
        _make_cfg(sampler_type="cmaes", extra_sampler={"warn_independent_sampling": True}),
        optimizer_pbounds={
            "continuous_a": (0.0, 1.0),
            "categorical_a": [0.0, 1.0],
            "categorical_b": ["x", "y"],
        },
    )

    assert captured_cmaes["warn_independent_sampling"] is True
    assert "categorical-heavy" in caplog.text
    assert "independent sampling" in caplog.text


def test_configure_sampler_cmaes_skips_categorical_warning_for_continuous_space(
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class DummyCma:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(sampler_factory, "CmaEsSampler", DummyCma)
    caplog.set_level("INFO")

    sampler_factory.configure_sampler(
        _make_cfg(sampler_type="cmaes"),
        optimizer_pbounds={
            "continuous_a": (0.0, 1.0),
            "continuous_b": {"range": (0.1, 0.9), "step": 0.1},
        },
    )

    assert "categorical-heavy" not in caplog.text


def test_configure_sampler_nsgaii_handles_crossover_variants(monkeypatch) -> None:
    captured_kwargs: list[dict] = []

    class DummyNsga:
        def __init__(self, **kwargs):
            captured_kwargs.append(kwargs)

    class DummyBaseCrossover:
        pass

    class DummyUniformCrossover(DummyBaseCrossover):
        def __call__(self):
            return self

    nsgaii_module = types.ModuleType("optuna.samplers.nsgaii")
    nsgaii_module.BaseCrossover = DummyBaseCrossover
    nsgaii_module.UniformCrossover = DummyUniformCrossover
    monkeypatch.setitem(sys.modules, "optuna.samplers.nsgaii", nsgaii_module)
    monkeypatch.setattr(sampler_factory, "NSGAIISampler", DummyNsga)

    sampler_factory.configure_sampler(_make_cfg(sampler_type="nsgaii", extra_sampler={"crossover": "UniformCrossover"}))
    sampler_factory.configure_sampler(_make_cfg(sampler_type="nsgaii", extra_sampler={"crossover": "MissingCrossover"}))

    assert captured_kwargs[0]["crossover"].__class__.__name__.endswith("UniformCrossover")
    assert "crossover" not in captured_kwargs[1]


def test_trial_logger_handles_missing_path_write_error_and_load_error(
    monkeypatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("ERROR")
    logger = trial_logger.TrialLogger()

    logger.log({"trial_number": 1})
    assert "Trial logger path not set" in caplog.text

    log_path = tmp_path / "trial_logs" / "trials.jsonl"
    logger.set_path(log_path)
    assert log_path.exists()
    assert logger.load_trials() == []

    broken_logger = trial_logger.TrialLogger()
    broken_logger.set_path(tmp_path / "broken" / "trials.jsonl")

    def failing_open(*args, **kwargs):
        raise OSError("disk error")

    monkeypatch.setattr(trial_logger.Path, "open", failing_open)
    logger.log({"trial_number": 2})
    assert "Failed to write to trial log" in caplog.text

    assert broken_logger.load_trials() == []
    assert "Failed to load trials log" in caplog.text


def test_dashboard_handles_launch_failures_and_storage_initialization(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(dashboard.time, "sleep", lambda _seconds: None)

    monkeypatch.setattr(dashboard.subprocess, "Popen", lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError()))
    assert dashboard.run_dashboard_in_background("sqlite:///tmp/test.db", 8080) is None

    monkeypatch.setattr(dashboard.subprocess, "Popen", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    assert dashboard.run_dashboard_in_background("sqlite:///tmp/test.db", 8080) is None

    optimizer = SimpleNamespace(cfg=_make_cfg(), optuna_storage_dir=None)
    monkeypatch.setattr(dashboard, "initialize_storage_dir", lambda cfg: tmp_path)  # noqa: ARG005
    monkeypatch.setattr(dashboard, "build_storage_uri_for_new_study", lambda cfg, storage_dir: ("sqlite:///tmp/test.db", "study.db"))  # noqa: ARG005
    monkeypatch.setattr(dashboard, "run_dashboard_in_background", lambda storage_uri, port: {"uri": storage_uri, "port": port})

    result = dashboard.start_dashboard_for_optimizer(optimizer, port=9091)

    assert optimizer.optuna_storage_dir == tmp_path
    assert result == {"uri": "sqlite:///tmp/test.db", "port": 9091}


def test_dashboard_returns_none_when_storage_lookup_fails(monkeypatch) -> None:
    optimizer = SimpleNamespace(cfg=_make_cfg(), optuna_storage_dir=Path("/tmp/does-not-matter"))
    monkeypatch.setattr(
        dashboard,
        "build_storage_uri_for_new_study",
        lambda cfg, storage_dir: (_ for _ in ()).throw(RuntimeError("lookup failed")),
    )

    assert dashboard.start_dashboard_for_optimizer(optimizer) is None
