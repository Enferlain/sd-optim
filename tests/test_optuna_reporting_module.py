from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import optuna
import pytest

from sd_optim.optimizers.optuna import reporting


class _DummyFig:
    def __init__(self, sink: list[str] | None = None, *, error: Exception | None = None) -> None:
        self.sink = sink
        self.error = error

    def write_image(self, path: str) -> None:
        if self.error is not None:
            raise self.error
        if self.sink is not None:
            self.sink.append(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("plot", encoding="utf-8")


class _DummyTrial:
    def __init__(
        self,
        *,
        number: int,
        value: float | None,
        params: dict[str, object] | None = None,
        state_name: str = "COMPLETE",
        finished: bool = True,
    ) -> None:
        self.number = number
        self.value = value
        self.params = params or {"alpha": 0.1}
        self.datetime_start = datetime(2026, 3, 17, 12, 0, 0)
        self.user_attrs = {"scorer_results": {"manual": value}, "scorer_results_payloads": [{"name": "payload"}]}
        self.attrs_set: dict[str, object] = {}
        self.state = getattr(optuna.trial.TrialState, state_name) if finished else SimpleNamespace(name=state_name, is_finished=lambda: finished)

    def set_user_attr(self, key: str, value: object) -> None:
        self.attrs_set[key] = value


def test_record_trial_callback_sets_attrs_and_triggers_progress_plot(monkeypatch) -> None:
    logged: list[dict] = []
    plot_calls: list[bool] = []
    optimizer = SimpleNamespace(
        optimization_start_time=1.0,
        merger=SimpleNamespace(output_file="model.safetensors"),
        iteration=7,
        logger=SimpleNamespace(log=lambda payload: logged.append(payload)),
    )
    trial = _DummyTrial(number=10, value=0.9)

    monkeypatch.setattr(reporting.time, "time", lambda: 6.0)
    monkeypatch.setattr(reporting, "create_progress_plots", lambda optimizer: plot_calls.append(True))  # noqa: ARG005

    reporting.record_trial_callback(optimizer, SimpleNamespace(), trial)

    assert trial.attrs_set["elapsed_seconds"] == 5.0
    assert trial.attrs_set["timestamp"] == 6.0
    assert trial.attrs_set["model_path"] == "model.safetensors"
    assert trial.attrs_set["iteration"] == 7
    assert logged[0]["trial_number"] == 10
    assert plot_calls == [True]


def test_get_hydra_output_dir_returns_none_without_hydra(caplog: pytest.LogCaptureFixture, monkeypatch) -> None:
    caplog.set_level("ERROR")
    monkeypatch.setattr(reporting.HydraConfig, "get", lambda: (_ for _ in ()).throw(ValueError("missing hydra")))

    assert reporting._get_hydra_output_dir("ctx") is None
    assert "Hydra context unavailable" in caplog.text


def test_write_plot_image_handles_kaleido_and_generic_errors(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("ERROR")

    reporting._write_plot_image(_DummyFig(error=ValueError("kaleido missing")), Path("kaleido.png"), label="plot A")
    reporting._write_plot_image(_DummyFig(error=ValueError("other value error")), Path("value.png"), label="plot C")
    reporting._write_plot_image(_DummyFig(error=RuntimeError("boom")), Path("boom.png"), label="plot B")

    assert "Kaleido missing/broken" in caplog.text
    assert "ValueError saving plot C" in caplog.text
    assert "Error saving plot B" in caplog.text


def test_create_progress_plots_writes_periodic_plot(monkeypatch, tmp_path: Path) -> None:
    paths: list[str] = []
    optimizer = SimpleNamespace(study=SimpleNamespace(trials=[_DummyTrial(number=1, value=0.5)]))

    monkeypatch.setattr(reporting.vis, "plot_optimization_history", lambda study: _DummyFig(paths))  # noqa: ARG005
    monkeypatch.setattr(reporting.HydraConfig, "get", lambda: SimpleNamespace(runtime=SimpleNamespace(output_dir=str(tmp_path))))

    reporting.create_progress_plots(optimizer)

    assert paths == [str(tmp_path / "visualizations" / "periodic" / "optuna_history_trial_0001.png")]


def test_create_progress_plots_handles_missing_trials_and_plot_errors(
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("DEBUG")

    reporting.create_progress_plots(SimpleNamespace(study=None))
    reporting.create_progress_plots(SimpleNamespace(study=SimpleNamespace(trials=[_DummyTrial(number=1, value=None)])))

    monkeypatch.setattr(reporting.vis, "plot_optimization_history", lambda study: (_ for _ in ()).throw(ImportError("no vis")))  # noqa: ARG005
    reporting.create_progress_plots(SimpleNamespace(study=SimpleNamespace(trials=[_DummyTrial(number=2, value=0.4)])))

    monkeypatch.setattr(reporting.vis, "plot_optimization_history", lambda study: (_ for _ in ()).throw(RuntimeError("plot boom")))  # noqa: ARG005
    reporting.create_progress_plots(SimpleNamespace(study=SimpleNamespace(trials=[_DummyTrial(number=3, value=0.4)])))

    assert "No study or trials available" in caplog.text
    assert "Not enough completed trials yet" in caplog.text
    assert "Optuna visualization module not available for periodic plots." in caplog.text
    assert "Failed to create periodic progress plot" in caplog.text


def test_analyze_parameter_importance_logs_and_writes_plot(monkeypatch, tmp_path: Path) -> None:
    paths: list[str] = []
    optimizer = SimpleNamespace(
        study=SimpleNamespace(trials=[_DummyTrial(number=1, value=0.4), _DummyTrial(number=2, value=0.8)]),
        study_name="run_demo",
    )

    monkeypatch.setattr(reporting.optuna.importance, "get_param_importances", lambda study: {"alpha": 0.75})  # noqa: ARG005
    monkeypatch.setattr(reporting.vis, "plot_param_importances", lambda study: _DummyFig(paths))  # noqa: ARG005
    monkeypatch.setattr(reporting.HydraConfig, "get", lambda: SimpleNamespace(runtime=SimpleNamespace(output_dir=str(tmp_path))))

    reporting.analyze_parameter_importance(optimizer)

    assert paths == [str(tmp_path / "visualizations" / "optuna_param_importances_run_demo.png")]


def test_analyze_parameter_importance_handles_short_circuits_and_errors(
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("WARNING")

    reporting.analyze_parameter_importance(SimpleNamespace(study=None, study_name="none"))

    optimizer = SimpleNamespace(study=SimpleNamespace(trials=[_DummyTrial(number=1, value=0.2), _DummyTrial(number=2, value=0.3)]), study_name="run")
    monkeypatch.setattr(reporting.optuna.importance, "get_param_importances", lambda study: (_ for _ in ()).throw(RuntimeError("imp boom")))  # noqa: ARG005
    reporting.analyze_parameter_importance(optimizer)

    monkeypatch.setattr(reporting.optuna.importance, "get_param_importances", lambda study: {"alpha": 0.2})  # noqa: ARG005
    monkeypatch.setattr(reporting.vis, "plot_param_importances", lambda study: (_ for _ in ()).throw(ImportError("missing dep")))  # noqa: ARG005
    reporting.analyze_parameter_importance(optimizer)

    monkeypatch.setattr(reporting.vis, "plot_param_importances", lambda study: _DummyFig([]))  # noqa: ARG005
    monkeypatch.setattr(reporting.HydraConfig, "get", lambda: (_ for _ in ()).throw(ValueError("no hydra")))
    reporting.analyze_parameter_importance(optimizer)

    assert "Not enough trials" in caplog.text
    assert "Could not calculate parameter importance" in caplog.text
    assert "Could not generate importance plot due to missing dependency" in caplog.text


def test_postprocess_study_handles_empty_and_failed_trials(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("WARNING")

    asyncio.run(reporting.postprocess_study(SimpleNamespace(study=None, optimization_start_time=None)))
    asyncio.run(
        reporting.postprocess_study(
            SimpleNamespace(
                study=SimpleNamespace(trials=[_DummyTrial(number=1, value=None)]),
                optimization_start_time=None,
            )
        )
    )

    assert "No trials completed. Nothing to report." in caplog.text
    assert "No successful completed trials to summarize yet." in caplog.text


def test_postprocess_study_generates_plots(monkeypatch, tmp_path: Path) -> None:
    paths: list[str] = []
    optimizer = SimpleNamespace(
        study=SimpleNamespace(
            trials=[
                _DummyTrial(number=1, value=0.2),
                _DummyTrial(number=2, value=0.9, params={"alpha": 0.9}),
            ]
        ),
        study_name="run_post",
        optimization_start_time=0.0,
    )

    monkeypatch.setattr(reporting.time, "time", lambda: 10.0)
    monkeypatch.setattr(reporting.HydraConfig, "get", lambda: SimpleNamespace(runtime=SimpleNamespace(output_dir=str(tmp_path))))
    monkeypatch.setattr(reporting.vis, "plot_optimization_history", lambda study: _DummyFig(paths))  # noqa: ARG005
    monkeypatch.setattr(reporting.vis, "plot_param_importances", lambda study: _DummyFig(paths))  # noqa: ARG005
    monkeypatch.setattr(reporting.vis, "plot_slice", lambda study: _DummyFig(paths))  # noqa: ARG005
    monkeypatch.setattr(reporting.vis, "plot_parallel_coordinate", lambda study: _DummyFig(paths))  # noqa: ARG005

    asyncio.run(reporting.postprocess_study(optimizer))

    assert len(paths) == 4
    assert any(path.endswith("optuna_parallel_coordinate_run_post.png") for path in paths)


def test_postprocess_study_handles_runtime_stats_and_plot_generation_errors(
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("WARNING")
    optimizer = SimpleNamespace(
        study=SimpleNamespace(trials=[_DummyTrial(number=1, value=0.6)]),
        study_name="run_short",
        optimization_start_time=2.0,
    )
    monkeypatch.setattr(reporting.time, "time", lambda: 8.0)
    asyncio.run(reporting.postprocess_study(optimizer))

    optimizer_many = SimpleNamespace(
        study=SimpleNamespace(trials=[_DummyTrial(number=1, value=0.2), _DummyTrial(number=2, value=0.6)]),
        study_name="run_many",
        optimization_start_time=None,
    )
    monkeypatch.setattr(reporting.HydraConfig, "get", lambda: (_ for _ in ()).throw(ValueError("no hydra")))
    asyncio.run(reporting.postprocess_study(optimizer_many))

    monkeypatch.setattr(reporting.HydraConfig, "get", lambda: SimpleNamespace(runtime=SimpleNamespace(output_dir=str(Path.cwd()))))
    monkeypatch.setattr(reporting.vis, "plot_optimization_history", lambda study: (_ for _ in ()).throw(ValueError("bad history")))  # noqa: ARG005
    monkeypatch.setattr(reporting.vis, "plot_param_importances", lambda study: (_ for _ in ()).throw(ImportError("missing dep")))  # noqa: ARG005
    monkeypatch.setattr(reporting.vis, "plot_slice", lambda study: (_ for _ in ()).throw(RuntimeError("slice boom")))  # noqa: ARG005
    monkeypatch.setattr(reporting.vis, "plot_parallel_coordinate", lambda study: _DummyFig([]))  # noqa: ARG005
    asyncio.run(reporting.postprocess_study(optimizer_many))

    assert "Not enough trials (need >= 2)" in caplog.text
    assert "Hydra context unavailable" in caplog.text
    assert "Could not generate Optuna plot 'optimization_history'" in caplog.text
    assert "Could not generate Optuna plot 'param_importances' due to missing dependency" in caplog.text
    assert "Unexpected error generating Optuna plot 'slice'" in caplog.text


def test_collect_history_and_create_visualization_report(monkeypatch, tmp_path: Path) -> None:
    paths: list[str] = []
    study = SimpleNamespace(
        trials=[
            _DummyTrial(number=1, value=0.2),
            _DummyTrial(number=2, value=None),
            _DummyTrial(number=3, value=0.8),
        ]
    )
    optimizer = SimpleNamespace(study=study, study_name="run_report")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(reporting.HydraConfig, "get", lambda: (_ for _ in ()).throw(ValueError("no hydra")))
    monkeypatch.setattr(reporting.vis, "plot_optimization_history", lambda study: _DummyFig(paths))  # noqa: ARG005
    monkeypatch.setattr(reporting.vis, "plot_param_importances", lambda study: _DummyFig(paths))  # noqa: ARG005
    monkeypatch.setattr(reporting.vis, "plot_slice", lambda study: _DummyFig(paths))  # noqa: ARG005
    monkeypatch.setattr(reporting.vis, "plot_parallel_coordinate", lambda study: _DummyFig(paths))  # noqa: ARG005
    monkeypatch.setattr(reporting.vis, "plot_contour", lambda study: _DummyFig(paths))  # noqa: ARG005
    monkeypatch.setattr(reporting.vis, "plot_edf", lambda study: _DummyFig(paths))  # noqa: ARG005
    monkeypatch.setattr(reporting.vis, "plot_rank", lambda study: _DummyFig(paths))  # noqa: ARG005

    history = reporting.collect_optimization_history(study)
    reporting.create_visualization_report(optimizer)

    assert [entry["trial_number"] for entry in history] == [1, 3]
    assert len(paths) == 7
    assert (tmp_path / "optuna_visualizations_report").is_dir()


def test_collect_history_none_and_visualization_report_error_paths(
    monkeypatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("WARNING")
    assert reporting.collect_optimization_history(None) == []

    reporting.create_visualization_report(SimpleNamespace(study=None, study_name="none"))

    optimizer = SimpleNamespace(
        study=SimpleNamespace(trials=[_DummyTrial(number=1, value=0.2), _DummyTrial(number=2, value=0.9)]),
        study_name="run_err",
    )
    output_dir = tmp_path / "report"
    monkeypatch.setattr(reporting.vis, "plot_optimization_history", lambda study: (_ for _ in ()).throw(ValueError("bad plot")))  # noqa: ARG005
    monkeypatch.setattr(reporting.vis, "plot_param_importances", lambda study: (_ for _ in ()).throw(ImportError("missing dep")))  # noqa: ARG005
    monkeypatch.setattr(reporting.vis, "plot_slice", lambda study: (_ for _ in ()).throw(RuntimeError("slice fail")))  # noqa: ARG005
    monkeypatch.setattr(reporting.vis, "plot_parallel_coordinate", lambda study: _DummyFig([]))  # noqa: ARG005
    monkeypatch.setattr(reporting.vis, "plot_contour", lambda study: _DummyFig([]))  # noqa: ARG005
    monkeypatch.setattr(reporting.vis, "plot_edf", lambda study: _DummyFig([]))  # noqa: ARG005
    monkeypatch.setattr(reporting.vis, "plot_rank", lambda study: _DummyFig([]))  # noqa: ARG005

    reporting.create_visualization_report(optimizer, output_dir=output_dir)

    assert "Not enough trials for visualization report." in caplog.text
    assert "Report: Cannot generate plot 'optimization_history'" in caplog.text
    assert "Report: Cannot generate plot 'param_importances', missing dependency" in caplog.text
    assert "Report: Unexpected error generating plot 'slice'" in caplog.text
