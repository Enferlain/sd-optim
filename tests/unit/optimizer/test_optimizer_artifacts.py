from __future__ import annotations

from pathlib import Path

from sd_optim.core import optimizer_artifacts


def test_build_best_output_path_appends_best_suffix() -> None:
    current_path = Path("/tmp/run/model.safetensors")

    assert optimizer_artifacts.build_best_output_path(current_path) == Path("/tmp/run/model_best.safetensors")


def test_replace_best_model_moves_current_and_removes_previous(tmp_path: Path) -> None:
    current_model = tmp_path / "candidate.safetensors"
    previous_best = tmp_path / "older_best.safetensors"
    current_model.write_text("new", encoding="utf-8")
    previous_best.write_text("old", encoding="utf-8")

    new_best_path = optimizer_artifacts.replace_best_model(
        current_model_path=current_model,
        previous_best_path=previous_best,
    )

    assert new_best_path == tmp_path / "candidate_best.safetensors"
    assert new_best_path.read_text(encoding="utf-8") == "new"
    assert not current_model.exists()
    assert not previous_best.exists()


def test_remove_model_file_returns_false_for_missing_path(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.safetensors"

    assert optimizer_artifacts.remove_model_file(missing_path) is False


def test_save_best_log_writes_iteration_and_params(tmp_path: Path) -> None:
    log_path = optimizer_artifacts.save_best_log(
        output_dir=tmp_path,
        params={"alpha": 0.25, "beta": 1},
        iteration=12,
    )

    assert log_path == tmp_path / "best.log"
    log_text = log_path.read_text(encoding="utf-8")
    assert "Best Iteration: 12." in log_text
    assert "alpha: 0.25" in log_text
    assert "beta: 1" in log_text
