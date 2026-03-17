from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any


def build_best_output_path(current_model_path: Path) -> Path:
    """Return the `_best` variant of a generated model path."""
    return current_model_path.with_name(f"{current_model_path.stem}_best{current_model_path.suffix}")


def replace_best_model(*, current_model_path: Path, previous_best_path: Path | None) -> Path:
    """Promote the current model to the best-model path, deleting the previous best when needed."""
    new_best_path = build_best_output_path(current_model_path)
    if previous_best_path and previous_best_path.exists() and previous_best_path != new_best_path:
        previous_best_path.unlink()
    shutil.move(current_model_path, new_best_path)
    return new_best_path


def remove_model_file(model_path: Path | None) -> bool:
    """Delete a model file when it exists."""
    if not model_path or not model_path.exists():
        return False
    model_path.unlink()
    return True


def save_best_log(*, output_dir: Path, params: dict[str, Any], iteration: int) -> Path:
    """Write the best-trial recap file into the run output directory."""
    log_path = output_dir / "best.log"
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"Best Iteration: {iteration}.\n\n")
        handle.write("\n".join(f"{key}: {value}" for key, value in params.items()))
        handle.write("\n")
    return log_path
