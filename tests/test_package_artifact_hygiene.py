from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_scorer_scratch_scripts_do_not_live_in_runtime_package() -> None:
    package_models_dir = REPO_ROOT / "sd_optim" / "models"
    scratch_files = sorted(path.name for path in package_models_dir.glob("test_*.py"))

    assert scratch_files == []


def test_sample_images_do_not_live_in_runtime_package() -> None:
    package_models_dir = REPO_ROOT / "sd_optim" / "models"
    image_files = sorted(
        path.name
        for pattern in ("*.png", "*.jpg", "*.jpeg")
        for path in package_models_dir.glob(pattern)
    )

    assert image_files == []


def test_archived_scratch_bundle_does_not_live_in_runtime_package() -> None:
    archive_path = REPO_ROOT / "sd_optim" / "random_scripts.7z"

    assert not archive_path.exists()
