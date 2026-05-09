from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from sd_optim.core import optimizer_cache_io


def test_load_history_cache_prefers_newer_manifest_entries(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    older_run = logs_dir / "2026-01-01_00-00-00"
    newer_run = logs_dir / "2026-01-02_00-00-00"
    older_imgs = older_run / "imgs"
    newer_imgs = newer_run / "imgs"
    older_imgs.mkdir(parents=True)
    newer_imgs.mkdir(parents=True)

    old_image = older_imgs / "shared.png"
    new_image = newer_imgs / "shared.png"
    old_image.write_bytes(b"old")
    new_image.write_bytes(b"new")

    (older_run / "run_manifest.json").write_text(
        json.dumps({"same-hash": {"path": "imgs/shared.png", "scores": {"manual": 0.1}, "final_score": 0.1}}),
        encoding="utf-8",
    )
    (newer_run / "run_manifest.json").write_text(
        json.dumps({"same-hash": {"path": "imgs/shared.png", "scores": {"manual": 0.9}, "final_score": 0.9}}),
        encoding="utf-8",
    )

    newer_manifest = newer_run / "run_manifest.json"
    older_manifest = older_run / "run_manifest.json"
    newer_manifest.touch()
    older_manifest.touch()

    history_cache = optimizer_cache_io.load_history_cache(logs_dir).entries

    assert history_cache["same-hash"]["full_path"] == new_image
    assert history_cache["same-hash"]["final_score"] == 0.9


def test_load_history_cache_reads_legacy_png_metadata_when_enabled(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    legacy_imgs = logs_dir / "2026-01-03_00-00-00" / "imgs"
    legacy_imgs.mkdir(parents=True)
    image_path = legacy_imgs / "legacy.png"

    image = Image.new("RGB", (2, 2), color=(255, 0, 0))
    pnginfo = PngInfo()
    pnginfo.add_text("sd_optim_hash", "legacy-hash")
    pnginfo.add_text("sd_optim_scores", json.dumps({"manual": 0.7}))
    pnginfo.add_text("sd_optim_final_score", "0.7")
    image.save(image_path, pnginfo=pnginfo)

    history_cache = optimizer_cache_io.load_history_cache(logs_dir, scan_legacy_pngs=True).entries

    assert history_cache["legacy-hash"]["full_path"] == image_path
    assert history_cache["legacy-hash"]["scores"] == {"manual": 0.7}
    assert history_cache["legacy-hash"]["final_score"] == 0.7


def test_save_run_manifest_writes_sorted_manifest_json(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    manifest = {
        "b": {"path": "imgs/b.png", "final_score": 0.2},
        "a": {"path": "imgs/a.png", "final_score": 0.9},
    }

    manifest_path = optimizer_cache_io.save_run_manifest(output_dir, manifest)

    assert manifest_path == output_dir / "run_manifest.json"
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    assert manifest_path.read_text(encoding="utf-8").find('"a"') < manifest_path.read_text(encoding="utf-8").find('"b"')


def test_build_run_manifest_entry_uses_relative_path_when_possible(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    image_path = output_dir / "imgs" / "image.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"png")

    entry = optimizer_cache_io.build_run_manifest_entry(
        image_path=image_path,
        output_dir=output_dir,
        scorer_results={"manual": 0.4},
        final_score=0.4,
        scorer_setup_fp="scorer-fp",
    )

    assert entry == {
        "path": "imgs/image.png",
        "scores": {"manual": 0.4},
        "final_score": 0.4,
        "scorer_setup_fp": "scorer-fp",
    }


def test_build_run_manifest_entry_uses_absolute_path_without_output_dir(tmp_path: Path) -> None:
    image_path = tmp_path / "imgs" / "external.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"png")

    entry = optimizer_cache_io.build_run_manifest_entry(
        image_path=image_path,
        output_dir=None,
        scorer_results={"manual": 0.2},
        final_score=0.2,
        scorer_setup_fp="scorer-fp",
    )

    assert entry["path"] == str(image_path)


def test_build_image_output_path_keeps_existing_filename_scheme(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"

    image_path = optimizer_cache_io.build_image_output_path(
        output_dir=output_dir,
        name="sample",
        score=0.12345,
        iteration=7,
        img_order_index=2,
    )

    assert image_path == output_dir / "imgs" / "007-02-sample-0.123.png"
