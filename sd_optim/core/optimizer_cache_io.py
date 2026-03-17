from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class HistoryCacheLoadResult:
    entries: dict[str, dict[str, Any]]
    manifest_hits: int = 0
    png_hits: int = 0


def load_history_cache(logs_dir: Path, *, scan_legacy_pngs: bool = False) -> HistoryCacheLoadResult:
    """Load cached image results from run manifests and optional legacy PNG metadata."""
    history_cache: dict[str, dict[str, Any]] = {}
    manifest_hits = 0
    png_hits = 0

    manifest_paths = sorted(
        logs_dir.rglob("run_manifest.json"),
        key=lambda path: (path.stat().st_mtime_ns, path.as_posix()),
        reverse=True,
    )
    for manifest_path in manifest_paths:
        try:
            run_dir = manifest_path.parent
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            for image_hash, data in manifest_data.items():
                if image_hash in history_cache:
                    continue
                relative_path = data.get("path")
                if not relative_path:
                    continue
                full_path = (run_dir / relative_path).resolve()
                if not full_path.exists():
                    continue
                history_cache[image_hash] = {
                    **data,
                    "full_path": full_path,
                }
                manifest_hits += 1
        except Exception as error:
            logger.debug("Could not load manifest %s: %s", manifest_path, error)

    if scan_legacy_pngs:
        legacy_dirs = []
        for image_dir in logs_dir.rglob("imgs"):
            run_dir = image_dir.parent
            if not (run_dir / "run_manifest.json").exists():
                legacy_dirs.append(image_dir)

        legacy_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        for image_dir in legacy_dirs:
            for png_path in image_dir.glob("*.png"):
                try:
                    with Image.open(png_path) as image:
                        meta_hash = image.info.get("sd_optim_hash")
                        if meta_hash and meta_hash not in history_cache:
                            history_cache[meta_hash] = {
                                "full_path": png_path,
                                "scores": json.loads(image.info.get("sd_optim_scores", "{}")),
                                "final_score": float(image.info.get("sd_optim_final_score", 0)),
                            }
                            png_hits += 1
                except Exception as error:
                    logger.debug("Could not read legacy PNG metadata from %s: %s", png_path, error)

    return HistoryCacheLoadResult(entries=history_cache, manifest_hits=manifest_hits, png_hits=png_hits)


def save_run_manifest(output_dir: Path, run_manifest: dict[str, dict[str, Any]]) -> Path:
    """Persist the current run manifest to the Hydra output directory."""
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def build_run_manifest_entry(
    *,
    image_path: Path,
    output_dir: Path | None,
    scorer_results: dict[str, Any],
    final_score: float,
    scorer_setup_fp: str,
) -> dict[str, Any]:
    """Build a manifest entry using a relative image path when possible."""
    if output_dir is None:
        stored_path = str(image_path)
    else:
        try:
            stored_path = image_path.relative_to(output_dir).as_posix()
        except ValueError:
            stored_path = str(image_path)
    return {
        "path": stored_path,
        "scores": scorer_results,
        "final_score": final_score,
        "scorer_setup_fp": scorer_setup_fp,
    }


def build_image_output_path(
    *,
    output_dir: Path,
    name: str,
    score: float,
    iteration: int,
    img_order_index: int,
) -> Path:
    """Build the on-disk output path for a scored image."""
    return output_dir / "imgs" / f"{iteration:03}-{img_order_index:02}-{name}-{score:4.3f}.png"
