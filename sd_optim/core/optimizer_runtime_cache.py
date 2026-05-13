from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hydra.core.hydra_config import HydraConfig
from PIL import Image

from sd_optim.core.optimizer_cache import calculate_image_hash, reuse_cached_results_enabled
from sd_optim.core.optimizer_cache_io import build_run_manifest_entry
from sd_optim.core.trial_scorer_summary import build_trial_scorer_summary
from sd_optim.scoring.runtime import average_calc

if TYPE_CHECKING:
    from sd_optim.core.optimizer_base import Optimizer

logger = logging.getLogger(__name__)

CacheResult = tuple[str, dict[str, Any], dict[str, Any], str]


def resolve_trial_cache_candidates(
    optimizer: Optimizer,
    params: dict[str, Any],
    payloads: list[dict[str, Any]],
) -> tuple[str, list[CacheResult]]:
    """Classify payloads against reusable history cache entries for one trial."""
    reuse_cached_results = reuse_cached_results_enabled(optimizer.cfg)
    current_scorers = {s.lower() for s in optimizer.cfg.scorer_method}
    scorer_setup_fp = optimizer.scorer_setup_fp
    cache_results: list[CacheResult] = []
    overall_tier = "full_hit"

    if reuse_cached_results:
        for payload in payloads:
            image_hash = calculate_image_hash(params, payload, optimizer.generation_setup_fp)
            cached = optimizer.history_cache.get(image_hash)

            if cached and cached.get("final_score") is not None:
                cached_scorers = set(cached.get("scores", {}).keys()) - {"combined"}
                cached_fp = cached.get("scorer_setup_fp")
                if (
                    cached_scorers
                    and cached_scorers == current_scorers
                    and isinstance(cached_fp, str)
                    and cached_fp == scorer_setup_fp
                ):
                    cache_results.append((image_hash, cached, payload, "full_hit"))
                elif cached.get("full_path") and Path(cached["full_path"]).exists():
                    cache_results.append((image_hash, cached, payload, "partial_hit"))
                    overall_tier = "partial_hit"
                else:
                    overall_tier = "full_miss"
                    break
            elif cached and cached.get("full_path") and Path(cached["full_path"]).exists():
                cache_results.append((image_hash, cached, payload, "partial_hit"))
                overall_tier = "partial_hit"
            else:
                overall_tier = "full_miss"
                break
    else:
        overall_tier = "full_miss"
        logger.info("Universal Reuse: Disabled for this run; generating fresh outputs.")

    return overall_tier, cache_results


async def maybe_reuse_cached_trial_results(
    optimizer: Optimizer,
    *,
    params: dict[str, Any],
    payloads: list[dict[str, Any]],
    target_paths: list[str],
    iteration_start_time: float,
) -> float | None:
    """Reuse or re-score cached outputs when the current trial fully or partially matches history."""
    overall_tier, cache_results = resolve_trial_cache_candidates(optimizer, params, payloads)
    if overall_tier == "full_hit" and cache_results:
        return _reuse_full_hit_results(
            optimizer,
            cache_results=cache_results,
            target_paths=target_paths,
            iteration_start_time=iteration_start_time,
        )

    if overall_tier == "partial_hit" and cache_results:
        return await _reuse_partial_hit_results(
            optimizer,
            cache_results=cache_results,
            target_paths=target_paths,
            iteration_start_time=iteration_start_time,
        )

    return None


def _reuse_full_hit_results(
    optimizer: Optimizer,
    *,
    cache_results: list[CacheResult],
    target_paths: list[str],
    iteration_start_time: float,
) -> float:
    cached_scores = [c[1]["final_score"] for c in cache_results]
    cached_weights = [c[2].get("score_weight", 1.0) for c in cache_results]
    avg_score = average_calc(cached_scores, cached_weights, optimizer.cfg.img_average_type)
    payload_entries: list[dict[str, Any]] = []
    for idx, (_, cached, payload, _) in enumerate(cache_results):
        payload_name = target_paths[idx] if idx < len(target_paths) else f"payload_{idx}"
        payload_entries.append(
            {
                "name": payload_name,
                "weight": payload.get("score_weight", 1.0),
                "scores": dict(cached.get("scores", {})),
                "combined": cached.get("final_score"),
            }
        )
    optimizer.last_trial_scorer_summary = build_trial_scorer_summary(
        payload_entries,
        final_score=avg_score,
        combine_scores=lambda values, weights: average_calc(values, weights, optimizer.cfg.img_average_type),
    )
    elapsed = time.time() - iteration_start_time
    logger.info("CACHE HIT: All %s images reused. Score: %.4f (%.2fs)", len(cached_scores), avg_score, elapsed)
    return avg_score


async def _reuse_partial_hit_results(
    optimizer: Optimizer,
    *,
    cache_results: list[CacheResult],
    target_paths: list[str],
    iteration_start_time: float,
) -> float | None:
    current_scorers = {s.lower() for s in optimizer.cfg.scorer_method}
    scorer_setup_fp = optimizer.scorer_setup_fp
    logger.info(
        "PARTIAL CACHE HIT: %s images found, re-scoring with current scorers (%s)",
        len(cache_results),
        ", ".join(sorted(current_scorers)),
    )
    rescored_scores = []
    rescored_weights = []
    payload_entries: list[dict[str, Any]] = []
    overall_tier = "partial_hit"

    for idx, (image_hash, cached, payload, tier) in enumerate(cache_results):
        payload_name = target_paths[idx] if idx < len(target_paths) else f"payload_{idx}"
        if tier == "full_hit":
            rescored_scores.append(cached["final_score"])
            rescored_weights.append(payload.get("score_weight", 1.0))
            payload_entries.append(
                {
                    "name": payload_name,
                    "weight": payload.get("score_weight", 1.0),
                    "scores": dict(cached.get("scores", {})),
                    "combined": cached.get("final_score"),
                }
            )
            continue

        try:
            image = Image.open(cached["full_path"])
            prompt_for_scorer = payload.get("prompt", "")
            individual_score = await optimizer.scorer.score(image, prompt_for_scorer, name=payload_name)
            rescored_scores.append(individual_score)
            rescored_weights.append(payload.get("score_weight", 1.0))

            scorer_results = dict(optimizer.scorer.last_scorer_results)
            scorer_results["combined"] = individual_score
            cached["scores"] = scorer_results
            cached["final_score"] = individual_score
            cached["scorer_setup_fp"] = scorer_setup_fp

            try:
                output_dir = Path(HydraConfig.get().runtime.output_dir)
                optimizer.current_run_manifest[image_hash] = build_run_manifest_entry(
                    image_path=Path(cached["full_path"]),
                    output_dir=output_dir,
                    scorer_results=scorer_results,
                    final_score=individual_score,
                    scorer_setup_fp=scorer_setup_fp,
                )
            except Exception:
                optimizer.current_run_manifest[image_hash] = build_run_manifest_entry(
                    image_path=Path(cached["full_path"]),
                    output_dir=None,
                    scorer_results=scorer_results,
                    final_score=individual_score,
                    scorer_setup_fp=scorer_setup_fp,
                )

            payload_entries.append(
                {
                    "name": payload_name,
                    "weight": payload.get("score_weight", 1.0),
                    "scores": scorer_results,
                    "combined": individual_score,
                }
            )
            logger.info("  Re-scored: %.4f", individual_score)
            image.close()
        except Exception as error:
            logger.warning("Failed to re-score cached image: %s", error)
            overall_tier = "full_miss"
            break

    if overall_tier == "partial_hit" and rescored_scores:
        avg_score = average_calc(rescored_scores, rescored_weights, optimizer.cfg.img_average_type)
        optimizer.last_trial_scorer_summary = build_trial_scorer_summary(
            payload_entries,
            final_score=avg_score,
            combine_scores=lambda values, weights: average_calc(values, weights, optimizer.cfg.img_average_type),
        )
        elapsed = time.time() - iteration_start_time
        logger.info(
            "PARTIAL HIT complete: Score: %.4f (re-scored in %.2fs, skipped merge+gen)",
            avg_score,
            elapsed,
        )
        return avg_score

    return None
