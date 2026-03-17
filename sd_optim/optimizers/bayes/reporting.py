from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def log_postprocess_summary(results: list[dict[str, Any]], optimization_start_time: float | None) -> None:
    """Log high-level BayesOpt recap information."""
    total_trials = len(results)
    logger.info("Total Trials Run: %s", total_trials)

    if optimization_start_time is not None:
        total_runtime = time.time() - optimization_start_time
        hours, remainder = divmod(total_runtime, 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info("Total runtime: %02d:%02d:%02d", int(hours), int(minutes), int(seconds))
        if total_trials > 0:
            logger.info("Average time per trial: %.2f seconds", total_runtime / total_trials)

    logger.info("\nTop 5 BayesOpt Trials:")
    sorted_results = sorted(results, key=lambda item: item["target"], reverse=True)
    for index, result in enumerate(sorted_results[:5], start=1):
        logger.info("Rank %s:", index)
        logger.info("\tTarget: %.4f", result["target"])
        param_str = ", ".join(
            f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in result["params"].items()
        )
        logger.info("\tParams: {%s}", param_str)


def build_artist_series(results: list[dict[str, Any]]) -> tuple[list[int], list[float], list[float]]:
    """Build trial, score, and rolling-best series for visualization."""
    iterations = list(range(1, len(results) + 1))
    scores = [result["target"] for result in results]

    best_scores: list[float] = []
    current_best = -float("inf")
    for result in results:
        current_best = max(current_best, result["target"])
        best_scores.append(current_best)

    return iterations, scores, best_scores
