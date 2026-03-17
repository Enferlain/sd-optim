from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def load_previous_iterations(
    cfg: DictConfig,
    *,
    target_log_path: Path,
) -> list[dict[str, Any]]:
    """Load prior Bayes iterations and optionally copy them into the new run log."""
    bayes_config = cfg.optimizer.get("bayes_config", {})
    load_log_path = bayes_config.get("load_log_file")
    if not load_log_path:
        return []

    load_log_file = Path(load_log_path)
    if not load_log_file.is_file():
        logger.info("No previous log file found at %s", load_log_file)
        return []

    try:
        with open(load_log_file, encoding="utf-8") as file:
            previous_iterations = [json.loads(line) for line in file if line.strip()]
    except Exception as error:
        logger.warning("Failed to load previous optimization data from %s: %s", load_log_file, error)
        return []

    if bayes_config.get("reset_log_file", False):
        logger.info(
            "Loaded %s iterations from %s but reset_log_file is enabled.",
            len(previous_iterations),
            load_log_file,
        )
        return previous_iterations

    with open(target_log_path, "w", encoding="utf-8") as file:
        for iteration_data in previous_iterations:
            file.write(json.dumps(iteration_data) + "\n")
    logger.info("Loaded and transferred %s iterations from %s", len(previous_iterations), load_log_file)
    return previous_iterations


def register_previous_iterations(
    optimizer: Any,
    previous_iterations: list[dict[str, Any]],
) -> int:
    """Register prior optimizer points, skipping duplicates when the backend exposes a check."""
    if not previous_iterations:
        return 0

    loaded_count = 0
    params_registered = getattr(getattr(optimizer, "space", None), "params_registered", None)

    for point in previous_iterations:
        params = point.get("params")
        target = point.get("target")
        if params is None or target is None:
            logger.warning("Skipping malformed previous Bayes iteration: %s", point)
            continue

        if callable(params_registered):
            try:
                if params_registered(params):
                    continue
            except Exception:
                logger.debug("Duplicate-check probe failed for params %s; proceeding with registration.", params, exc_info=True)

        optimizer.register(params=params, target=target)
        loaded_count += 1

    return loaded_count
