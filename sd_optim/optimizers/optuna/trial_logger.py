from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TrialLogger:
    """Small JSONL logger used by the Optuna optimizer."""

    def __init__(self) -> None:
        self.log_path: Path | None = None

    def set_path(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            self.log_path.write_text("", encoding="utf-8")
        logger.info("Trial log (.jsonl) will be saved to: %s", self.log_path)

    def log(self, data: dict[str, Any]) -> None:
        if not self.log_path:
            logger.error("Trial logger path not set. Cannot log trial.")
            return
        try:
            with self.log_path.open("a", encoding="utf-8") as handle:
                json.dump(data, handle)
                handle.write("\n")
        except Exception as error:
            logger.error("Failed to write to trial log %s: %s", self.log_path, error)

    def load_trials(self) -> list[dict[str, Any]]:
        if not self.log_path or not self.log_path.exists():
            return []

        trials = []
        try:
            with self.log_path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        trials.append(json.loads(line))
                    except json.JSONDecodeError as error:
                        logger.warning("Skipping invalid line in trial log: %s", error)
            return trials
        except Exception as error:
            logger.error("Failed to load trials log %s: %s", self.log_path, error)
            return []

