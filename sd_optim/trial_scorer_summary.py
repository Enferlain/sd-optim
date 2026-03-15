from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_trial_scorer_summary(
    payload_entries: list[dict[str, Any]],
    *,
    final_score: float,
    combine_scores: Any,
) -> dict[str, Any]:
    """Build an aggregate scorer summary for a trial.

    The payload list is preserved verbatim so callers can store per-payload
    metadata, while the aggregate section combines each scorer across payloads
    using the caller's weighting strategy.
    """
    payloads = [dict(entry) for entry in payload_entries]
    aggregate: dict[str, float] = {}

    scorer_names = sorted(
        {
            str(scorer_name)
            for entry in payload_entries
            for scorer_name in (entry.get("scores") or {})
        }
    )

    for scorer_name in scorer_names:
        values: list[float] = []
        weights: list[float] = []

        for entry in payload_entries:
            scores = entry.get("scores")
            if not isinstance(scores, dict) or scorer_name not in scores:
                continue

            try:
                value = float(scores[scorer_name])
                weight = float(entry.get("weight", 1.0))
            except (TypeError, ValueError) as error:
                logger.warning(
                    "Skipping invalid scorer summary entry for '%s': %s",
                    scorer_name,
                    error,
                )
                continue

            values.append(value)
            weights.append(weight)

        if values:
            aggregate[scorer_name] = float(combine_scores(values, weights))

    aggregate["combined"] = float(final_score)

    return {
        "aggregate": aggregate,
        "payloads": payloads,
    }
