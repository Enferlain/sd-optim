from __future__ import annotations

from typing import Any


VALID_OPTIMIZATION_MODES = {"merge", "recipe", "layer_adjust"}
VALID_PRECISIONS = {"fp16", "bf16", "fp32", "fp64"}


def validate_config(cfg: Any) -> None:
    """Validate cross-field runtime config semantics not expressible by dataclass types."""
    optimizer_cfg = cfg.optimizer
    selected_optimizers = [
        name
        for name in ("bayes", "optuna")
        if bool(getattr(optimizer_cfg, name))
    ]
    if len(selected_optimizers) != 1:
        raise ValueError(
            "Exactly one optimizer must be enabled under optimizer: "
            f"got {selected_optimizers or 'none'}."
        )

    if cfg.optimization_mode not in VALID_OPTIMIZATION_MODES:
        raise ValueError(f"Invalid optimization_mode: '{cfg.optimization_mode}'")

    if cfg.merge.merge_dtype not in VALID_PRECISIONS:
        raise ValueError(f"Invalid 'merge_dtype': '{cfg.merge.merge_dtype}'. Must be one of {sorted(VALID_PRECISIONS)}")
    if cfg.merge.save_dtype not in VALID_PRECISIONS:
        raise ValueError(f"Invalid 'save_dtype': '{cfg.merge.save_dtype}'. Must be one of {sorted(VALID_PRECISIONS)}")
