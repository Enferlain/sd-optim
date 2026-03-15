"""Builtin scorer package.

Top-level package files hold registry/package metadata.
Concrete scorer implementations live in `sd_optim.builtin.scorers.models`.
"""

from .registry import MODEL_DATA, SCORER_CLASS_PATHS, get_scorer_class

__all__ = ["MODEL_DATA", "SCORER_CLASS_PATHS", "get_scorer_class"]
