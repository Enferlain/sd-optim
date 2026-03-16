"""Lazy class resolution for scorer implementations."""

from __future__ import annotations

import importlib
import logging
from functools import cache
from typing import Any

from .catalog import SCORER_CLASS_PATHS

logger = logging.getLogger(__name__)


@cache
def _import_attr(module_path: str, attr_name: str) -> Any | None:
    try:
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    except ImportError as exc:
        logger.warning(
            "Optional scorer dependency missing while importing %s.%s: %s",
            module_path,
            attr_name,
            exc,
        )
    except AttributeError:
        logger.error(
            "Scorer class '%s' not found in module '%s'.",
            attr_name,
            module_path,
        )
    except Exception as exc:
        logger.error(
            "Unexpected error while importing %s.%s: %s",
            module_path,
            attr_name,
            exc,
        )
    return None


def get_scorer_class(scorer_name: str) -> Any | None:
    """Resolve a scorer class lazily by configured scorer name."""
    class_path = SCORER_CLASS_PATHS.get(scorer_name.lower())
    if class_path is None:
        return None
    module_path, attr_name = class_path
    return _import_attr(module_path, attr_name)


__all__ = ["get_scorer_class"]
