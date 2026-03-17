import importlib
import logging
import re

from pathlib import Path
from typing import Any

import sd_mecha
from sd_mecha.extensions import merge_methods

logger = logging.getLogger(__name__)

_BUNDLED_MERGE_METHOD_INDEX: dict[str, list[str]] | None = None
_LEGACY_MERGE_METHOD_NAMES: set[str] | None = None


def _package_root_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def _scan_merge_method_names(source: str) -> set[str]:
    names: set[str] = set()
    lines = source.splitlines()
    for index, line in enumerate(lines):
        if "@merge_method" not in line:
            continue

        scan_index = index + 1
        while scan_index < len(lines) and lines[scan_index].strip().startswith("@"):
            scan_index += 1

        if scan_index >= len(lines):
            continue

        match = re.match(r"\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", lines[scan_index])
        if match:
            names.add(match.group(1))

    return names


def _module_path_from_file(path: Path) -> str:
    relative = path.relative_to(_package_root_dir()).with_suffix("")
    return ".".join(("sd_optim", *relative.parts))


def _get_bundled_merge_method_index() -> dict[str, list[str]]:
    global _BUNDLED_MERGE_METHOD_INDEX
    if _BUNDLED_MERGE_METHOD_INDEX is not None:
        return _BUNDLED_MERGE_METHOD_INDEX

    methods_dir = _package_root_dir() / "extensions" / "bundled" / "merge_methods"
    index: dict[str, list[str]] = {}

    for path in methods_dir.rglob("*.py"):
        if path.name == "__init__.py":
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as error:
            logger.warning("Could not read merge-method module %s while building index: %s", path, error)
            continue

        module_path = _module_path_from_file(path)
        for method_name in _scan_merge_method_names(source):
            index.setdefault(method_name, []).append(module_path)

    _BUNDLED_MERGE_METHOD_INDEX = index
    return index


def _get_legacy_merge_method_names() -> set[str]:
    global _LEGACY_MERGE_METHOD_NAMES
    if _LEGACY_MERGE_METHOD_NAMES is not None:
        return _LEGACY_MERGE_METHOD_NAMES

    legacy_path = _package_root_dir() / "merge_methods.py"
    try:
        source = legacy_path.read_text(encoding="utf-8")
    except OSError as error:
        logger.warning("Could not read legacy merge_methods.py while building index: %s", error)
        _LEGACY_MERGE_METHOD_NAMES = set()
        return _LEGACY_MERGE_METHOD_NAMES

    _LEGACY_MERGE_METHOD_NAMES = _scan_merge_method_names(source)
    return _LEGACY_MERGE_METHOD_NAMES


def _wrap_merge_method_callable(candidate: Any, merge_method_name: str) -> merge_methods.MergeMethod | None:
    if isinstance(candidate, staticmethod | classmethod):
        candidate = candidate.__func__

    if isinstance(candidate, merge_methods.MergeMethod):
        return candidate

    if callable(candidate):
        try:
            wrapped_func = sd_mecha.merge_method(candidate, identifier=merge_method_name, register=False)
            logger.warning(
                "Manually wrapping merge method '%s'. Decorate with @merge_method for proper registration.",
                merge_method_name,
            )
            return wrapped_func
        except Exception as wrap_error:
            logger.error(
                "Requested merge method '%s' exists but could not be wrapped as an sd-mecha method: %s",
                merge_method_name,
                wrap_error,
            )
            return None

    return None


def _extract_merge_method_from_namespace(namespace: Any, merge_method_name: str) -> merge_methods.MergeMethod | None:
    if not hasattr(namespace, merge_method_name):
        return None
    return _wrap_merge_method_callable(getattr(namespace, merge_method_name), merge_method_name)


def _resolve_bundled_merge_method(merge_method_name: str) -> merge_methods.MergeMethod | None:
    index = _get_bundled_merge_method_index()
    candidates = index.get(merge_method_name, [])

    for module_path in candidates:
        try:
            module = importlib.import_module(module_path)
        except Exception as import_error:
            raise ImportError(
                f"Failed to import requested merge method '{merge_method_name}' from '{module_path}': {import_error}"
            ) from import_error

        merge_func = _extract_merge_method_from_namespace(module, merge_method_name)
        if merge_func is not None:
            logger.debug("Resolved merge method '%s' from bundled module '%s'.", merge_method_name, module_path)
            return merge_func

        merge_methods_class = getattr(module, "MergeMethods", None)
        if merge_methods_class is not None:
            merge_func = _extract_merge_method_from_namespace(merge_methods_class, merge_method_name)
            if merge_func is not None:
                logger.debug(
                    "Resolved merge method '%s' from class-based bundled module '%s'.",
                    merge_method_name,
                    module_path,
                )
                return merge_func

    return None


def _resolve_legacy_merge_method(merge_method_name: str) -> merge_methods.MergeMethod | None:
    if merge_method_name not in _get_legacy_merge_method_names():
        return None

    try:
        legacy_module = importlib.import_module("sd_optim.merge_methods")
    except Exception as import_error:
        raise ImportError(
            f"Failed to import legacy merge method surface for '{merge_method_name}': {import_error}"
        ) from import_error

    merge_methods_class = getattr(legacy_module, "MergeMethods", None)
    if merge_methods_class is None:
        return _extract_merge_method_from_namespace(legacy_module, merge_method_name)

    merge_func = _extract_merge_method_from_namespace(merge_methods_class, merge_method_name)
    if merge_func is not None:
        logger.debug("Resolved merge method '%s' from legacy merge_methods.py.", merge_method_name)
    return merge_func


def resolve_merge_method(merge_method_name: str) -> merge_methods.MergeMethod:
    """
    Resolve a merge method without importing unrelated custom modules.
    """
    try:
        merge_func = _resolve_bundled_merge_method(merge_method_name)
        if merge_func is not None:
            return merge_func

        merge_func = _resolve_legacy_merge_method(merge_method_name)
        if merge_func is not None:
            return merge_func

        merge_func = sd_mecha.extensions.merge_methods.resolve(merge_method_name)
        logger.debug("Resolved merge method '%s' from sd-mecha built-ins.", merge_method_name)
        return merge_func
    except ImportError as import_error:
        logger.error("Failed to load requested merge method '%s': %s", merge_method_name, import_error)
        raise ValueError(f"Failed to load merge method '{merge_method_name}'.") from import_error
    except ValueError as lookup_error:
        logger.error("Merge method '%s' could not be resolved: %s", merge_method_name, lookup_error)
        raise ValueError(f"Merge method '{merge_method_name}' not found.") from lookup_error
