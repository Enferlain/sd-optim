"""Lazy exports for scorer model classes.

Avoid eager imports here so optional scorer dependencies can stay uninstalled
until a scorer is actually used.
"""

from importlib import import_module

_LAZY_IMPORTS = {
    "Laion": "sd_optim.models.Laion",
    "CLIPScore": "sd_optim.models.CLIPScore",
}

__all__ = sorted(_LAZY_IMPORTS.keys())


def __getattr__(name: str):
    module_path = _LAZY_IMPORTS.get(name)
    if not module_path:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals().keys()) | set(__all__))
