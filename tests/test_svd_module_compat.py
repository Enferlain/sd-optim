from __future__ import annotations

import importlib

import pytest


def test_bundled_merge_methods_package_exports_svd_helpers() -> None:
    module = importlib.import_module("sd_optim.extensions.bundled.merge_methods")

    assert callable(module.torch_svd_lowrank)
    assert callable(module.fractional_matrix_power)


def test_top_level_svd_module_is_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("sd_optim.svd")
