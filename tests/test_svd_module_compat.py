from __future__ import annotations

import importlib


def test_builtin_merge_methods_package_exports_svd_helpers() -> None:
    module = importlib.import_module("sd_optim.builtin.merge_methods")

    assert callable(module.torch_svd_lowrank)
    assert callable(module.fractional_matrix_power)


def test_svd_compat_module_exports_lowrank_helper() -> None:
    module = importlib.import_module("sd_optim.svd")
    builtin_module = importlib.import_module("sd_optim.builtin.merge_methods")

    assert callable(module.torch_svd_lowrank)
    assert module.torch_svd_lowrank is builtin_module.torch_svd_lowrank
