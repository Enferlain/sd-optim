from __future__ import annotations

import importlib


def test_svd_compat_module_exports_lowrank_helper() -> None:
    module = importlib.import_module("sd_optim.svd")

    assert callable(module.torch_svd_lowrank)
