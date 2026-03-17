from __future__ import annotations

import importlib
import sys
import types


def _install_pynput_stub() -> None:
    pynput_mod = types.ModuleType("pynput")
    keyboard_mod = types.ModuleType("pynput.keyboard")
    keyboard_mod.Key = types.SimpleNamespace(ctrl="ctrl", shift="shift", alt="alt")
    pynput_mod.keyboard = keyboard_mod
    sys.modules.setdefault("pynput", pynput_mod)
    sys.modules.setdefault("pynput.keyboard", keyboard_mod)


def test_utils_package_root_is_importable_without_eager_submodule_imports() -> None:
    _install_pynput_stub()

    for module_name in [
        "sd_optim.utils",
        "sd_optim.utils.artifacts",
        "sd_optim.utils.config",
        "sd_optim.utils.conversions",
        "sd_optim.utils.hotkeys",
        "sd_optim.utils.images",
        "sd_optim.utils.methods",
        "sd_optim.utils.recipes",
    ]:
        sys.modules.pop(module_name, None)

    utils_package = importlib.import_module("sd_optim.utils")

    assert utils_package.__doc__
    assert "sd_optim.utils.hotkeys" not in sys.modules
    assert "sd_optim.utils.methods" not in sys.modules


def test_utils_submodules_are_importable_directly() -> None:
    _install_pynput_stub()

    modules = [
        importlib.import_module("sd_optim.utils.artifacts"),
        importlib.import_module("sd_optim.utils.config"),
        importlib.import_module("sd_optim.utils.conversions"),
        importlib.import_module("sd_optim.utils.hotkeys"),
        importlib.import_module("sd_optim.utils.images"),
        importlib.import_module("sd_optim.utils.methods"),
        importlib.import_module("sd_optim.utils.recipes"),
    ]

    assert all(module is not None for module in modules)


def test_utils_methods_package_root_dir_still_points_to_sd_optim_root() -> None:
    _install_pynput_stub()

    methods = importlib.import_module("sd_optim.utils.methods")
    package_root = methods._package_root_dir()

    assert (package_root / "merge_methods.py").is_file()
    assert (package_root / "extensions" / "bundled" / "merge_methods").is_dir()
