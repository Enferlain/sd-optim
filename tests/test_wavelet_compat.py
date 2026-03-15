from __future__ import annotations

import sys
import types

import pytest

import sd_optim.merge_methods as merge_methods


class _DummyForward:
    pass


class _DummyInverse:
    pass


def test_load_wavelet_transforms_retries_when_pkg_resources_missing(monkeypatch):
    merge_methods._WAVELET_TRANSFORMS = None
    monkeypatch.delitem(sys.modules, "pkg_resources", raising=False)

    call_count = {"n": 0}

    def fake_import_module(name: str):
        if name == "pkg_resources":
            raise ModuleNotFoundError("No module named 'pkg_resources'", name="pkg_resources")
        assert name == "pytorch_wavelets"
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise ModuleNotFoundError("No module named 'pkg_resources'", name="pkg_resources")
        return types.SimpleNamespace(DWTForward=_DummyForward, DWTInverse=_DummyInverse)

    monkeypatch.setattr(merge_methods, "import_module", fake_import_module)

    dwt_forward, dwt_inverse = merge_methods._load_wavelet_transforms()

    assert dwt_forward is _DummyForward
    assert dwt_inverse is _DummyInverse
    assert call_count["n"] == 2
    assert "pkg_resources" in sys.modules
    assert hasattr(sys.modules["pkg_resources"], "resource_stream")


def test_pkg_resources_shim_resource_stream_can_read_package_file(monkeypatch):
    monkeypatch.delitem(sys.modules, "pkg_resources", raising=False)

    merge_methods._ensure_pkg_resources_resource_stream()

    pkg_resources_mod = sys.modules["pkg_resources"]
    with pkg_resources_mod.resource_stream("sd_optim", "__init__.py") as stream:
        contents = stream.read()

    assert contents


def test_load_wavelet_transforms_propagates_non_pkg_resources_import_errors(monkeypatch):
    merge_methods._WAVELET_TRANSFORMS = None

    def fake_import_module(name: str):
        assert name == "pytorch_wavelets"
        raise ModuleNotFoundError("No module named 'pytorch_wavelets'", name="pytorch_wavelets")

    monkeypatch.setattr(merge_methods, "import_module", fake_import_module)

    with pytest.raises(ModuleNotFoundError, match="pytorch_wavelets"):
        merge_methods._load_wavelet_transforms()


def test_load_wavelet_transforms_raises_clear_error_when_pywt_missing(monkeypatch):
    merge_methods._WAVELET_TRANSFORMS = None

    def fake_import_module(name: str):
        assert name == "pytorch_wavelets"
        raise ModuleNotFoundError("No module named 'pywt'", name="pywt")

    monkeypatch.setattr(merge_methods, "import_module", fake_import_module)

    with pytest.raises(ImportError, match="PyWavelets"):
        merge_methods._load_wavelet_transforms()
