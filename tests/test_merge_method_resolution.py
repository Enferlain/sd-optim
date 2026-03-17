import importlib
import types

import pytest
import sd_mecha
from torch import Tensor

from sd_optim.utils import methods


def _make_merge_method(identifier: str):
    @sd_mecha.merge_method(identifier=identifier, register=False)
    def merge(
        a: sd_mecha.Parameter(Tensor, "weight"),
        b: sd_mecha.Parameter(Tensor, "weight"),
    ) -> sd_mecha.Return(Tensor, "weight"):
        return a

    return merge


def test_resolve_merge_method_only_imports_requested_bundled_module(monkeypatch) -> None:
    monkeypatch.setattr(methods, "_get_bundled_merge_method_index", lambda: {"wanted": ["pkg.good"], "broken": ["pkg.bad"]})
    monkeypatch.setattr(sd_mecha.extensions.merge_methods, "resolve", lambda name: (_ for _ in ()).throw(ValueError(name)))

    imported: list[str] = []

    def fake_import_module(name: str):
        imported.append(name)
        if name == "pkg.bad":
            raise AssertionError("Unrelated broken module should not be imported")
        if name == "pkg.good":
            module = types.SimpleNamespace()
            module.wanted = _make_merge_method("wanted")
            return module
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    resolved = methods.resolve_merge_method("wanted")

    assert resolved.identifier == "wanted"
    assert imported == ["pkg.good"]


def test_resolve_merge_method_supports_class_based_bundled_module(monkeypatch) -> None:
    monkeypatch.setattr(methods, "_get_bundled_merge_method_index", lambda: {"classy": ["pkg.classy"]})
    monkeypatch.setattr(sd_mecha.extensions.merge_methods, "resolve", lambda name: (_ for _ in ()).throw(ValueError(name)))

    def fake_import_module(name: str):
        assert name == "pkg.classy"

        merge_func = _make_merge_method("classy")

        class MergeMethods:
            classy = merge_func

        return types.SimpleNamespace(MergeMethods=MergeMethods)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    resolved = methods.resolve_merge_method("classy")

    assert resolved.identifier == "classy"


def test_resolve_merge_method_only_fails_for_requested_broken_module(monkeypatch) -> None:
    monkeypatch.setattr(methods, "_get_bundled_merge_method_index", lambda: {"broken": ["pkg.broken"]})
    monkeypatch.setattr(sd_mecha.extensions.merge_methods, "resolve", lambda name: (_ for _ in ()).throw(ValueError(name)))
    monkeypatch.setattr(importlib, "import_module", lambda name: (_ for _ in ()).throw(ImportError("boom")))

    with pytest.raises(ValueError, match="broken"):
        methods.resolve_merge_method("broken")
