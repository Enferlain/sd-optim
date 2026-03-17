from __future__ import annotations

import contextlib
import importlib
import sys
import types
from pathlib import Path

import sd_mecha


def _import_utils_module_with_pynput_stub(module_name: str):
    pynput_mod = types.ModuleType("pynput")
    keyboard_mod = types.ModuleType("pynput.keyboard")
    keyboard_mod.Key = types.SimpleNamespace(ctrl="ctrl", shift="shift", alt="alt")
    pynput_mod.keyboard = keyboard_mod
    sys.modules.setdefault("pynput", pynput_mod)
    sys.modules.setdefault("pynput.keyboard", keyboard_mod)
    return importlib.import_module(module_name)


def test_open_model_graph_root_registers_model_dirs_temporarily(monkeypatch) -> None:
    recipes = _import_utils_module_with_pynput_stub("sd_optim.utils.recipes")

    original_registry = [Path("/existing-model-dir")]
    monkeypatch.setattr(recipes.sd_mecha.extensions.model_dirs, "_registry", original_registry.copy())

    opened_root = object()
    observed_registry: list[Path] = []

    @contextlib.contextmanager
    def fake_open_graph(node, root_only=True):  # noqa: ARG001 - signature mirrors the sd_mecha entry point.
        observed_registry.extend(recipes.sd_mecha.extensions.model_dirs.get_all())
        yield types.SimpleNamespace(root_non_finalized=opened_root)

    monkeypatch.setattr(recipes.sd_mecha, "open_graph", fake_open_graph)

    with recipes.open_model_graph_root("fake-node", [Path("/run-model-dir")]) as root:
        assert root is opened_root

    assert observed_registry == [Path("/existing-model-dir"), Path("/run-model-dir")]
    assert recipes.sd_mecha.extensions.model_dirs.get_all() == original_registry


def test_get_model_config_candidates_reads_open_graph_candidates(monkeypatch) -> None:
    recipes = _import_utils_module_with_pynput_stub("sd_optim.utils.recipes")

    config_a = types.SimpleNamespace(identifier="sdxl-sgm")
    config_b = types.SimpleNamespace(identifier="sdxl-kohya")

    @contextlib.contextmanager
    def fake_open_graph(node, root_only=True):  # noqa: ARG001 - signature mirrors the sd_mecha entry point.
        yield types.SimpleNamespace(
            root_non_finalized=object(),
            root_candidates=lambda: types.SimpleNamespace(model_config=(config_a, config_b)),
        )

    monkeypatch.setattr(recipes.sd_mecha, "open_graph", fake_open_graph)

    candidates = recipes.get_model_config_candidates("fake-node", [Path("/models")])

    assert candidates == (config_a, config_b)


def test_merge_with_model_dirs_uses_registry_and_modern_kwarg(monkeypatch) -> None:
    recipes = _import_utils_module_with_pynput_stub("sd_optim.utils.recipes")

    original_registry = [Path("/existing-model-dir")]
    monkeypatch.setattr(recipes.sd_mecha.extensions.model_dirs, "_registry", original_registry.copy())

    captured: dict[str, object] = {}

    def fake_merge(**kwargs):
        captured.update(kwargs)
        captured["registry_during_merge"] = recipes.sd_mecha.extensions.model_dirs.get_all()
        return "merged"

    monkeypatch.setattr(recipes.sd_mecha, "merge", fake_merge)

    result = recipes.merge_with_model_dirs(
        model_dirs_to_add=[Path("/run-model-dir")],
        recipe="recipe-node",
        output="out.safetensors",
        strict_mandatory_keys=False,
    )

    assert result == "merged"
    assert captured["registry_during_merge"] == [Path("/existing-model-dir"), Path("/run-model-dir")]
    assert captured["strict_mandatory_keys"] is False
    assert "model_dirs" not in captured
    assert "check_mandatory_keys" not in captured
    assert recipes.sd_mecha.extensions.model_dirs.get_all() == original_registry


def test_convert_with_model_dirs_uses_registry_without_old_kwarg(monkeypatch) -> None:
    recipes = _import_utils_module_with_pynput_stub("sd_optim.utils.recipes")

    original_registry = [Path("/existing-model-dir")]
    monkeypatch.setattr(recipes.sd_mecha.extensions.model_dirs, "_registry", original_registry.copy())

    captured: dict[str, object] = {}

    def fake_convert(recipe, config):
        captured["recipe"] = recipe
        captured["config"] = config
        captured["registry_during_convert"] = recipes.sd_mecha.extensions.model_dirs.get_all()
        return "converted"

    monkeypatch.setattr(recipes.sd_mecha, "convert", fake_convert)

    result = recipes.convert_with_model_dirs(
        "recipe-node",
        "target-config",
        model_dirs_to_add=[Path("/run-model-dir")],
    )

    assert result == "converted"
    assert captured["recipe"] == "recipe-node"
    assert captured["config"] == "target-config"
    assert captured["registry_during_convert"] == [Path("/existing-model-dir"), Path("/run-model-dir")]
    assert recipes.sd_mecha.extensions.model_dirs.get_all() == original_registry


def test_build_recipe_cache_map_assigns_shared_cache_to_each_merge_node() -> None:
    recipes = _import_utils_module_with_pynput_stub("sd_optim.utils.recipes")

    inner = sd_mecha.weighted_sum(sd_mecha.literal(1.0), sd_mecha.literal(2.0))
    root = sd_mecha.weighted_sum(inner, sd_mecha.literal(3.0))
    shared_cache: dict[str, object] = {}

    cache_map = recipes.build_recipe_cache_map(root, shared_cache)

    assert all(value is shared_cache for value in cache_map.values())
    weighted_sum_nodes = [node for node in cache_map if node.merge_method.identifier == "weighted_sum"]
    assert len(weighted_sum_nodes) == 2


def test_serialize_recipe_text_uses_registry_and_finalize_flag(monkeypatch) -> None:
    recipes = _import_utils_module_with_pynput_stub("sd_optim.utils.recipes")

    original_registry = [Path("/existing-model-dir")]
    monkeypatch.setattr(recipes.sd_mecha.extensions.model_dirs, "_registry", original_registry.copy())

    captured: dict[str, object] = {}

    def fake_serialize(node, *, finalize=True, output=None):  # noqa: ARG001 - mirrors sd-mecha signature.
        captured["finalize"] = finalize
        captured["registry_during_serialize"] = recipes.sd_mecha.extensions.model_dirs.get_all()
        return "version 0.1.0\n"

    monkeypatch.setattr(recipes.sd_mecha, "serialize", fake_serialize)

    result = recipes.serialize_recipe_text(
        sd_mecha.model("model.safetensors"),
        model_dirs_to_add=[Path("/run-model-dir")],
        finalize=False,
    )

    assert result == "version 0.1.0\n"
    assert captured["finalize"] is False
    assert captured["registry_during_serialize"] == [Path("/existing-model-dir"), Path("/run-model-dir")]
    assert recipes.sd_mecha.extensions.model_dirs.get_all() == original_registry


def test_finalize_recipe_with_model_dirs_uses_registry_and_merge_preferences(monkeypatch) -> None:
    recipes = _import_utils_module_with_pynput_stub("sd_optim.utils.recipes")

    original_registry = [Path("/existing-model-dir")]
    monkeypatch.setattr(recipes.sd_mecha.extensions.model_dirs, "_registry", original_registry.copy())

    captured: dict[str, object] = {}
    finalized_root = object()

    @contextlib.contextmanager
    def fake_open_graph(node, buffer_size_per_dict=0, root_only=False):  # noqa: ARG001 - signature mirrors sd-mecha.
        captured["node"] = node
        captured["registry_during_finalize"] = recipes.sd_mecha.extensions.model_dirs.get_all()

        def _finalize_root(**kwargs):
            captured["finalize_kwargs"] = kwargs
            return finalized_root

        yield types.SimpleNamespace(finalize_root=_finalize_root)

    monkeypatch.setattr(recipes.sd_mecha, "open_graph", fake_open_graph)

    result = recipes.finalize_recipe_with_model_dirs(
        sd_mecha.model("model.safetensors"),
        model_dirs_to_add=[Path("/run-model-dir")],
        check_mandatory_keys=False,
        merge_space_preference=recipes.sd_mecha.extensions.merge_spaces.get_all(),
    )

    assert result is finalized_root
    assert captured["registry_during_finalize"] == [Path("/existing-model-dir"), Path("/run-model-dir")]
    assert captured["finalize_kwargs"]["check_mandatory_keys"] is False
    assert captured["finalize_kwargs"]["model_config_preference"] == ("singleton-mecha",)
    assert recipes.sd_mecha.extensions.model_dirs.get_all() == original_registry


def test_serialize_nodes_for_rewrite_inlines_singleton_literal_value_dict(monkeypatch) -> None:
    artifacts = _import_utils_module_with_pynput_stub("sd_optim.utils.artifacts")

    monkeypatch.setattr(artifacts, "serialize_recipe_text", lambda node, **kwargs: "version 0.1.0\n")
    literal_node = sd_mecha.recipe_nodes.LiteralRecipeNode({"key": 0.75})

    new_lines, replacements = artifacts.serialize_nodes_for_rewrite({"alpha": literal_node})

    assert new_lines == []
    assert replacements == {"alpha": "0.75"}


def test_relativize_model_paths_rewrites_paths_under_base_dir(tmp_path) -> None:
    recipes = _import_utils_module_with_pynput_stub("sd_optim.utils.recipes")

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model_path = models_dir / "nested" / "model.safetensors"
    model_path.parent.mkdir()
    model_path.write_text("fake")

    rewritten = recipes.relativize_model_paths(
        sd_mecha.model(model_path, config="sdxl-sgm"),
        base_dir=models_dir,
    )

    assert rewritten.path == Path("nested/model.safetensors")
    assert rewritten.model_config.identifier == "sdxl-sgm"


def test_converter_finder_visit_literal_traverses_value_dict_nodes() -> None:
    artifacts = _import_utils_module_with_pynput_stub("sd_optim.utils.artifacts")
    finder = artifacts.ConverterFinder()
    child = sd_mecha.weighted_sum(sd_mecha.literal(1.0), sd_mecha.literal(2.0))

    finder.visit_literal(sd_mecha.literal({"child": child}))

    assert child in finder.visited
