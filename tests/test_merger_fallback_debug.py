from __future__ import annotations

import importlib
import logging
import sys
import types

import sd_mecha
from omegaconf import OmegaConf
from sd_mecha.keys_map import RealizedKeyRelation
from sd_mecha.recipe_nodes import MergeRecipeNode
from sd_mecha.streaming import StateDictKeyError


def _import_merger_with_pynput_stub():
    pynput_mod = types.ModuleType("pynput")
    keyboard_mod = types.ModuleType("pynput.keyboard")
    keyboard_mod.Key = types.SimpleNamespace(ctrl="ctrl", shift="shift", alt="alt")
    pynput_mod.keyboard = keyboard_mod
    sys.modules.setdefault("pynput", pynput_mod)
    sys.modules.setdefault("pynput.keyboard", keyboard_mod)
    return importlib.import_module("sd_optim.merger")


merger_mod = _import_merger_with_pynput_stub()


def _fallback_relation(*params: str) -> RealizedKeyRelation:
    return RealizedKeyRelation(
        outputs=("x.key",),
        inputs=dict.fromkeys(params, ("x.key",)),
        meta=params,
    )


def test_fallback_debug_logged_matches_builtin_key_map() -> None:
    input_config = sd_mecha.extensions.model_configs.resolve("sdxl-sgm")
    output_config = sd_mecha.extensions.model_configs.resolve("sdxl-sgm")

    expected = sd_mecha.fallback.build_key_map((input_config, input_config), {}, output_config)
    actual = merger_mod.fallback_debug_logged.build_key_map((input_config, input_config), {}, output_config)

    sample_key = "model.diffusion_model.output_blocks.5.2.conv.weight"
    assert actual[sample_key] == expected[sample_key]


def test_fallback_debug_logged_emits_info_on_first_fallback_hit(caplog) -> None:
    class MissingStateDict(dict):
        def __getitem__(self, key):
            raise StateDictKeyError(key)

    caplog.set_level(logging.INFO, logger="sd_optim.merger")
    context = merger_mod.fallback_debug_logged.instantiate()
    value = merger_mod.fallback_debug_logged.merge_key(
        [MissingStateDict(), {"x.key": 123}],
        {},
        "x.key",
        _fallback_relation("a", "default"),
        cache=None,
        context=context,
        output_reused=False,
    )

    assert value == 123
    assert any("Fallback hit 1 for key: x.key" in rec.message for rec in caplog.records)


def test_fallback_debug_logged_logs_key_at_debug(caplog) -> None:
    class MissingStateDict(dict):
        def __getitem__(self, key):
            raise StateDictKeyError(key)

    caplog.set_level(logging.DEBUG, logger="sd_optim.merger")
    context = merger_mod.fallback_debug_logged.instantiate()
    value = merger_mod.fallback_debug_logged.merge_key(
        [MissingStateDict(), {"x.key": 456}],
        {},
        "x.key",
        _fallback_relation("a", "default"),
        cache=None,
        context=context,
        output_reused=False,
    )

    assert value == 456
    assert any("Using fallback for key: x.key" in rec.message for rec in caplog.records)


def test_execute_recipe_wraps_fallback_in_recipe_instead_of_sd_mecha_kwarg(monkeypatch, tmp_path) -> None:
    merger = object.__new__(merger_mod.Merger)
    merger.cfg = OmegaConf.create(
        {
            "optimization_mode": "merge",
            "fallback_model_index": 0,
            "device": "cpu",
            "merge_dtype": "fp32",
            "save_dtype": "fp32",
            "threads": 1,
        }
    )
    merger.models = [sd_mecha.model("fallback.safetensors")]
    merger.models_dir = tmp_path

    captured: dict = {}

    def _fake_merge(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(sd_mecha, "merge", _fake_merge)

    merger._execute_recipe(
        final_recipe_node=sd_mecha.model("main.safetensors"),
        model_path=tmp_path / "out.safetensors",
    )

    assert captured["fallback_model"] is None
    assert isinstance(captured["recipe"], MergeRecipeNode)
    assert captured["recipe"].merge_method.identifier == "fallback_debug_logged"


def test_handle_delta_output_uses_bound_args_shape() -> None:
    merger = object.__new__(merger_mod.Merger)
    base_model = sd_mecha.model("base.safetensors")
    delta_node = sd_mecha.subtract(sd_mecha.model("other.safetensors"), base_model)

    wrapped = merger._handle_delta_output(delta_node, base_model, delta_node.merge_method)

    assert isinstance(wrapped, MergeRecipeNode)
    assert wrapped.merge_method.identifier == "add_difference"


def test_serialize_and_save_recipe_uses_finalized_execution_recipe(monkeypatch, tmp_path) -> None:
    merger = object.__new__(merger_mod.Merger)
    merger.cfg = OmegaConf.create(
        {
            "optimization_mode": "merge",
            "fallback_model_index": 0,
            "save_dtype": "fp32",
        }
    )
    merger.models = [sd_mecha.model("fallback.safetensors")]
    merger.models_dir = tmp_path / "models"
    merger.models_dir.mkdir()
    finalized_model_path = merger.models_dir / "finalized.safetensors"
    finalized_model_path.write_text("fake")

    captured: dict[str, object] = {}
    finalized_recipe = sd_mecha.model(finalized_model_path, config="sdxl-sgm")

    def fake_finalize_recipe_with_model_dirs(node, *, model_dirs_to_add=(), **kwargs):
        captured["finalize_node"] = node
        captured["finalize_model_dirs"] = list(model_dirs_to_add)
        captured["finalize_kwargs"] = kwargs
        return finalized_recipe

    def fake_serialize_recipe_text(node, *, model_dirs_to_add=(), finalize=False):
        captured["serialize_node"] = node
        captured["serialize_model_dirs"] = list(model_dirs_to_add)
        captured["serialize_finalize"] = finalize
        return "version 0.1.0\n"

    monkeypatch.setattr(merger_mod.utils, "finalize_recipe_with_model_dirs", fake_finalize_recipe_with_model_dirs)
    monkeypatch.setattr(merger_mod.utils, "serialize_recipe_text", fake_serialize_recipe_text)

    recipe_node = sd_mecha.model("relative-merged.safetensors")
    merger._serialize_and_save_recipe(recipe_node, tmp_path / "merged.safetensors")

    assert isinstance(captured["finalize_node"], MergeRecipeNode)
    assert captured["finalize_node"].merge_method.identifier == "fallback_debug_logged"
    assert captured["finalize_model_dirs"] == [merger.models_dir]
    assert captured["serialize_node"] is not finalized_recipe
    assert captured["serialize_node"].path == finalized_model_path.relative_to(merger.models_dir)
    assert captured["serialize_model_dirs"] == [merger.models_dir]
    assert captured["serialize_finalize"] is False
