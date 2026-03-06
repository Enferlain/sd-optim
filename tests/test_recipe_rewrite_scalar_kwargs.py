from __future__ import annotations

import importlib
import sys
import types

import sd_mecha


def _import_utils_with_pynput_stub(monkeypatch):
    """Import sd_optim.utils without requiring an X server for pynput."""
    pynput_mod = types.ModuleType("pynput")
    keyboard_mod = types.ModuleType("pynput.keyboard")
    keyboard_mod.Key = types.SimpleNamespace(ctrl="ctrl", shift="shift", alt="alt")
    pynput_mod.keyboard = keyboard_mod
    monkeypatch.setitem(sys.modules, "pynput", pynput_mod)
    monkeypatch.setitem(sys.modules, "pynput.keyboard", keyboard_mod)
    return importlib.import_module("sd_optim.utils")


def test_recipe_rewrite_inlines_scalar_kwargs_instead_of_aliasing_refs(monkeypatch) -> None:
    utils = _import_utils_with_pynput_stub(monkeypatch)

    new_nodes = {
        "magnitude_ratio": sd_mecha.literal({"model.diffusion_model.out.0.weight": 1.25}, config="sdxl-sgm"),
        "rank_blend": sd_mecha.literal(0.5),
    }

    new_node_strings, param_to_replacement = utils.serialize_nodes_for_rewrite(new_nodes)

    assert param_to_replacement["magnitude_ratio"].startswith("&")
    assert param_to_replacement["rank_blend"] == "0.5"
    assert param_to_replacement["magnitude_ratio"] != param_to_replacement["rank_blend"]

    original_recipe = "\n".join(
        [
            "version 0.1.0",
            'model "a.safetensors" model_config="sdxl-sgm" merge_space="weight"',
            'merge "delta_widen" &0 magnitude_ratio=1.0 rank_blend=0.0',
        ]
    )

    rewritten = utils.rewrite_recipe_text(
        original_recipe_text=original_recipe,
        target_node_idx=1,
        new_node_strings=new_node_strings,
        param_to_replacement=param_to_replacement,
    )

    assert "rank_blend=0.5" in rewritten
    assert "rank_blend=&" not in rewritten
