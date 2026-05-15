from __future__ import annotations

from types import SimpleNamespace

import pytest
import sd_mecha
import torch
from omegaconf import OmegaConf


def test_sanitize_recipe_text_for_deserialize_drops_blank_lines() -> None:
    from sd_optim.merge.recipe_optimization import sanitize_recipe_text_for_deserialize

    recipe_text = "version 0.1.0\n\nmodel \"a.safetensors\"\n   \nmerge \"weighted_sum\" &0 alpha=0.5\n"

    assert sanitize_recipe_text_for_deserialize(recipe_text) == (
        "version 0.1.0\nmodel \"a.safetensors\"\nmerge \"weighted_sum\" &0 alpha=0.5"
    )


def test_resolve_layer_adjust_model_path_falls_back_to_models_dir(tmp_path) -> None:
    from sd_optim.merge.layer_adjust import resolve_layer_adjust_model_path

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model_path = models_dir / "base.safetensors"
    model_path.write_text("fake", encoding="utf-8")

    resolved = resolve_layer_adjust_model_path(
        models_dir=models_dir,
        model_path_str="base.safetensors",
    )

    assert resolved == model_path


def test_detect_sdxl_model_looks_for_conditioner_embedder_key() -> None:
    from sd_optim.merge.layer_adjust import detect_is_xl_model

    assert detect_is_xl_model({"conditioner.embedders.1.model.weight": torch.tensor(1.0)}) is True
    assert detect_is_xl_model({"model.diffusion_model.out.0.weight": torch.tensor(1.0)}) is False


def test_layer_adjust_reraises_known_load_errors(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from sd_optim.merge import layer_adjust as layer_adjust_mod

    output_path = tmp_path / "adjusted.safetensors"
    merger = SimpleNamespace(output_file=output_path, models_dir=tmp_path)
    cfg = OmegaConf.create(
        {
            "merge": {"model_paths": ["base.safetensors"], "device": "cpu"},
            "paths": {"models_dir": str(tmp_path)},
        }
    )

    monkeypatch.setattr(layer_adjust_mod, "resolve_layer_adjust_model_path", lambda **kwargs: tmp_path / "base.safetensors")
    monkeypatch.setattr(layer_adjust_mod, "load_layer_adjust_state_dict", lambda model_path, device: (_ for _ in ()).throw(OSError("boom")))

    with pytest.raises(OSError, match="boom"):
        layer_adjust_mod.layer_adjust(merger, params={}, cfg=cfg)


def test_layer_adjust_reraises_known_mutation_errors(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from sd_optim.merge import layer_adjust as layer_adjust_mod

    output_path = tmp_path / "adjusted.safetensors"
    merger = SimpleNamespace(output_file=output_path, models_dir=tmp_path)
    cfg = OmegaConf.create(
        {
            "merge": {"model_paths": ["base.safetensors"], "device": "cpu"},
            "paths": {"models_dir": str(tmp_path)},
        }
    )

    monkeypatch.setattr(layer_adjust_mod, "resolve_layer_adjust_model_path", lambda **kwargs: tmp_path / "base.safetensors")
    monkeypatch.setattr(layer_adjust_mod, "load_layer_adjust_state_dict", lambda model_path, device: {"a": torch.tensor(1.0)})
    monkeypatch.setattr(layer_adjust_mod, "modify_state_dict", lambda state_dict, params, is_xl_model: (_ for _ in ()).throw(KeyError("bad key")))

    with pytest.raises(KeyError, match="bad key"):
        layer_adjust_mod.layer_adjust(merger, params={}, cfg=cfg)


def test_layer_adjust_reraises_known_save_errors(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from sd_optim.merge import layer_adjust as layer_adjust_mod

    output_path = tmp_path / "nested" / "adjusted.safetensors"
    merger = SimpleNamespace(output_file=output_path, models_dir=tmp_path)
    cfg = OmegaConf.create(
        {
            "merge": {"model_paths": ["base.safetensors"], "device": "cpu"},
            "paths": {"models_dir": str(tmp_path)},
        }
    )

    monkeypatch.setattr(layer_adjust_mod, "resolve_layer_adjust_model_path", lambda **kwargs: tmp_path / "base.safetensors")
    monkeypatch.setattr(layer_adjust_mod, "load_layer_adjust_state_dict", lambda model_path, device: {"a": torch.tensor(1.0)})
    monkeypatch.setattr(layer_adjust_mod, "modify_state_dict", lambda state_dict, params, is_xl_model: {"a": torch.tensor(1.0)})
    monkeypatch.setattr(layer_adjust_mod.safetensors.torch, "save_file", lambda state_dict, path: (_ for _ in ()).throw(OSError("write failed")))

    with pytest.raises(OSError, match="write failed"):
        layer_adjust_mod.layer_adjust(merger, params={}, cfg=cfg)


def test_recipe_optimization_rejects_non_merge_target(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from sd_optim.merge import recipe_optimization as recipe_mod

    recipe_path = tmp_path / "test.mecha"
    recipe_path.write_text(
        "\n".join(
            [
                "version 0.1.0",
                'model "a.safetensors"',
            ]
        ),
        encoding="utf-8",
    )

    merger = SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "recipe_optimization": {
                    "recipe_path": str(recipe_path),
                    "target_nodes": "&0",
                    "target_params": ["alpha"],
                }
            }
        )
    )

    monkeypatch.setattr(recipe_mod.sd_mecha, "deserialize", lambda lines: sd_mecha.model("a.safetensors"))

    with pytest.raises(TypeError, match="is not a merge node"):
        recipe_mod.load_validated_target_node(merger, recipe_path.read_text(encoding="utf-8"))


def test_run_merge_uses_default_output_path_when_merger_has_none(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from sd_optim.merge import runtime as runtime_mod

    merger = SimpleNamespace(
        cfg=OmegaConf.create({"merge": {"merge_method": "weighted_sum"}}),
        output_file=None,
        models_dir=tmp_path,
        models=["model-a", "model-b"],
    )

    merge_func_calls: dict[str, object] = {}

    def fake_merge_func(*args, **kwargs):
        merge_func_calls["args"] = args
        merge_func_calls["kwargs"] = kwargs
        return "core-recipe"

    fake_merge_func.identifier = "weighted_sum"

    monkeypatch.setattr(runtime_mod, "resolve_merge_method", lambda name: fake_merge_func if name == "weighted_sum" else None)
    monkeypatch.setattr(runtime_mod, "select_base_model", lambda merger_arg: SimpleNamespace(path="base-model"))
    monkeypatch.setattr(runtime_mod, "prepare_model_recipe_args", lambda merger_arg, models, base_model, merge_func: ["prepared-a", "prepared-b"])
    monkeypatch.setattr(runtime_mod, "slice_models", lambda merger_arg, prepared_nodes, merge_func: ["prepared-a"])
    monkeypatch.setattr(runtime_mod, "prepare_param_recipe_args", lambda merger_arg, params, param_info, merge_func: {"alpha": 0.5})
    monkeypatch.setattr(runtime_mod, "handle_delta_output", lambda merger_arg, core_recipe, base_model, merge_func: "final-recipe")
    monkeypatch.setattr(runtime_mod, "build_recipe_cache_map", lambda final_recipe, cache: {"cache": "map", "input_cache": cache})

    saved: dict[str, object] = {}
    executed: dict[str, object] = {}
    monkeypatch.setattr(
        runtime_mod,
        "save_recipe_artifacts",
        lambda merger_arg, final_recipe, model_path, iteration: saved.update(
            {"merger": merger_arg, "final_recipe": final_recipe, "model_path": model_path, "iteration": iteration}
        ),
    )
    monkeypatch.setattr(
        runtime_mod,
        "execute_recipe",
        lambda merger_arg, final_recipe, model_path, cache_map=None: executed.update(
            {"merger": merger_arg, "final_recipe": final_recipe, "model_path": model_path, "cache_map": cache_map}
        ),
    )

    output_path = runtime_mod.run_merge(merger, params={"alpha": 0.25}, param_info={"ignored": True}, cache=None, iteration=9)

    assert output_path == tmp_path / "merge_output_default_weighted_sum.safetensors"
    assert merger.output_file == output_path
    assert merge_func_calls == {"args": ("prepared-a",), "kwargs": {"alpha": 0.5}}
    assert saved["model_path"] == output_path
    assert saved["iteration"] == 9
    assert executed["model_path"] == output_path
    assert executed["cache_map"] == {"cache": "map", "input_cache": {}}


def test_merger_merge_delegates_to_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import sd_optim.merger as merger_mod

    merger = merger_mod.Merger.__new__(merger_mod.Merger)
    merger.cfg = OmegaConf.create({})
    merger.models_dir = tmp_path

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        merger_mod,
        "run_merge",
        lambda merger_arg, params, param_info, cache, iteration: captured.update(
            {
                "merger": merger_arg,
                "params": params,
                "param_info": param_info,
                "cache": cache,
                "iteration": iteration,
            }
        )
        or (tmp_path / "delegated.safetensors"),
    )

    output = merger.merge({"alpha": 1.0}, {"alpha": {"target": "value"}}, {"cached": True}, iteration=3)

    assert output == tmp_path / "delegated.safetensors"
    assert captured == {
        "merger": merger,
        "params": {"alpha": 1.0},
        "param_info": {"alpha": {"target": "value"}},
        "cache": {"cached": True},
        "iteration": 3,
    }
