from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from sd_optim.core import optimizer_cache as core_cache


def test_optimizer_core_cache_module_exports_cache_helpers() -> None:
    module = importlib.import_module("sd_optim.core.optimizer_cache")

    assert callable(module.calculate_image_hash)
    assert callable(module.compute_generation_setup_fingerprint)
    assert callable(module.compute_scorer_setup_fingerprint)
    assert callable(module.fail_on_error_enabled)
    assert callable(module.reuse_cached_results_enabled)


def test_core_generation_fingerprint_is_stable_for_equivalent_config() -> None:
    cfg = OmegaConf.create(
        {
            "optimization_mode": "merge",
            "merge_method": "weighted_sum",
            "models_dir": "/models",
            "model_paths": ["model_a.safetensors", "model_b.safetensors"],
            "base_model_index": 0,
            "fallback_model_index": 1,
            "add_extra_keys": False,
            "merge_dtype": "fp32",
            "save_dtype": "bf16",
            "webui": "comfy",
            "recipe_optimization": {
                "recipe_path": "/recipes/example.mecha",
                "target_nodes": "&12",
                "target_params": ["alpha"],
            },
        }
    )

    copied_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    assert core_cache.compute_generation_setup_fingerprint(cfg) == core_cache.compute_generation_setup_fingerprint(
        copied_cfg
    )


def test_core_image_hash_is_deterministic() -> None:
    params = {"alpha": 0.5, "beta": 0.2}
    payload = {
        "prompt": "test",
        "seed": 42,
        "steps": 20,
        "cfg": 7,
        "width": 1024,
        "height": 1024,
    }

    assert core_cache.calculate_image_hash(
        params,
        payload,
        generation_setup_fp="weighted-sum-fingerprint",
    ) == core_cache.calculate_image_hash(
        params,
        payload,
        generation_setup_fp="weighted-sum-fingerprint",
    )


def test_core_fail_on_error_enabled_supports_none_and_attribute_objects() -> None:
    assert core_cache.fail_on_error_enabled(None) is True
    assert core_cache.fail_on_error_enabled(SimpleNamespace(fail_on_error=False)) is False


def test_core_reuse_cached_results_enabled_defaults_true() -> None:
    assert core_cache.reuse_cached_results_enabled(None) is True
    assert core_cache.reuse_cached_results_enabled(SimpleNamespace(reuse_cached_results=False)) is False


def test_core_scorer_setup_fingerprint_changes_with_effective_scorer_config() -> None:
    base_cfg = OmegaConf.create(
        {
            "scorer_method": ["Aesthetic", "MANUAL"],
            "scorer_weight": {"aesthetic": 1.0, "manual": 0.5},
            "scorer_filters": {"manual": {"only": ["portrait"]}},
            "scorer_average_type": "weighted_arithmetic",
            "scorer_model_dir": "/models/scorers",
            "manual_prompt": "rate this",
            "aesthetic_model": "shadow-v1",
        }
    )
    variant_cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    variant_cfg.scorer_weight.manual = 0.75

    assert core_cache.compute_scorer_setup_fingerprint(base_cfg) != core_cache.compute_scorer_setup_fingerprint(
        variant_cfg
    )


def test_core_generation_fingerprint_handles_non_dict_configs() -> None:
    list_cfg = OmegaConf.create(["not", "a", "dict"])
    weird_recipe_cfg = OmegaConf.create(
        {
            "optimization_mode": "recipe",
            "merge_method": "weighted_sum",
            "model_paths": ["relative_model.safetensors"],
            "recipe_optimization": ["unexpected", "shape"],
        }
    )

    assert isinstance(core_cache.compute_generation_setup_fingerprint(list_cfg), str)
    assert isinstance(core_cache.compute_generation_setup_fingerprint(weird_recipe_cfg), str)


def test_core_normalize_model_path_for_fingerprint_resolves_relative_inputs() -> None:
    models_dir = Path("/models")

    normalized_with_base = core_cache.normalize_model_path_for_fingerprint("nested/model.safetensors", models_dir)
    normalized_without_base = core_cache.normalize_model_path_for_fingerprint("nested/model.safetensors", None)

    assert normalized_with_base == str((models_dir / "nested/model.safetensors").resolve())
    assert normalized_without_base == str(Path("nested/model.safetensors").resolve())
