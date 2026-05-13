from __future__ import annotations

from omegaconf import OmegaConf

from sd_optim.core.optimizer_cache import calculate_image_hash, compute_generation_setup_fingerprint


def test_generation_setup_fingerprint_changes_with_merge_method() -> None:
    base_cfg = OmegaConf.create(
        {
            "optimization_mode": "merge",
            "paths": {"models_dir": "/models"},
            "merge": {
                "merge_method": "weighted_sum",
                "model_paths": ["model_a.safetensors", "model_b.safetensors"],
                "base_model_index": 0,
                "fallback_model_index": 1,
                "add_extra_keys": False,
                "merge_dtype": "fp32",
                "save_dtype": "bf16",
            },
            "webui": "comfy",
            "recipe_optimization": {
                "recipe_path": "/recipes/example.mecha",
                "target_nodes": "&12",
                "target_params": ["alpha"],
            },
        }
    )

    variant_cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    variant_cfg.merge.merge_method = "ties_sum"

    assert compute_generation_setup_fingerprint(base_cfg) != compute_generation_setup_fingerprint(variant_cfg)


def test_image_hash_changes_when_generation_setup_changes() -> None:
    params = {"alpha": 0.5, "beta": 0.2}
    payload = {
        "prompt": "test",
        "seed": 42,
        "steps": 20,
        "cfg": 7,
        "width": 1024,
        "height": 1024,
    }

    weighted_sum_hash = calculate_image_hash(
        params,
        payload,
        generation_setup_fp="weighted-sum-fingerprint",
    )
    ties_sum_hash = calculate_image_hash(
        params,
        payload,
        generation_setup_fp="ties-sum-fingerprint",
    )

    assert weighted_sum_hash != ties_sum_hash
