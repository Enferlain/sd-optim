from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def fail_on_error_enabled(cfg: Any) -> bool:
    """Return whether runtime errors should stop the optimization immediately."""
    if cfg is None:
        return True
    if hasattr(cfg, "get"):
        return bool(cfg.get("fail_on_error", True))
    return bool(getattr(cfg, "fail_on_error", True))


def compute_scorer_setup_fingerprint(cfg: DictConfig) -> str:
    """
    Fingerprint the scoring objective so cached `final_score` is only reused when
    scorer configuration is effectively identical.
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        cfg_dict = {}

    scorer_method_raw = cfg_dict.get("scorer_method", []) or []
    scorer_method = [str(s).lower() for s in scorer_method_raw]

    scorer_weight_raw = cfg_dict.get("scorer_weight", {}) or {}
    if not isinstance(scorer_weight_raw, dict):
        scorer_weight_raw = {}
    scorer_weight = {name: scorer_weight_raw.get(name, scorer_weight_raw.get(name.lower(), 1.0)) for name in scorer_method}

    scorer_filters_raw = cfg_dict.get("scorer_filters", {}) or {}
    if not isinstance(scorer_filters_raw, dict):
        scorer_filters_raw = {}
    scorer_filters = {name: scorer_filters_raw.get(name, scorer_filters_raw.get(name.lower(), {})) for name in scorer_method}

    per_scorer_cfg: dict[str, dict[str, Any]] = {}
    for name in scorer_method:
        prefix = f"{name}_"
        per_scorer_cfg[name] = {k: v for k, v in cfg_dict.items() if isinstance(k, str) and k.lower().startswith(prefix)}

    fingerprint_input = {
        "v": 1,
        "scorer_method": scorer_method,
        "scorer_average_type": cfg_dict.get("scorer_average_type"),
        "scorer_weight": scorer_weight,
        "scorer_filters": scorer_filters,
        "per_scorer_cfg": per_scorer_cfg,
        "scorer_model_dir": cfg_dict.get("scorer_model_dir"),
    }
    recipe_json = json.dumps(fingerprint_input, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(recipe_json.encode("utf-8")).hexdigest()


def compute_generation_setup_fingerprint(cfg: DictConfig) -> str:
    """Fingerprint the generation/merge setup used to produce images for reuse."""
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        cfg_dict = {}

    models_dir_raw = cfg_dict.get("models_dir")
    models_dir = Path(models_dir_raw).resolve() if models_dir_raw else None

    model_paths_raw = cfg_dict.get("model_paths", []) or []
    normalized_model_paths = [
        normalize_model_path_for_fingerprint(model_path, models_dir)
        for model_path in model_paths_raw
    ]

    recipe_optimization_raw = cfg_dict.get("recipe_optimization", {}) or {}
    if not isinstance(recipe_optimization_raw, dict):
        recipe_optimization_raw = {}

    recipe_path_raw = recipe_optimization_raw.get("recipe_path")
    recipe_path = str(Path(recipe_path_raw).resolve()) if recipe_path_raw else None

    fingerprint_input = {
        "v": 1,
        "optimization_mode": cfg_dict.get("optimization_mode"),
        "merge_method": cfg_dict.get("merge_method"),
        "model_paths": normalized_model_paths,
        "base_model_index": cfg_dict.get("base_model_index"),
        "fallback_model_index": cfg_dict.get("fallback_model_index"),
        "add_extra_keys": cfg_dict.get("add_extra_keys"),
        "merge_dtype": cfg_dict.get("merge_dtype"),
        "save_dtype": cfg_dict.get("save_dtype"),
        "webui": cfg_dict.get("webui"),
        "recipe_optimization": {
            "recipe_path": recipe_path,
            "target_nodes": recipe_optimization_raw.get("target_nodes"),
            "target_params": recipe_optimization_raw.get("target_params"),
        },
    }
    recipe_json = json.dumps(fingerprint_input, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(recipe_json.encode("utf-8")).hexdigest()


def normalize_model_path_for_fingerprint(model_path: Any, models_dir: Path | None) -> str:
    """Resolve model paths so reuse fingerprints are stable across relative inputs."""
    raw_path = Path(str(model_path))
    if raw_path.is_absolute() or models_dir is None:
        return str(raw_path.resolve())
    return str((models_dir / raw_path).resolve())


def calculate_image_hash(params: dict[str, Any], payload: dict[str, Any], generation_setup_fp: str = "") -> str:
    """
    Create a deterministic SHA256 hash from generation settings and merge params.
    """
    gen_keys = [
        "prompt",
        "negative_prompt",
        "seed",
        "steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "width",
        "height",
        "workflow_json",
    ]

    stable_params = {k: params[k] for k in sorted(params.keys())}
    stable_payload = {k: payload.get(k) for k in gen_keys if k in payload}
    recipe = {
        "generation_setup_fp": generation_setup_fp,
        "params": stable_params,
        "payload": stable_payload,
    }
    recipe_json = json.dumps(recipe, sort_keys=True, default=str)
    return hashlib.sha256(recipe_json.encode("utf-8")).hexdigest()
