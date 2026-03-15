"""Factory and model-loading helpers for scorer runtime setup."""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any

from sd_optim.builtin.scorers.registry import MODEL_DATA, get_scorer_class

logger = logging.getLogger(__name__)


def build_scorer_factory(clip_l_path: Path, clip_b_path: Path) -> dict[str, dict[str, Any]]:
    """Return scorer construction metadata keyed by scorer id."""
    return {
        "laion": {
            "class_ref": "laion",
            "files": {"model_path": "file_name"},
            "extra_args": {"clip_model_path": str(clip_l_path)},
        },
        "chad": {
            "class_ref": "chad",
            "files": {"model_path": "file_name"},
            "extra_args": {"clip_model_path": str(clip_l_path)},
        },
        "wdaes": {
            "class_ref": "wdaes",
            "files": {"model_path": "file_name"},
            "extra_args": {"clip_path": str(clip_b_path)},
        },
        "clip": {
            "class_ref": "clip",
            "files": {"model_path": "file_name"},
        },
        "pick": {
            "class_ref": "pick",
            "files": {"model_path": "file_name"},
        },
        "shadowv2": {
            "class_ref": "shadowv2",
            "files": {"model_path": "file_name"},
        },
        "cafe": {
            "class_ref": "cafe",
            "files": {"model_path": "file_name"},
        },
        "noai": {
            "class_ref": "noai",
            "files": {
                "model_path_class": "class",
                "model_path_real": "real",
                "model_path_anime": "anime",
            },
        },
        "cityaes": {
            "class_ref": "cityaes",
            "files": {"pathname": "file_name"},
        },
        "aestheticv25": {
            "class_ref": "aestheticv25",
            "files": {"model_path": "file_name"},
        },
        "luminaflex": {
            "class_ref": "luminaflex",
            "files": {"model_path": "file_name", "config_path": "config_name"},
        },
        "lumidinov3": {
            "class_ref": "lumidinov3",
            "files": {"model_path": "file_name", "config_path": "config_name"},
        },
        "lumidinov2l": {
            "class_ref": "lumidinov2l",
            "files": {"model_path": "file_name", "config_path": "config_name"},
        },
        "lumidinov2g": {
            "class_ref": "lumidinov2g",
            "files": {"model_path": "file_name", "config_path": "config_name"},
        },
        "simplequality": {
            "class_ref": "simplequality",
            "files": {},
            "extra_args": {},
        },
        "hybridnoise": {
            "class_ref": "hybridnoise",
            "files": {},
            "extra_args": {"rembg_session": "self.rembg_session"},
        },
        "hybridnoise_fullimg": {
            "class_ref": "hybridnoise_fullimg",
            "files": {},
            "extra_args": {},
        },
        "backgroundblackness": {
            "class_ref": "backgroundblackness",
            "files": {},
            "extra_args": {"rembg_session": "self.rembg_session"},
        },
        "pcascorer": {
            "class_ref": "pcascorer",
            "files": {},
            "extra_args": {},
        },
        "textureclean": {
            "class_ref": "textureclean",
            "files": {},
            "extra_args": {"rembg_session": "self.rembg_session"},
        },
        "textureclean_fullimg": {
            "class_ref": "textureclean_fullimg",
            "files": {},
            "extra_args": {},
        },
    }


def load_model(scorer: Any, evaluator_lower: str) -> bool:
    """Load a single scorer model instance on demand."""
    if evaluator_lower in scorer.model:
        logger.debug("Model '%s' is already loaded.", evaluator_lower)
        return True

    if evaluator_lower in scorer._scorers_needing_rembg:
        scorer._ensure_rembg_session()

    logger.info("Lazy loading scorer model: '%s'", evaluator_lower)
    scorer_model_dir_path = Path(scorer.cfg.scorer_model_dir)
    clip_l_path = scorer_model_dir_path / "CLIP-ViT-L-14.pt"
    clip_b_path = scorer_model_dir_path / "CLIP-ViT-B-32.safetensors"
    scorer_factory = build_scorer_factory(clip_l_path, clip_b_path)

    if evaluator_lower not in scorer_factory:
        logger.error("Unknown scorer '%s' cannot be lazy-loaded.", evaluator_lower)
        return False

    config = scorer_factory[evaluator_lower]
    scorer_class = get_scorer_class(config.get("class_ref", evaluator_lower))

    if scorer_class is None:
        logger.error("Scorer class for '%s' not available.", evaluator_lower)
        return False

    constructor_args, file_paths_ok = _resolve_constructor_args(
        scorer,
        evaluator_lower,
        scorer_class,
        config,
    )
    if not file_paths_ok:
        return False

    try:
        scorer.model[evaluator_lower] = scorer_class(**constructor_args)
        logger.info(
            "Successfully lazy-loaded instance for scorer: '%s'",
            evaluator_lower,
        )
        return True
    except Exception as init_error:
        logger.error(
            "Failed to initialize lazy-loaded instance for '%s': %s",
            evaluator_lower,
            init_error,
            exc_info=True,
        )
        return False


def load_all_models(scorer: Any) -> None:
    """Load all non-lazy configured scorer model instances."""
    logger.info("Loading scorer model instances...")
    lazy_load_list = [s.lower() for s in scorer.cfg.get("scorer_lazy_load_list", [])]
    scorer_model_dir_path = Path(scorer.cfg.scorer_model_dir)
    clip_l_path = scorer_model_dir_path / "CLIP-ViT-L-14.pt"
    clip_b_path = scorer_model_dir_path / "CLIP-ViT-B-32.safetensors"
    scorer_factory = build_scorer_factory(clip_l_path, clip_b_path)

    for evaluator in scorer.cfg.scorer_method:
        evaluator_lower = evaluator.lower()
        if evaluator_lower in ["manual", "background_blackness"]:
            continue

        if evaluator_lower in lazy_load_list:
            logger.info(
                "Deferring loading of scorer '%s' due to lazy load list.",
                evaluator,
            )
            continue

        logger.info("Loading instance for scorer: '%s'", evaluator)

        if evaluator_lower not in scorer_factory:
            logger.error(
                "Unknown scorer '%s' defined in config but not found in scorer_factory. Skipping.",
                evaluator,
            )
            continue

        config = scorer_factory[evaluator_lower]
        scorer_class = get_scorer_class(config.get("class_ref", evaluator_lower))

        if scorer_class is None:
            logger.error(
                "Scorer class for '%s' not available (possibly failed import). Skipping.",
                evaluator,
            )
            continue

        constructor_args, file_paths_ok = _resolve_constructor_args(
            scorer,
            evaluator_lower,
            scorer_class,
            config,
        )
        if not file_paths_ok:
            continue

        if evaluator_lower in {"hybridnoise", "hybridnoise_fullimg"}:
            constructor_args["kernel_size"] = scorer.cfg.get(
                "hybridnoise_kernel_size",
                3,
            )
            constructor_args["noise_threshold"] = scorer.cfg.get(
                "hybridnoise_noise_threshold",
                20.0,
            )
            if evaluator_lower == "hybridnoise":
                constructor_args["color_tolerance"] = scorer.cfg.get(
                    "hybridnoise_color_tolerance",
                    30,
                )

        try:
            logger.debug(
                "Instantiating %s with args: %s",
                scorer_class.__name__,
                constructor_args,
            )
            scorer.model[evaluator_lower] = scorer_class(**constructor_args)
            logger.info("Successfully loaded instance for scorer: '%s'", evaluator)
        except Exception as init_error:
            logger.error(
                "Failed to initialize instance for '%s': %s",
                evaluator,
                init_error,
                exc_info=True,
            )


def _resolve_constructor_args(
    scorer: Any,
    evaluator_lower: str,
    scorer_class: type[Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    constructor_args: dict[str, Any] = {}
    scorer_model_dir_path = Path(scorer.cfg.scorer_model_dir)

    if "device" in inspect.signature(scorer_class.__init__).parameters:
        try:
            constructor_args["device"] = scorer.cfg.scorer_device.get(
                evaluator_lower,
                scorer.cfg.scorer_default_device,
            )
        except KeyError:
            logger.error("Device config missing for '%s'.", evaluator_lower)
            return constructor_args, False

    if "files" in config:
        for arg_name, model_data_key in config["files"].items():
            try:
                model_data_entry = MODEL_DATA.get(evaluator_lower)
                if not model_data_entry:
                    raise KeyError("MODEL_DATA entry missing")

                filename = model_data_entry.get(model_data_key)
                if not filename:
                    raise KeyError(
                        f"Filename key '{model_data_key}' not found in MODEL_DATA "
                        f"for '{evaluator_lower}'"
                    )

                if (
                    arg_name in ["model_path", "pathname"]
                    and evaluator_lower in scorer.model_path
                ):
                    file_path = scorer.model_path[evaluator_lower]
                    if file_path.name != filename:
                        logger.warning(
                            "Filename mismatch for %s arg %s: Expected %s, "
                            "Path has %s. Using path.",
                            evaluator_lower,
                            arg_name,
                            filename,
                            file_path.name,
                        )
                else:
                    file_path = scorer_model_dir_path / filename

                if not file_path.is_file():
                    logger.error(
                        "Required file for '%s', arg '%s' not found: %s",
                        evaluator_lower,
                        arg_name,
                        file_path,
                    )
                    return constructor_args, False
                constructor_args[arg_name] = str(file_path)
            except KeyError as config_error:
                logger.error(
                    "Config error resolving file for '%s', arg '%s': %s",
                    evaluator_lower,
                    arg_name,
                    config_error,
                )
                return constructor_args, False
            except Exception as path_error:
                logger.error(
                    "Error resolving path for '%s', arg '%s': %s",
                    evaluator_lower,
                    arg_name,
                    path_error,
                )
                return constructor_args, False

    if "extra_args" in config:
        constructor_args.update(_resolve_extra_args(scorer, config["extra_args"]))

    return constructor_args, True


def _resolve_extra_args(
    scorer: Any,
    extra_args: dict[str, Any],
) -> dict[str, Any]:
    resolved_extra_args: dict[str, Any] = {}
    for key, value in extra_args.items():
        if key == "rembg_session" and value == "self.rembg_session":
            if scorer.rembg_session:
                resolved_extra_args[key] = scorer.rembg_session
            else:
                logger.warning(
                    "rembg_session not available, but it was requested."
                )
        else:
            resolved_extra_args[key] = str(value) if isinstance(value, Path) else value
    return resolved_extra_args


__all__ = ["build_scorer_factory", "load_all_models", "load_model"]
