from __future__ import annotations

import asyncio
import inspect
import logging
import threading

from functools import partial
from pathlib import Path
from typing import Any

from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict
from PIL import Image
from sd_optim.scoring.interaction import get_user_score, open_image

logger = logging.getLogger(__name__)

SCORERS_NEEDING_REMBG = frozenset(
    {
        "hybridnoise",
        "backgroundblackness",
        "textureclean",
    }
)


def ensure_rembg_session(scorer: Any, *, session_factory: Any) -> None:
    if scorer.rembg_session is not None or not scorer._rembg_required:
        return

    logger.info("A configured scorer requires background removal. Initializing rembg session...")
    if session_factory is None:
        raise ImportError(
            "A configured scorer requires 'rembg', but it is not installed. "
            "Install the corresponding scorer extra (for example "
            "'scorer-textureclean', 'scorer-hybridnoise', or "
            "'scorer-backgroundblackness')."
        )
    try:
        scorer.rembg_session = session_factory(providers=["CPUExecutionProvider"])
    except ImportError as exc:
        raise ImportError(
            "A configured scorer requires 'rembg', but it is not installed. "
            "Install the corresponding scorer extra (for example "
            "'scorer-textureclean', 'scorer-hybridnoise', or "
            "'scorer-backgroundblackness')."
        ) from exc


def setup_img_saving(scorer: Any) -> None:
    """Set up the directory for saving images if enabled."""
    save_enabled = scorer.cfg.get("save_imgs", False)
    if "manual" in scorer.cfg.get("scorer_method", []):
        save_enabled = True

    if save_enabled:
        try:
            scorer.imgs_dir = Path(HydraConfig.get().runtime.output_dir, "imgs")
        except ValueError:
            logger.warning("Hydra context not available, saving images to ./imgs_fallback")
            scorer.imgs_dir = Path("./imgs_fallback").resolve()

        scorer.imgs_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Image saving enabled. Saving to: %s", scorer.imgs_dir)
        with open_dict(scorer.cfg):
            scorer.cfg.save_imgs = True
        return

    scorer.imgs_dir = None
    logger.info("Image saving disabled.")


def build_manual_preview_path(scorer: Any, name: str | None = None) -> Path | None:
    if scorer.imgs_dir is None:
        return None

    safe_name = sanitize_manual_preview_name(name)
    preview_path = scorer.imgs_dir / f"manual-{scorer._manual_preview_index:04}-{safe_name}.png"
    scorer._manual_preview_index += 1
    return preview_path


def sanitize_manual_preview_name(name: str | None) -> str:
    raw_name = (name or "preview").strip()
    safe_name = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in raw_name)
    return safe_name or "preview"


def show_image_with_pil(image: Image.Image) -> None:
    try:
        image.show()
    except Exception as error:  # noqa: BLE001 - preserve scorer fallback behavior.
        logger.error("Error displaying image with PIL: %s", error)


def show_manual_preview(scorer: Any, image: Image.Image, name: str | None = None) -> None:
    preview_path = build_manual_preview_path(scorer, name)
    if preview_path is None:
        show_image_with_pil(image)
        return

    try:
        image.save(preview_path)
    except OSError as error:
        logger.error("Error saving manual preview image to %s: %s", preview_path, error)
        show_image_with_pil(image)
        return

    logger.debug("Saved manual scoring preview to %s", preview_path)
    threading.Thread(
        target=partial(open_image, warning_state=scorer._runtime_warnings),
        args=(preview_path,),
        daemon=True,
    ).start()


async def score_image(scorer: Any, image: Image.Image, prompt: str, name: str | None = None) -> float:
    values: list[float] = []
    scorer_weights: list[float] = []
    scorer.last_scorer_results = {}
    logger.info("Entering score method.")

    for evaluator in scorer.cfg.scorer_method:
        if evaluator == "manual":
            show_manual_preview(scorer, image, name)
            individual_eval_score = await asyncio.to_thread(get_user_score)
            if individual_eval_score == -1.0:
                return -1.0

            weight = scorer.cfg.scorer_weight.get(evaluator, 1.0)
            values.append(individual_eval_score)
            scorer_weights.append(weight)
            scorer.last_scorer_results[evaluator] = individual_eval_score
            _log_individual_score(scorer, evaluator, individual_eval_score)
            continue

        evaluator_lower = evaluator.lower()
        if not _should_run_scorer(scorer, evaluator_lower, name):
            logger.debug("Skipping scorer '%s' for payload '%s' due to exclude filter.", evaluator, name)
            continue

        try:
            scorer_instance = _get_scorer_instance(scorer, evaluator, evaluator_lower)
            if scorer_instance is None:
                continue
            individual_eval_score = _score_with_instance(scorer, evaluator, evaluator_lower, scorer_instance, image, prompt)
        except Exception as error:  # noqa: BLE001 - preserve broad scorer failure handling.
            logger.error("Error scoring with %s: %s", evaluator, error, exc_info=True)
            individual_eval_score = 0.0

        weight = scorer.cfg.scorer_weight.get(evaluator_lower, 1.0)
        values.append(individual_eval_score)
        scorer_weights.append(weight)
        scorer.last_scorer_results[evaluator_lower] = individual_eval_score

    return average_calc(values, scorer_weights, scorer.cfg.scorer_average_type)


def average_calc(values: list[float], scorer_weights: list[float], average_type: str) -> float:
    if len(values) != len(scorer_weights):
        logger.error(
            "Score calculation error: Mismatched values (%s) and weights (%s). Using default weights.",
            len(values),
            len(scorer_weights),
        )
        scorer_weights = [1.0] * len(values)

    valid_data = [(value, weight) for value, weight in zip(values, scorer_weights) if value is not None]
    if not valid_data:
        return 0.0

    values_tuple, weights_tuple = zip(*valid_data)
    norm = sum(weights_tuple)
    if norm == 0:
        return 0.0

    if average_type == "geometric":
        product = 1.0
        total_weight = 0.0
        for value, weight in zip(values_tuple, weights_tuple):
            if value > 0:
                product *= value**weight
                total_weight += weight
            else:
                logger.warning("Skipping non-positive score %s in geometric mean calculation.", value)
        return product ** (1 / total_weight) if total_weight > 0 else 0.0
    if average_type == "arithmetic":
        return sum(value * weight for value, weight in zip(values_tuple, weights_tuple)) / norm
    if average_type == "quadratic":
        avg_sq = sum((value**2) * weight for value, weight in zip(values_tuple, weights_tuple))
        return (avg_sq / norm) ** 0.5
    raise ValueError(f"Invalid average type: {average_type}")


def _log_individual_score(scorer: Any, evaluator: str, score: float) -> None:
    if scorer.cfg.scorer_print_individual:
        logger.info("%s:%.4f", evaluator, score)


def _should_run_scorer(scorer: Any, evaluator_lower: str, name: str | None) -> bool:
    scorer_filters = scorer.cfg.get("scorer_filters", {})
    if not (name and scorer_filters and evaluator_lower in scorer_filters):
        return True

    filter_config = scorer_filters[evaluator_lower]
    exclude_list = filter_config.get("exclude")
    return not (exclude_list and name in exclude_list)


def _get_scorer_instance(scorer: Any, evaluator: str, evaluator_lower: str) -> Any | None:
    from .loading import load_model

    lazy_load_list = [str(name).lower() for name in scorer.cfg.get("scorer_lazy_load_list", [])]
    scorer_instance = scorer.model.get(evaluator_lower)

    if scorer_instance is None and evaluator_lower in lazy_load_list:
        logger.info("'%s' is in lazy load list and not loaded. Attempting to load now.", evaluator)
        if load_model(scorer, evaluator_lower):
            scorer_instance = scorer.model.get(evaluator_lower)
        else:
            logger.error("Failed to lazy-load model for '%s'. Skipping scoring.", evaluator)
            return None

    if scorer_instance is None:
        logger.error("Scorer instance for '%s' not found and not lazy-loadable. Skipping.", evaluator)
        return None

    return scorer_instance


def _score_with_instance(
    scorer: Any,
    evaluator: str,
    evaluator_lower: str,
    scorer_instance: Any,
    image: Image.Image,
    prompt: str,
) -> float:
    if evaluator_lower == "pcascorer":
        individual_eval_score = _score_pca_scorer(scorer, scorer_instance, image)
        _log_individual_score(scorer, evaluator, individual_eval_score)
        return individual_eval_score

    if evaluator_lower == "hpsv3":
        return _score_hpsv3_scorer(scorer, scorer_instance, evaluator, image, prompt)

    score_args = {"image": image}
    score_params = inspect.signature(scorer_instance.score).parameters
    if "prompt" in score_params:
        score_args["prompt"] = prompt

    individual_eval_score = scorer_instance.score(**score_args)
    _log_individual_score(scorer, evaluator, individual_eval_score)
    return individual_eval_score


def _score_pca_scorer(scorer: Any, scorer_instance: Any, image: Image.Image) -> float:
    score_args = {"image": image}
    score_args["component"] = scorer.cfg.get("pcascorer_component", 1)
    score_args["mode"] = scorer.cfg.get("pcascorer_mode", "projection")
    score_args["input_type"] = scorer.cfg.get("pcascorer_input_type", "color")
    score_args["linearize"] = scorer.cfg.get("pcascorer_linearize", False)
    score_args["invert"] = scorer.cfg.get("pcascorer_invert", False)
    score_args["enhancement"] = scorer.cfg.get("pcascorer_enhancement", "equalize")
    score_args["gamma"] = scorer.cfg.get("pcascorer_gamma", 1.0)
    return scorer_instance.score(**score_args)


def _score_hpsv3_scorer(
    scorer: Any,
    scorer_instance: Any,
    evaluator: str,
    image: Image.Image,
    prompt: str,
) -> float:
    mu_score, sigma_score = scorer_instance.score(image=image, prompt=prompt)

    if scorer.cfg.scorer_print_individual:
        logger.info("%s (score): %.4f", evaluator, mu_score)
        logger.info("%s (uncertainty): %.4f", evaluator, sigma_score)

    k = scorer.cfg.get("hpsv3_uncertainty_penalty", 0.5)
    individual_eval_score = mu_score - (k * sigma_score)

    if scorer.cfg.scorer_print_individual:
        logger.info("%s (processed final): %.4f", evaluator, individual_eval_score)

    return individual_eval_score
