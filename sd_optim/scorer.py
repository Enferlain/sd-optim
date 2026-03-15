import asyncio
import inspect
import threading

import logging

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict
from PIL import Image
from sd_optim.builtin.scorers import assets as scorer_assets
from sd_optim.builtin.scorers import interaction as scorer_interaction
from sd_optim.builtin.scorers import loading as scorer_loading
from sd_optim.builtin.scorers.registry import get_scorer_class

try:
    from rembg import new_session
except ImportError:
    new_session = None

logger = logging.getLogger(__name__)


def _get_scorer_class(scorer_name: str):
    return get_scorer_class(scorer_name)


@dataclass
class AestheticScorer:
    cfg: DictConfig

    # --- REMOVED scorer_model_name, model_path, model from dataclass fields ---
    # These will be instance attributes initialized later

    def __post_init__(self):
        # Initialize instance attributes
        self.model: dict[str, Any] = {}  # Dictionary to hold loaded scorer instances
        self.model_path: dict[str, Path] = {}  # Dictionary to hold Path objects for models
        # Stores individual scorer results from the last score() call for metadata
        self.last_scorer_results: dict[str, float] = {}
        self.rembg_session: Any | None = None
        self._manual_preview_index = 0
        self._runtime_warnings: set[str] = set()
        self._scorers_needing_rembg = {
            "hybridnoise",
            "backgroundblackness",
            "textureclean",
        }
        self._rembg_required = any(
            str(s).lower() in self._scorers_needing_rembg for s in self.cfg.get("scorer_method", [])
        )

        self.setup_img_saving()

        with open_dict(self.cfg):
            self.cfg.scorer_weight = self.cfg.scorer_weight or {}
            # Ensure scorer_device exists before setup_evaluator_paths uses it
            self.cfg.scorer_device = self.cfg.scorer_device or {}

        self.setup_evaluator_paths()  # Populates self.model_path and sets default devices/weights
        self.get_models()  # Downloads files if needed
        self._ensure_rembg_session()
        self._load_all_models()

    def _ensure_rembg_session(self) -> None:
        if self.rembg_session is not None or not self._rembg_required:
            return

        logger.info("A configured scorer requires background removal. Initializing rembg session...")
        if new_session is None:
            raise ImportError(
                "A configured scorer requires 'rembg', but it is not installed. "
                "Install the corresponding scorer extra (for example "
                "'scorer-textureclean', 'scorer-hybridnoise', or "
                "'scorer-backgroundblackness')."
            )
        try:
            self.rembg_session = new_session(providers=["CPUExecutionProvider"])
        except ImportError as exc:
            raise ImportError(
                "A configured scorer requires 'rembg', but it is not installed. "
                "Install the corresponding scorer extra (for example "
                "'scorer-textureclean', 'scorer-hybridnoise', or "
                "'scorer-backgroundblackness')."
            ) from exc

    def unload_lazy_models(self):
        """Unloads models that were loaded on-demand."""
        lazy_load_list = [s.lower() for s in self.cfg.get("scorer_lazy_load_list", [])]
        models_to_unload = [model_name for model_name in self.model if model_name in lazy_load_list]

        for model_name in models_to_unload:
            logger.info(f"Unloading lazy-loaded scorer model: '{model_name}'")
            del self.model[model_name]

        # Optional: Add garbage collection for immediate memory release
        if models_to_unload:
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def setup_img_saving(self):
        """Sets up the directory for saving images if enabled."""
        # Use .get() for safer access, default to False
        save_enabled = self.cfg.get("save_imgs", False)
        # Also enable saving if manual scoring is used
        if "manual" in self.cfg.get("scorer_method", []):
            save_enabled = True

        if save_enabled:
            try:
                # Try getting path from Hydra context
                self.imgs_dir = Path(HydraConfig.get().runtime.output_dir, "imgs")
            except ValueError:
                # Fallback if Hydra context not available
                logger.warning("Hydra context not available, saving images to ./imgs_fallback")
                self.imgs_dir = Path("./imgs_fallback").resolve()

            self.imgs_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Image saving enabled. Saving to: {self.imgs_dir}")
            # Update config back if manual mode forced it (optional)
            with open_dict(self.cfg):
                self.cfg.save_imgs = True
        else:
            self.imgs_dir = None  # Explicitly set to None if disabled
            logger.info("Image saving disabled.")

    def setup_evaluator_paths(self):
        """Sets up model paths (Path objects) and default configs for each configured evaluator."""
        scorer_assets.setup_evaluator_paths(self)

    # get_models can be simplified or adjusted based on the factory pattern if needed
    # It mainly needs to ensure *all* required files listed in MODEL_DATA (file_name, config_name, class, real, anime etc.)
    # for the *configured* scorers are downloaded if missing.
    # (Keeping previous refined version for now)
    def get_models(self) -> None:
        """Downloads necessary model files if they do not exist."""
        scorer_assets.get_models(self)

    def download_file(self, url: str, path: Path):
        """Downloads a file from a URL to the specified path."""
        scorer_assets.download_file(url, path)

    def _build_scorer_factory(self, clip_l_path: Path, clip_b_path: Path):
        return scorer_loading.build_scorer_factory(clip_l_path, clip_b_path)

    def _load_model(self, evaluator_lower: str):
        """Loads a single scorer model instance on demand."""
        return scorer_loading.load_model(self, evaluator_lower)

    def _load_all_models(self):
        """Loads instances for all configured scorers using a factory pattern."""
        scorer_loading.load_all_models(self)

    def _build_manual_preview_path(self, name: str | None = None) -> Path | None:
        if self.imgs_dir is None:
            return None

        safe_name = self._sanitize_manual_preview_name(name)
        preview_path = self.imgs_dir / f"manual-{self._manual_preview_index:04}-{safe_name}.png"
        self._manual_preview_index += 1
        return preview_path

    @staticmethod
    def _sanitize_manual_preview_name(name: str | None) -> str:
        raw_name = (name or "preview").strip()
        safe_name = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in raw_name)
        return safe_name or "preview"

    def _show_manual_preview(self, image: Image.Image, name: str | None = None) -> None:
        preview_path = self._build_manual_preview_path(name)
        if preview_path is None:
            self._show_image_with_pil(image)
            return

        try:
            image.save(preview_path)
        except OSError as error:
            logger.error("Error saving manual preview image to %s: %s", preview_path, error)
            self._show_image_with_pil(image)
            return

        logger.debug("Saved manual scoring preview to %s", preview_path)
        threading.Thread(target=self.open_image, args=(preview_path,), daemon=True).start()

    @staticmethod
    def _show_image_with_pil(image: Image.Image) -> None:
        try:
            image.show()
        except Exception as error:
            logger.error("Error displaying image with PIL: %s", error)

    async def score(self, image: Image.Image, prompt: str, name: str | None = None) -> float:
        values: list[float] = []
        scorer_weights: list[float] = []
        self.last_scorer_results = {}  # Reset for this image
        logger.info("Entering score method.")

        # --- Scoring Loop ---
        for evaluator in self.cfg.scorer_method:
            # --- Manual Scoring Path ---
            if evaluator == "manual":
                self._show_manual_preview(image, name)
                individual_eval_score = await asyncio.to_thread(self.get_user_score)
                if individual_eval_score == -1.0:
                    return -1.0

                weight = self.cfg.scorer_weight.get(evaluator, 1.0)
                values.append(individual_eval_score)
                scorer_weights.append(weight)
                self.last_scorer_results[evaluator] = individual_eval_score

                # Print its own score
                if self.cfg.scorer_print_individual:
                    logger.info("%s:%.4f", evaluator, individual_eval_score)

            # --- General Automatic Scorer Path ---
            else:
                evaluator_lower = evaluator.lower()

                # --- Payload Filtering Logic (Exclude Only) ---
                run_scorer = True
                scorer_filters = self.cfg.get("scorer_filters", {})
                if name and scorer_filters and evaluator_lower in scorer_filters:
                    filter_config = scorer_filters[evaluator_lower]
                    exclude_list = filter_config.get("exclude")
                    if exclude_list and name in exclude_list:
                        run_scorer = False

                if not run_scorer:
                    logger.debug(f"Skipping scorer '{evaluator}' for payload '{name}' due to exclude filter.")
                    continue
                # --- End Filtering Logic ---

                individual_eval_score = 0.0  # Default score

                try:
                    lazy_load_list = [s.lower() for s in self.cfg.get("scorer_lazy_load_list", [])]
                    scorer_instance = self.model.get(evaluator_lower)

                    if scorer_instance is None and evaluator_lower in lazy_load_list:
                        logger.info(f"'{evaluator}' is in lazy load list and not loaded. Attempting to load now.")
                        if self._load_model(evaluator_lower):
                            scorer_instance = self.model.get(evaluator_lower)
                        else:
                            logger.error(f"Failed to lazy-load model for '{evaluator}'. Skipping scoring.")
                            continue

                    if scorer_instance is None:
                        logger.error(f"Scorer instance for '{evaluator}' not found and not lazy-loadable. Skipping.")
                        continue

                    elif evaluator_lower == "pcascorer":
                        # This scorer has extra parameters for analysis
                        score_args = {"image": image}
                        # Read PCA settings from config, with defaults
                        score_args["component"] = self.cfg.get("pcascorer_component", 1)
                        score_args["mode"] = self.cfg.get("pcascorer_mode", "projection")
                        score_args["input_type"] = self.cfg.get("pcascorer_input_type", "color")
                        score_args["linearize"] = self.cfg.get("pcascorer_linearize", False)
                        score_args["invert"] = self.cfg.get("pcascorer_invert", False)
                        score_args["enhancement"] = self.cfg.get("pcascorer_enhancement", "equalize")
                        score_args["gamma"] = self.cfg.get("pcascorer_gamma", 1.0)

                        individual_eval_score = scorer_instance.score(**score_args)

                        if self.cfg.scorer_print_individual:
                            logger.info("%s:%.4f", evaluator, individual_eval_score)

                    elif evaluator_lower == "hpsv3":
                        # HPSv3 returns a tuple (mu, sigma)
                        mu_score, sigma_score = scorer_instance.score(image=image, prompt=prompt)

                        # Print individual mu and sigma
                        if self.cfg.scorer_print_individual:
                            logger.info("%s (score): %.4f", evaluator, mu_score)
                            logger.info("%s (uncertainty): %.4f", evaluator, sigma_score)

                        # Process a final score combining mu and sigma.
                        # Using Lower Confidence Bound: score = mu - k * sigma
                        # This penalizes scores with high uncertainty.
                        # A higher 'k' means more penalty for uncertainty.
                        k = self.cfg.get("hpsv3_uncertainty_penalty", 0.5)
                        individual_eval_score = mu_score - (k * sigma_score)

                        if self.cfg.scorer_print_individual:
                            logger.info("%s (processed final): %.4f", evaluator, individual_eval_score)

                    else:
                        # Standard scoring for other models
                        score_args = {"image": image}
                        score_params = inspect.signature(scorer_instance.score).parameters
                        if "prompt" in score_params:
                            score_args["prompt"] = prompt
                        individual_eval_score = scorer_instance.score(**score_args)

                        # Print its own score for other models
                        if self.cfg.scorer_print_individual:
                            logger.info("%s:%.4f", evaluator, individual_eval_score)

                except Exception as e:
                    logger.error(f"Error scoring with {evaluator}: {e}", exc_info=True)
                    individual_eval_score = 0.0  # Ensure it's a float on error

                weight = self.cfg.scorer_weight.get(evaluator_lower, 1.0)
                values.append(individual_eval_score)
                scorer_weights.append(weight)
                self.last_scorer_results[evaluator_lower] = individual_eval_score
        # --- End Scoring Loop ---

        score = self.average_calc(values, scorer_weights, self.cfg.scorer_average_type)
        return score

    # --- ADDED: Static method to handle override prompt ---
    @staticmethod
    def handle_override_prompt() -> float:
        """Prompt the user for a fake average score during an override."""
        return scorer_interaction.handle_override_prompt()

    def average_calc(self, values: list[float], scorer_weights: list[float], average_type: str) -> float:
        # Ensure weights and values match length
        if len(values) != len(scorer_weights):
            logger.error(
                f"Score calculation error: Mismatched values ({len(values)}) and weights ({len(scorer_weights)}). Using default weights."
            )
            # Fallback to equal weights if lengths mismatch
            scorer_weights = [1.0] * len(values)

        # Filter out potential None values if errors occurred and weren't handled before
        valid_data = [(v, w) for v, w in zip(values, scorer_weights) if v is not None]
        if not valid_data:
            return 0.0  # Return 0 if no valid scores

        values, scorer_weights = zip(*valid_data)
        norm = sum(scorer_weights)
        if norm == 0:
            return 0.0  # Avoid division by zero

        if average_type == "geometric":
            # Avoid log(0) or negative numbers for geometric mean
            product = 1.0
            total_weight = 0.0
            for value, weight in zip(values, scorer_weights):
                if value > 0:
                    product *= value**weight
                    total_weight += weight
                else:  # Handle non-positive scores - maybe skip or use a floor? Skipping is safer.
                    logger.warning(f"Skipping non-positive score {value} in geometric mean calculation.")
            return product ** (1 / total_weight) if total_weight > 0 else 0.0
        elif average_type == "arithmetic":
            return sum(value * weight for value, weight in zip(values, scorer_weights)) / norm
        elif average_type == "quadratic":
            # Ensure values are non-negative for quadratic mean if that's intended
            avg_sq = sum((value**2) * weight for value, weight in zip(values, scorer_weights))
            return (avg_sq / norm) ** 0.5  # Use 0.5 for square root
        else:
            raise ValueError(f"Invalid average type: {average_type}")

    def open_image(self, image_path: Path) -> None:
        scorer_interaction.open_image(image_path, warning_state=self._runtime_warnings)

    @staticmethod
    def get_user_score() -> float:
        return scorer_interaction.get_user_score()
