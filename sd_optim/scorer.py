import logging

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, open_dict
from PIL import Image
from sd_optim.scoring.assets import download_file, get_models, setup_evaluator_paths
from sd_optim.scoring.interaction import get_user_score, handle_override_prompt, open_image
from sd_optim.scoring.loading import build_scorer_factory, load_all_models, load_model
from sd_optim.scoring.registry import get_scorer_class
from sd_optim.scoring.runtime import (
    SCORERS_NEEDING_REMBG,
    average_calc,
    build_manual_preview_path,
    ensure_rembg_session,
    sanitize_manual_preview_name,
    score_image,
    setup_img_saving,
    show_image_with_pil,
    show_manual_preview,
)

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
        self._scorers_needing_rembg = set(SCORERS_NEEDING_REMBG)
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
        ensure_rembg_session(self, session_factory=new_session)

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
        setup_img_saving(self)

    def setup_evaluator_paths(self):
        """Sets up model paths (Path objects) and default configs for each configured evaluator."""
        setup_evaluator_paths(self)

    # get_models can be simplified or adjusted based on the factory pattern if needed
    # It mainly needs to ensure *all* required files listed in MODEL_DATA (file_name, config_name, class, real, anime etc.)
    # for the *configured* scorers are downloaded if missing.
    # (Keeping previous refined version for now)
    def get_models(self) -> None:
        """Downloads necessary model files if they do not exist."""
        get_models(self)

    def download_file(self, url: str, path: Path):
        """Downloads a file from a URL to the specified path."""
        download_file(url, path)

    def _build_scorer_factory(self, clip_l_path: Path, clip_b_path: Path):
        return build_scorer_factory(clip_l_path, clip_b_path)

    def _load_model(self, evaluator_lower: str):
        """Loads a single scorer model instance on demand."""
        return load_model(self, evaluator_lower)

    def _load_all_models(self):
        """Loads instances for all configured scorers using a factory pattern."""
        load_all_models(self)

    def _build_manual_preview_path(self, name: str | None = None) -> Path | None:
        return build_manual_preview_path(self, name)

    @staticmethod
    def _sanitize_manual_preview_name(name: str | None) -> str:
        return sanitize_manual_preview_name(name)

    def _show_manual_preview(self, image: Image.Image, name: str | None = None) -> None:
        show_manual_preview(self, image, name)

    @staticmethod
    def _show_image_with_pil(image: Image.Image) -> None:
        show_image_with_pil(image)

    async def score(self, image: Image.Image, prompt: str, name: str | None = None) -> float:
        return await score_image(self, image, prompt, name)

    # --- ADDED: Static method to handle override prompt ---
    @staticmethod
    def handle_override_prompt() -> float:
        """Prompt the user for a fake average score during an override."""
        return handle_override_prompt()

    def average_calc(self, values: list[float], scorer_weights: list[float], average_type: str) -> float:
        return average_calc(values, scorer_weights, average_type)

    def open_image(self, image_path: Path) -> None:
        open_image(image_path, warning_state=self._runtime_warnings)

    @staticmethod
    def get_user_score() -> float:
        return get_user_score()
