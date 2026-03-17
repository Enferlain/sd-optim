import logging

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, open_dict
from PIL import Image
from sd_optim.scoring.assets import get_models, setup_evaluator_paths
from sd_optim.scoring.loading import load_all_models
from sd_optim.scoring.registry import get_scorer_class
from sd_optim.scoring.runtime import (
    SCORERS_NEEDING_REMBG,
    ensure_rembg_session,
    score_image,
    setup_img_saving,
)

try:
    from rembg import new_session
except ImportError:
    new_session = None

logger = logging.getLogger(__name__)

__all__ = ["Scorer", "get_scorer_class"]

@dataclass
class Scorer:
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
        self.rembg_session_factory = new_session
        self._rembg_required = any(
            str(s).lower() in self._scorers_needing_rembg for s in self.cfg.get("scorer_method", [])
        )

        setup_img_saving(self)

        with open_dict(self.cfg):
            self.cfg.scorer_weight = self.cfg.scorer_weight or {}
            # Ensure scorer_device exists before setup_evaluator_paths uses it
            self.cfg.scorer_device = self.cfg.scorer_device or {}

        setup_evaluator_paths(self)  # Populates self.model_path and sets default devices/weights
        get_models(self)  # Downloads files if needed
        ensure_rembg_session(self, session_factory=self.rembg_session_factory)
        load_all_models(self)

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

    async def score(self, image: Image.Image, prompt: str, name: str | None = None) -> float:
        return await score_image(self, image, prompt, name)
