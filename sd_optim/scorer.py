import asyncio
import importlib
import inspect
import platform
import subprocess
import threading

import requests
import logging

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict, ListConfig
from PIL import Image

try:
    from rembg import new_session
except ImportError:
    new_session = None

logger = logging.getLogger(__name__)

SCORER_CLASS_PATHS = {
    "laion": ("sd_optim.models.Laion", "Laion"),
    "chad": ("sd_optim.models.Laion", "Laion"),
    "clip": ("sd_optim.models.CLIPScore", "CLIPScore"),
    "pick": ("sd_optim.models.PickScore", "PickScore"),
    "wdaes": ("sd_optim.models.WDAes", "WDAes"),
    "shadowv2": ("sd_optim.models.ShadowScore", "ShadowScore"),
    "cafe": ("sd_optim.models.CafeScore", "CafeScore"),
    "noai": ("sd_optim.models.NoAIScore", "NoAIScore"),
    "cityaes": ("sd_optim.models.CityAesthetics", "CityAestheticsScorer"),
    "aestheticv25": ("sd_optim.models.AestheticV25", "AestheticV25"),
    "luminaflex": ("sd_optim.models.LumiAnatomyv2", "Dinov3AnatomyScorer"),
    "lumidinov3": ("sd_optim.models.LumiAnatomyv2", "Dinov3AnatomyScorer"),
    "lumidinov2l": ("sd_optim.models.LumiAnatomyv2", "Dinov3AnatomyScorer"),
    "lumidinov2g": ("sd_optim.models.LumiAnatomyv2", "Dinov3AnatomyScorer"),
    "simplequality": ("sd_optim.models.SimpleQuality", "SimpleQualityScorer"),
    "hybridnoise": ("sd_optim.models.HybridNoiseScorer", "HybridNoiseScorer"),
    "hybridnoise_fullimg": (
        "sd_optim.models.HybridNoiseScorer",
        "HybridNoiseFullImageScorer",
    ),
    "backgroundblackness": (
        "sd_optim.models.BackgroundBlacknessScorer",
        "BackgroundBlacknessScorer",
    ),
    "pcascorer": ("sd_optim.models.PCAScorer", "PCAScorer"),
    "textureclean": ("sd_optim.models.TextureScorer", "TextureScorer"),
    "textureclean_fullimg": (
        "sd_optim.models.TextureScorer",
        "TextureScorerFullImage",
    ),
}


def _import_attr(module_path: str, attr_name: str):
    try:
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    except ImportError as exc:
        logger.warning(
            "Optional scorer dependency missing while importing %s.%s: %s",
            module_path,
            attr_name,
            exc,
        )
    except AttributeError:
        logger.error("Scorer class '%s' not found in module '%s'.", attr_name, module_path)
    except Exception as exc:
        logger.error(
            "Unexpected error while importing %s.%s: %s",
            module_path,
            attr_name,
            exc,
        )
    return None


SCORER_CLASSES = {
    key: _import_attr(module_path, attr_name)
    for key, (module_path, attr_name) in SCORER_CLASS_PATHS.items()
}


def _get_scorer_class(scorer_name: str):
    return SCORER_CLASSES.get(scorer_name.lower())


MODEL_DATA = {
    "laion": {
        "url": "https://github.com/grexzen/SD-Chad/blob/main/sac+logos+ava1-l14-linearMSE.pth?raw=true",
        "file_name": "Laion.pth",
    },
    "chad": {
        "url": "https://github.com/grexzen/SD-Chad/blob/main/chadscorer.pth?raw=true",
        "file_name": "Chad.pth",
    },
    "wdaes": {
        "url": "https://huggingface.co/hakurei/waifu-diffusion-v1-4/resolve/main/models/aes-B32-v0.pth?download=true",
        "file_name": "WD_Aes.pth",
    },
    "imagereward": {
        "url": "https://huggingface.co/THUDM/ImageReward/resolve/main/ImageReward.pt?download=true",
        "file_name": "ImageReward.pt",
    },
    "clip": {
        "url": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt?raw=true",
        "file_name": "CLIP-ViT-L-14.pt",
    },
    "blip": {
        "url": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large.pth?raw=true",
        "file_name": "BLIP_Large.safetensors",
    },
    "hpsv21": {
        "url": "https://huggingface.co/xswu/HPSv2/resolve/main/HPS_v2.1_compressed.pt?download=true",
        "file_name": "HPS_v2.1.pt",
    },
    "hpsv3": {
        "url": "https://huggingface.co/MizzenAI/HPSv3/resolve/main/HPSv3.safetensors?download=true",
        "file_name": "HPSv3.safetensors",
    },
    "pick": {
        "url": "https://huggingface.co/yuvalkirstain/PickScore_v1/resolve/main/model.safetensors?download=true",
        "file_name": "Pick-A-Pic.safetensors",
    },
    "shadowv2": {
        "url": "https://huggingface.co/shadowlilac/aesthetic-shadow-v2/resolve/main/model.safetensors?download=true",
        "file_name": "ShadowV2.safetensors",
    },
    "cafe": {
        "url": "https://huggingface.co/cafeai/cafe_aesthetic/resolve/3bca27c5c0b6021056b1e84e5a18cf1db9fe5d4c/model.safetensors?download=true",
        "file_name": "Cafe.safetensors",
    },
    "class": {
        "url": "https://huggingface.co/cafeai/cafe_style/resolve/d5ae1a7ac05a12ab84732c25f2ea7225d35ac81b/model.safetensors?download=true",
        "file_name": "CLASS.safetensors",
    },
    "real": {
        "url": "https://huggingface.co/Sumsub/Sumsub-ffs-synthetic-2.0/resolve/main/synthetic.pt?download=true",
        "file_name": "REAL.pt",
    },
    "anime": {
        "url": "https://huggingface.co/saltacc/anime-ai-detect/resolve/e175bb6b5e19cda40bc6c9ad85b138ee7c7ce23a/model.safetensors?download=true",
        "file_name": "ANIME.safetensors",
    },
    "cityaes": {
        "url": "https://huggingface.co/city96/CityAesthetics/resolve/main/CityAesthetics-Anime-v1.8.safetensors?download=true",
        "file_name": "CityAesthetics-Anime-v1.8.safetensors",
    },
    "aestheticv25": {
        "url": None,  # No direct download needed
        "file_name": "aesthetic-predictor-v2-5",
    },
    "luminaflex": {  # <<< Our identifier
        "url": None,
        "file_name": "AnatomyFlaws-v14.7_adabelief_fl_naflex_4670_s1K.safetensors",
        # "file_name": "AnatomyFlaws-v11.3_adabelief_fl_naflex_3000_s9K.safetensors",
        # "file_name": "AnatomyFlaws-v6.6_adabeleif_fl_sigmoid_so400m_naflex_efinal_s10K_final.safetensors", # Head filename
        "config_name": "AnatomyFlaws-v14.7_adabelief_fl_naflex_4670.config.json",
        # "config_name": "AnatomyFlaws-v11.3_adabelief_fl_naflex_3000.config.json",
        # "config_name": "AnatomyFlaws-v6.6_adabeleif_fl_sigmoid_so400m_naflex.config.json" # Config filename
    },
    "lumidinov3": {  # <<< Our identifier
        "url": None,
        "file_name": "AnatomyFlaws-v15.5_dinov3_7b_bnb_fl_s3K_best_val.safetensors",
        "config_name": "AnatomyFlaws-v15.5_dinov3_7b_bnb_fl.config.json",
    },
    "lumidinov2l": {
        "url": None,
        "file_name": "AnatomyFlaws-v6.3_adabeleif_fl_sigmoid_dinov2_large_efinal_s10K_final.safetensors",
        "config_name": "AnatomyFlaws-v6.3_adabeleif_fl_sigmoid_dinov2_large.config.json",
    },
    "lumidinov2g": {
        "url": None,
        "file_name": "AnatomyFlaws-v6.4_adabeleif_fl_sigmoid_dinov2_giant_efinal_s10K_final.safetensors",
        "config_name": "AnatomyFlaws-v6.4_adabeleif_fl_sigmoid_dinov2_giant.config.json",
    },
    "simplequality": {
        "url": None,  # No download needed - uses OpenCV/numpy
        "file_name": None,
    },
    "hybridnoise": {
        "url": None,
        "file_name": None,
    },
    "hybridnoise_fullimg": {
        "url": None,
        "file_name": None,
    },
    "backgroundblackness": {
        "url": None,
        "file_name": None,
    },
    "pcascorer": {
        "url": None,
        "file_name": None,
    },
    "textureclean": {
        "url": None,
        "file_name": None,
    },
    "textureclean_fullimg": {
        "url": None,
        "file_name": None,
    },
}

printWSLFlag = 0


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
        logger.debug("Setting up evaluator paths...")
        scorer_model_dir_path = Path(self.cfg.scorer_model_dir)

        # --- Use open_dict context manager ---
        with open_dict(self.cfg):
            # Ensure scorer_device and scorer_weight exist as dicts, create if missing/null
            if not isinstance(self.cfg.get("scorer_device"), DictConfig):
                self.cfg.scorer_device = {}
            if not isinstance(self.cfg.get("scorer_weight"), DictConfig):
                self.cfg.scorer_weight = {}
            # Now self.cfg.scorer_device and self.cfg.scorer_weight are modifiable

            # Make sure self.cfg.scorer_method is iterable (list or ListConfig)
            configured_scorers = self.cfg.get("scorer_method", [])
            if not isinstance(configured_scorers, (list, ListConfig)):
                logger.warning("scorer_method is not a list, cannot process evaluators.")
                configured_scorers = []

            for evaluator in configured_scorers:
                # Use .lower() for case-insensitive matching
                evaluator_lower = str(evaluator).lower()  # Ensure it's a string first
                if evaluator_lower == "manual":
                    continue

                model_data_entry = MODEL_DATA.get(evaluator_lower)
                if not model_data_entry:
                    logger.warning(f"No MODEL_DATA entry for '{evaluator}'. Cannot set path or defaults.")
                    continue

                # --- Handle Alternative Locations ---
                alt_location = self.cfg.get("scorer_alt_location", {}) or {}  # Default to empty dict
                evaluator_alt_config = alt_location.get(evaluator_lower)
                current_model_dir = scorer_model_dir_path
                primary_filename = model_data_entry.get("file_name")

                if isinstance(evaluator_alt_config, (dict, DictConfig)):  # Check if it's dict-like
                    alt_name = evaluator_alt_config.get("model_name")
                    alt_dir_str = evaluator_alt_config.get("model_dir")
                    if alt_name and alt_dir_str:
                        logger.info(f"Using alternative location for '{evaluator}': Dir='{alt_dir_str}', File='{alt_name}'")
                        try:
                            current_model_dir = Path(alt_dir_str)
                            primary_filename = alt_name
                        except Exception as e_path:
                            logger.warning(f"Invalid alternative path for '{evaluator}': {e_path}. Using default path.")
                    else:
                        logger.warning(f"Alternative location config for '{evaluator}' incomplete. Using default path.")
                # --- End Alt Location Handling ---

                # Set path in self.model_path (only if a filename exists)
                if primary_filename and evaluator_lower != "aestheticv25":
                    try:
                        self.model_path[evaluator_lower] = current_model_dir / primary_filename
                    except TypeError as e_path_join:
                        logger.error(f"Error creating path for '{evaluator}': {e_path_join}. Ensure directory and filename are valid.")
                        continue  # Skip defaults if path fails
                elif evaluator_lower == "aestheticv25":
                    logger.debug(f"No file path needed for '{evaluator}'.")
                else:
                    logger.warning(f"MODEL_DATA for '{evaluator}' missing primary 'file_name'. Cannot set base path.")
                    # Continue to set defaults even if path missing? Or skip? Let's continue for now.

                # --- Set defaults (safe now due to open_dict) ---
                try:
                    # Use .get() on the main cfg object for the default device
                    default_device = self.cfg.get("scorer_default_device", "cpu")
                    # Set default device for this evaluator
                    self.cfg.scorer_device.setdefault(evaluator_lower, default_device)
                    # Set default weight for this evaluator
                    self.cfg.scorer_weight.setdefault(evaluator_lower, 1.0)
                except Exception as e_setdefault:
                    # Catch potential errors during setdefault if keys are weird
                    logger.error(f"Error setting default config for '{evaluator}': {e_setdefault}")
            # --- End Loop ---
        # --- End open_dict context ---

        logger.debug(f"Populated model paths: {self.model_path}")
        # Log final config state after defaults are set
        logger.debug(f"Final scorer devices: {self.cfg.get('scorer_device', {})}")
        logger.debug(f"Final scorer weights: {self.cfg.get('scorer_weight', {})}")

    # get_models can be simplified or adjusted based on the factory pattern if needed
    # It mainly needs to ensure *all* required files listed in MODEL_DATA (file_name, config_name, class, real, anime etc.)
    # for the *configured* scorers are downloaded if missing.
    # (Keeping previous refined version for now)
    def get_models(self) -> None:
        """Downloads necessary model files if they do not exist."""
        logger.debug("Checking for necessary scorer model files...")
        scorer_model_dir_path = Path(self.cfg.scorer_model_dir)

        # --- Files needed by specific scorers explicitly ---
        med_config_path = scorer_model_dir_path / "med_config.json"
        if any(x.lower() in ["blip", "imagereward"] for x in self.cfg.scorer_method):
            if not med_config_path.is_file():
                logger.info("Downloading med_config.json (needed for BLIP/ImageReward)")
                self.download_file(
                    "https://huggingface.co/THUDM/ImageReward/resolve/main/med_config.json?download=true",
                    med_config_path,
                )

        clip_l_path = scorer_model_dir_path / "CLIP-ViT-L-14.pt"
        if any(x.lower() in ["laion", "chad"] for x in self.cfg.scorer_method):
            if not clip_l_path.is_file():
                logger.info("Downloading CLIP ViT-L-14 model (required for Laion/Chad)")
                self.download_file(
                    "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt?raw=true",
                    clip_l_path,
                )

        clip_b_path = scorer_model_dir_path / "CLIP-ViT-B-32.safetensors"
        if any(x.lower() in ["wdaes"] for x in self.cfg.scorer_method):
            if not clip_b_path.is_file():
                logger.warning(
                    f"CLIP ViT-B-32 model needed by WDAes not found at {clip_b_path}. Please ensure it exists or WDAes might fail."
                )
                # Add download logic if a reliable URL is found
        # --- End explicit file checks ---

        # --- Download files listed in MODEL_DATA for configured scorers ---
        downloaded_this_run = set()  # Track downloads per run to avoid repeats
        for evaluator in self.cfg.scorer_method:
            evaluator_lower = evaluator.lower()
            if evaluator_lower in ["manual", "aestheticv25"]:
                continue

            model_data_entry = MODEL_DATA.get(evaluator_lower)
            if not model_data_entry:
                continue

            # Get all potential filenames associated with this scorer from MODEL_DATA
            filenames_to_check = []
            for key, value in model_data_entry.items():
                if key.endswith("_name") or key == "file_name" or key in ["class", "real", "anime"]:
                    if isinstance(value, str):  # Ensure it's a filename string
                        filenames_to_check.append(value)

            # Check and download each unique required filename
            for filename in set(filenames_to_check):  # Use set for uniqueness
                if not filename or filename in downloaded_this_run:
                    continue  # Skip empty or already handled

                file_path = scorer_model_dir_path / filename
                if not file_path.is_file():
                    # Find URL associated with this filename (might only be on primary key like 'url')
                    url = model_data_entry.get("url")  # Try default 'url' key first
                    if filename != model_data_entry.get("file_name"):  # If it's not the primary file
                        url = model_data_entry.get(
                            f"url_{filename.split('.')[0].lower()}", url
                        )  # Try url_key (e.g., url_class) or fallback to main url

                    if url:
                        logger.info(f"Downloading {evaluator} file: {filename}")
                        self.download_file(url, file_path)
                        downloaded_this_run.add(filename)
                    else:
                        logger.warning(
                            f"Required file '{filename}' for scorer '{evaluator}' not found and no download URL could be determined. Please place it manually in '{scorer_model_dir_path}'."
                        )
                else:
                    downloaded_this_run.add(filename)  # Mark as checked

    def download_file(self, url: str, path: Path):
        """Downloads a file from a URL to the specified path."""
        logger.info(f"Attempting to download file from {url} to {path}...")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
            # Use stream=True for potentially large files
            with requests.get(url, stream=True, timeout=60) as r:  # Add timeout
                r.raise_for_status()  # Check for HTTP errors
                total_size = int(r.headers.get("content-length", 0))
                # Basic progress indication (can be replaced with tqdm if preferred)
                chunk_size = 8192
                downloaded = 0
                with open(path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        downloaded += len(chunk)
                        # Optional: Print progress
                        # done = int(50 * downloaded / total_size) if total_size > 0 else 0
                        # print(f"\r  Downloading [{'>'*done}{'.'*(50-done)}] {downloaded/1024/1024:.1f} MB", end='')
                # print() # Newline after download
            logger.info(f"Download successful. Saved to {path}")
        except requests.exceptions.RequestException as req_err:
            logger.error(f"Failed to download {url}: {req_err}")
            # Optionally delete partial file if it exists
            if path.exists():
                path.unlink(missing_ok=True)
        except Exception as e:
            logger.error(f"An unexpected error occurred during download: {e}")
            if path.exists():
                path.unlink(missing_ok=True)

    def _build_scorer_factory(self, clip_l_path: Path, clip_b_path: Path):
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

    def _load_model(self, evaluator_lower: str):
        """Loads a single scorer model instance on demand."""
        if evaluator_lower in self.model:
            logger.debug(f"Model '{evaluator_lower}' is already loaded.")
            return True

        if evaluator_lower in self._scorers_needing_rembg:
            self._ensure_rembg_session()

        logger.info(f"Lazy loading scorer model: '{evaluator_lower}'")
        scorer_model_dir_path = Path(self.cfg.scorer_model_dir)
        clip_l_path = scorer_model_dir_path / "CLIP-ViT-L-14.pt"
        clip_b_path = scorer_model_dir_path / "CLIP-ViT-B-32.safetensors"
        scorer_factory = self._build_scorer_factory(clip_l_path, clip_b_path)

        if evaluator_lower not in scorer_factory:
            logger.error(f"Unknown scorer '{evaluator_lower}' cannot be lazy-loaded.")
            return False

        config = scorer_factory[evaluator_lower]
        ScorerClass = _get_scorer_class(config.get("class_ref", evaluator_lower))

        if ScorerClass is None:
            logger.error(f"Scorer class for '{evaluator_lower}' not available.")
            return False

        constructor_args = {}
        file_paths_ok = True

        if "device" in inspect.signature(ScorerClass.__init__).parameters:
            constructor_args["device"] = self.cfg.scorer_device.get(evaluator_lower, self.cfg.scorer_default_device)

        if "files" in config:
            for arg_name, model_data_key in config["files"].items():
                model_data_entry = MODEL_DATA.get(evaluator_lower)
                if not model_data_entry:
                    logger.error(f"MODEL_DATA entry missing for '{evaluator_lower}'.")
                    file_paths_ok = False
                    break

                filename = model_data_entry.get(model_data_key)
                if not filename:
                    if evaluator_lower == "aestheticv25":
                        continue
                    logger.error(f"Filename key '{model_data_key}' not found for '{evaluator_lower}'.")
                    file_paths_ok = False
                    break

                file_path = scorer_model_dir_path / filename
                if not file_path.is_file():
                    logger.error(f"Required file for '{evaluator_lower}' not found: {file_path}")
                    file_paths_ok = False
                    break
                constructor_args[arg_name] = str(file_path)

        if not file_paths_ok:
            return False

        if "extra_args" in config:
            resolved_extra_args = {}
            for k, v in config["extra_args"].items():
                if k == "rembg_session" and v == "self.rembg_session":
                    if self.rembg_session:
                        resolved_extra_args[k] = self.rembg_session
                else:
                    resolved_extra_args[k] = str(v) if isinstance(v, Path) else v
            constructor_args.update(resolved_extra_args)

        try:
            self.model[evaluator_lower] = ScorerClass(**constructor_args)
            logger.info(f"Successfully lazy-loaded instance for scorer: '{evaluator_lower}'")
            return True
        except Exception as e_init:
            logger.error(
                f"Failed to initialize lazy-loaded instance for '{evaluator_lower}': {e_init}",
                exc_info=True,
            )
            return False

    def _load_all_models(self):
        """Loads instances for all configured scorers using a factory pattern."""
        logger.info("Loading scorer model instances...")
        lazy_load_list = [s.lower() for s in self.cfg.get("scorer_lazy_load_list", [])]
        scorer_model_dir_path = Path(self.cfg.scorer_model_dir)
        clip_l_path = scorer_model_dir_path / "CLIP-ViT-L-14.pt"
        clip_b_path = scorer_model_dir_path / "CLIP-ViT-B-32.safetensors"

        # --- Scorer Factory Configuration ---
        scorer_factory = self._build_scorer_factory(clip_l_path, clip_b_path)
        # --- End Factory Config ---

        # --- Instantiation Loop ---
        for evaluator in self.cfg.scorer_method:
            evaluator_lower = evaluator.lower()
            if evaluator_lower in ["manual", "background_blackness"]:
                continue

            if evaluator_lower in lazy_load_list:
                logger.info(f"Deferring loading of scorer '{evaluator}' due to lazy load list.")
                continue

            logger.info(f"Loading instance for scorer: '{evaluator}'")

            if evaluator_lower not in scorer_factory:
                logger.error(f"Unknown scorer '{evaluator}' defined in config but not found in scorer_factory. Skipping.")
                continue

            config = scorer_factory[evaluator_lower]
            ScorerClass = _get_scorer_class(config.get("class_ref", evaluator_lower))

            if ScorerClass is None:
                logger.error(f"Scorer class for '{evaluator}' not available (possibly failed import). Skipping.")
                continue

            # Prepare constructor arguments
            constructor_args = {}
            file_paths_ok = True

            # 1. Add device
            if "device" in inspect.signature(ScorerClass.__init__).parameters:
                try:
                    constructor_args["device"] = self.cfg.scorer_device.get(evaluator_lower, self.cfg.scorer_default_device)
                except KeyError:
                    logger.error(f"Device config missing for '{evaluator}'. Skipping.")
                    continue

            # 2. Resolve and check file paths
            if "files" in config:
                for arg_name, model_data_key in config["files"].items():
                    try:
                        # Derive filename from MODEL_DATA using the key
                        model_data_entry = MODEL_DATA.get(evaluator_lower)
                        if not model_data_entry:
                            raise KeyError("MODEL_DATA entry missing")

                        filename = None
                        if model_data_key == "file_name":
                            filename = model_data_entry.get("file_name")
                        elif model_data_key == "config_name":
                            filename = model_data_entry.get("config_name")
                        elif model_data_key in ["class", "real", "anime"]:
                            filename = model_data_entry.get(model_data_key)
                        else:
                            filename = model_data_entry.get(model_data_key)

                        if not filename:
                            # Special case: aestheticv25 has no file
                            if evaluator_lower == "aestheticv25" and not config["files"]:
                                logger.debug(f"No file needed for {evaluator_lower}, arg '{arg_name}'.")
                                continue  # Skip adding this arg if no file needed
                            else:
                                raise KeyError(f"Filename key '{model_data_key}' not found in MODEL_DATA for '{evaluator_lower}'")

                        # Use the Path object stored in self.model_path if it's the primary file, otherwise construct path
                        if arg_name in ["model_path", "pathname"] and evaluator_lower in self.model_path:
                            # Use the primary path object already created
                            file_path = self.model_path[evaluator_lower]
                            # Verify filename matches if needed (optional sanity check)
                            if file_path.name != filename:
                                logger.warning(
                                    f"Filename mismatch for {evaluator_lower} arg {arg_name}: Expected {filename}, Path has {file_path.name}. Using path."
                                )
                        else:
                            # Construct path for secondary files (like config, or NOAI parts)
                            file_path = scorer_model_dir_path / filename

                        # Check existence
                        if not file_path.is_file():
                            logger.error(f"Required file for '{evaluator}', arg '{arg_name}' not found: {file_path}")
                            file_paths_ok = False
                            break
                        constructor_args[arg_name] = str(file_path)  # Pass path as string

                    except KeyError as e:
                        logger.error(f"Config error resolving file for '{evaluator}', arg '{arg_name}': {e}")
                        file_paths_ok = False
                        break
                    except Exception as e_path:
                        logger.error(f"Error resolving path for '{evaluator}', arg '{arg_name}': {e_path}")
                        file_paths_ok = False
                        break

            if not file_paths_ok:
                continue  # Skip if files missing

            # 3. Add extra arguments
            if "extra_args" in config:
                resolved_extra_args = {}
                for k, v in config["extra_args"].items():
                    # Special handling for passing the rembg_session
                    if k == "rembg_session" and v == "self.rembg_session":
                        if self.rembg_session:
                            resolved_extra_args[k] = self.rembg_session
                        else:
                            logger.warning(f"rembg_session not available for '{evaluator}', but it was requested.")
                    else:
                        # Original logic for other args
                        resolved_extra_args[k] = str(v) if isinstance(v, Path) else v
                constructor_args.update(resolved_extra_args)

            if evaluator_lower in {"hybridnoise", "hybridnoise_fullimg"}:
                constructor_args["kernel_size"] = self.cfg.get("hybridnoise_kernel_size", 3)
                constructor_args["noise_threshold"] = self.cfg.get("hybridnoise_noise_threshold", 20.0)
                if evaluator_lower == "hybridnoise":
                    constructor_args["color_tolerance"] = self.cfg.get("hybridnoise_color_tolerance", 30)

            # 4. Instantiate
            try:
                logger.debug(f"Instantiating {ScorerClass.__name__} with args: {constructor_args}")
                self.model[evaluator_lower] = ScorerClass(**constructor_args)  # Store instance using lowercase key
                logger.info(f"Successfully loaded instance for scorer: '{evaluator}'")
            except Exception as e_init:
                logger.error(
                    f"Failed to initialize instance for '{evaluator}': {e_init}",
                    exc_info=True,
                )
        # --- End Instantiation Loop ---

    async def score(self, image: Image.Image, prompt: str, name: str | None = None) -> float:
        values: list[float] = []
        scorer_weights: list[float] = []
        self.last_scorer_results = {}  # Reset for this image
        logger.info("Entering score method.")

        def show_image():
            try:
                image.show()
            except Exception as e:
                logger.error(f"Error displaying image: {e}")

        # --- Scoring Loop ---
        for evaluator in self.cfg.scorer_method:
            # --- Manual Scoring Path ---
            if evaluator == "manual":
                threading.Thread(target=show_image, daemon=True).start()
                individual_eval_score = await asyncio.to_thread(self.get_user_score)
                if individual_eval_score == -1.0:
                    return -1.0

                weight = self.cfg.scorer_weight.get(evaluator, 1.0)
                values.append(individual_eval_score)
                scorer_weights.append(weight)

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
        """Prompts the user for a fake average score during an override."""
        logger.warning("Score override activated!")
        fake_score = 0.0
        while True:
            # Use a slightly different prompt to indicate context
            fake_score_input = input("\tOVERRIDE: Enter the final average score for this entire iteration (0-10): ")
            if fake_score_input:
                try:
                    fake_score = float(fake_score_input)
                    if 0 <= fake_score <= 10:
                        logger.info(f"Using fake average score: {fake_score:.4f}")
                        return fake_score  # Return the validated fake score
                    else:
                        logger.warning("Invalid score. Please enter a number between 0 and 10.")
                except ValueError:
                    logger.warning("Invalid input. Please enter a number.")
            else:
                logger.warning("Input cannot be empty.")

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
        system = platform.system()

        try:
            if system == "Windows":
                subprocess.run(["start", str(image_path)], shell=True, check=True)
            elif system == "Linux":
                if "microsoft-standard" in platform.uname().release:
                    if not hasattr(self, "wsl_instructions_printed"):
                        logger.warning(
                            "Install xdg-open-wsl from https://github.com/cpbotha/xdg-open-wsl or image opening will fail."
                        )
                        self.wsl_instructions_printed = True  # Set a flag to avoid printing multiple times
                subprocess.run(["xdg-open", str(image_path)], check=True)
            elif system == "Darwin":  # macOS
                subprocess.run(["open", str(image_path)], check=True)
            else:
                logger.warning("Automatic image opening not supported on '%s'. Open manually: %s", system, image_path)
        except FileNotFoundError:
            logger.error("Could not find default image viewer. Ensure it is installed/configured.")
        except (subprocess.CalledProcessError, OSError) as e:
            logger.error("Error opening image: %s", e)
            logger.warning("Try opening the image manually: %s", image_path)

    @staticmethod
    def get_user_score() -> float:
        while True:
            user_input = input("\n\tPlease enter the score for the shown image (a number between 0 and 10)\n\t> ")

            # Cheat code handling
            if user_input == "OVERRIDE_SCORE":  # Check for the cheat code
                return -1.0  # Signal override to batch_score

            # Input validation
            if not user_input.replace(".", "", 1).isdigit():  # Allow one decimal point
                logger.warning("Invalid input. Please enter a number between 0 and 10.")
                continue

            try:
                score = float(user_input)
                if 0 <= score <= 10:
                    return score
                else:
                    logger.warning("Invalid input. Please enter a number between 0 and 10.")
            except ValueError:
                logger.warning("Invalid input. Please enter a number between 0 and 10.")
