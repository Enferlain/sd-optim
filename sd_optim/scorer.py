import asyncio
import platform
import subprocess
import threading
import requests
import logging

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict, ListConfig
from PIL import Image, PngImagePlugin
from sd_optim.models.Laion import Laion as AES
from sd_optim.models.ImageReward import ImageReward as IMGR
from sd_optim.models.CLIPScore import CLIPScore as CLP
from sd_optim.models.BLIPScore import BLIPScore as BLP
from sd_optim.models.HPSv21 import HPSv21Scorer as HPS
from sd_optim.models.PickScore import PickScore as PICK
from sd_optim.models.WDAes import WDAes as WDA
from sd_optim.models.ShadowScore import ShadowScore as SS
from sd_optim.models.CafeScore import CafeScore as CAFE
from sd_optim.models.NoAIScore import NoAIScore as NOAI
from sd_optim.models.CityAesthetics import CityAestheticsScorer as CITY
from sd_optim.models.AestheticV25 import AestheticV25 as AES25
from sd_optim.models.LumiAnatomy import AnatomyScorer as LUMI

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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
    "luminaflex": { # <<< Our identifier
        "url": None,
        "file_name": "AnatomyFlaws-v6.6_adabeleif_fl_sigmoid_so400m_naflex_efinal_s10K_final.safetensors", # Head filename
        "config_name": "AnatomyFlaws-v6.6_adabeleif_fl_sigmoid_so400m_naflex.config.json" # Config filename
    },
    "lumidinov2l": {
        "url": None,
        "file_name": "AnatomyFlaws-v6.3_adabeleif_fl_sigmoid_dinov2_large_efinal_s10K_final.safetensors",
        "config_name": "AnatomyFlaws-v6.3_adabeleif_fl_sigmoid_dinov2_large.config.json"
    },
    "lumidinov2g": {
        "url": None,
        "file_name": "AnatomyFlaws-v6.4_adabeleif_fl_sigmoid_dinov2_giant_efinal_s10K_final.safetensors",
        "config_name": "AnatomyFlaws-v6.4_adabeleif_fl_sigmoid_dinov2_giant.config.json"
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
        self.model: Dict[str, Any] = {} # Dictionary to hold loaded scorer instances
        self.model_path: Dict[str, Path] = {} # Dictionary to hold Path objects for models

        self.setup_img_saving()

        with open_dict(self.cfg):
            self.cfg.scorer_weight = self.cfg.scorer_weight or {}
            # Ensure scorer_device exists before setup_evaluator_paths uses it
            self.cfg.scorer_device = self.cfg.scorer_device or {}

        self.setup_evaluator_paths() # Populates self.model_path and sets default devices/weights
        self.get_models() # Downloads files if needed
        self._load_all_models() # NEW: Call the loader function

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
            self.imgs_dir = None # Explicitly set to None if disabled
            logger.info("Image saving disabled.")

    def setup_evaluator_paths(self):
        """Sets up model paths (Path objects) and default configs for each configured evaluator."""
        logger.debug("Setting up evaluator paths...")
        scorer_model_dir_path = Path(self.cfg.scorer_model_dir)

        # --- Use open_dict context manager ---
        with open_dict(self.cfg):
            # Ensure scorer_device and scorer_weight exist as dicts, create if missing/null
            if not isinstance(self.cfg.get('scorer_device'), DictConfig):
                 self.cfg.scorer_device = {}
            if not isinstance(self.cfg.get('scorer_weight'), DictConfig):
                 self.cfg.scorer_weight = {}
            # Now self.cfg.scorer_device and self.cfg.scorer_weight are modifiable

            # Make sure self.cfg.scorer_method is iterable (list or ListConfig)
            configured_scorers = self.cfg.get('scorer_method', [])
            if not isinstance(configured_scorers, (list, ListConfig)):
                 logger.warning("scorer_method is not a list, cannot process evaluators.")
                 configured_scorers = []

            for evaluator in configured_scorers:
                # Use .lower() for case-insensitive matching
                evaluator_lower = str(evaluator).lower() # Ensure it's a string first
                if evaluator_lower == 'manual': continue

                model_data_entry = MODEL_DATA.get(evaluator_lower)
                if not model_data_entry:
                    logger.warning(f"No MODEL_DATA entry for '{evaluator}'. Cannot set path or defaults.")
                    continue

                # --- Handle Alternative Locations ---
                alt_location = self.cfg.get("scorer_alt_location", {}) or {} # Default to empty dict
                evaluator_alt_config = alt_location.get(evaluator_lower)
                current_model_dir = scorer_model_dir_path
                primary_filename = model_data_entry.get("file_name")

                if isinstance(evaluator_alt_config, (dict, DictConfig)): # Check if it's dict-like
                    alt_name = evaluator_alt_config.get('model_name')
                    alt_dir_str = evaluator_alt_config.get('model_dir')
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
                         continue # Skip defaults if path fails
                elif evaluator_lower == "aestheticv25":
                    logger.debug(f"No file path needed for '{evaluator}'.")
                else:
                     logger.warning(f"MODEL_DATA for '{evaluator}' missing primary 'file_name'. Cannot set base path.")
                     # Continue to set defaults even if path missing? Or skip? Let's continue for now.

                # --- Set defaults (safe now due to open_dict) ---
                try:
                    # Use .get() on the main cfg object for the default device
                    default_device = self.cfg.get('scorer_default_device', 'cpu')
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
        med_config_path = scorer_model_dir_path / 'med_config.json'
        if any(x.lower() in ['blip', 'imagereward'] for x in self.cfg.scorer_method):
             if not med_config_path.is_file():
                logger.info("Downloading med_config.json (needed for BLIP/ImageReward)")
                self.download_file("https://huggingface.co/THUDM/ImageReward/resolve/main/med_config.json?download=true", med_config_path)

        clip_l_path = scorer_model_dir_path / "CLIP-ViT-L-14.pt"
        if any(x.lower() in ['laion', 'chad'] for x in self.cfg.scorer_method):
            if not clip_l_path.is_file():
                logger.info("Downloading CLIP ViT-L-14 model (required for Laion/Chad)")
                self.download_file("https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt?raw=true", clip_l_path)

        clip_b_path = scorer_model_dir_path / "CLIP-ViT-B-32.safetensors"
        if any(x.lower() in ['wdaes'] for x in self.cfg.scorer_method):
            if not clip_b_path.is_file():
                logger.warning(f"CLIP ViT-B-32 model needed by WDAes not found at {clip_b_path}. Please ensure it exists or WDAes might fail.")
                # Add download logic if a reliable URL is found
        # --- End explicit file checks ---

        # --- Download files listed in MODEL_DATA for configured scorers ---
        downloaded_this_run = set() # Track downloads per run to avoid repeats
        for evaluator in self.cfg.scorer_method:
            evaluator_lower = evaluator.lower()
            if evaluator_lower in ['manual', 'aestheticv25']: continue

            model_data_entry = MODEL_DATA.get(evaluator_lower)
            if not model_data_entry: continue

            # Get all potential filenames associated with this scorer from MODEL_DATA
            filenames_to_check = []
            for key, value in model_data_entry.items():
                if key.endswith("_name") or key == "file_name" or key in ['class', 'real', 'anime']:
                     if isinstance(value, str): # Ensure it's a filename string
                          filenames_to_check.append(value)

            # Check and download each unique required filename
            for filename in set(filenames_to_check): # Use set for uniqueness
                 if not filename or filename in downloaded_this_run: continue # Skip empty or already handled

                 file_path = scorer_model_dir_path / filename
                 if not file_path.is_file():
                     # Find URL associated with this filename (might only be on primary key like 'url')
                     url = model_data_entry.get("url") # Try default 'url' key first
                     if filename != model_data_entry.get("file_name") : # If it's not the primary file
                          url = model_data_entry.get(f"url_{filename.split('.')[0].lower()}", url) # Try url_key (e.g., url_class) or fallback to main url

                     if url:
                         logger.info(f"Downloading {evaluator} file: {filename}")
                         self.download_file(url, file_path)
                         downloaded_this_run.add(filename)
                     else:
                         logger.warning(f"Required file '{filename}' for scorer '{evaluator}' not found and no download URL could be determined. Please place it manually in '{scorer_model_dir_path}'.")
                 else:
                      downloaded_this_run.add(filename) # Mark as checked

    def download_file(self, url: str, path: Path):
        """Downloads a file from a URL to the specified path."""
        logger.info(f"Attempting to download file from {url} to {path}...")
        try:
            path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
            # Use stream=True for potentially large files
            with requests.get(url, stream=True, timeout=60) as r: # Add timeout
                r.raise_for_status() # Check for HTTP errors
                total_size = int(r.headers.get('content-length', 0))
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
             if path.exists(): path.unlink(missing_ok=True)
        except Exception as e:
             logger.error(f"An unexpected error occurred during download: {e}")
             if path.exists(): path.unlink(missing_ok=True)

    def _load_all_models(self):
        """Loads instances for all configured scorers using a factory pattern."""
        logger.info("Loading scorer model instances...")
        # Define paths needed by some factories
        scorer_model_dir_path = Path(self.cfg.scorer_model_dir)
        med_config_path = scorer_model_dir_path / 'med_config.json'
        clip_l_path = scorer_model_dir_path / "CLIP-ViT-L-14.pt"
        clip_b_path = scorer_model_dir_path / "CLIP-ViT-B-32.safetensors"

        # --- Scorer Factory Configuration ---
        scorer_factory = {
            "laion":        {"class": AES, "files": {"model_path": "file_name"}, "extra_args": {"clip_model_path": str(clip_l_path)}},
            "chad":         {"class": AES, "files": {"model_path": "file_name"}, "extra_args": {"clip_model_path": str(clip_l_path)}},
            "wdaes":        {"class": WDA, "files": {"model_path": "file_name"}, "extra_args": {"clip_path": str(clip_b_path)}},
            "clip":         {"class": CLP, "files": {"model_path": "file_name"}},
            "blip":         {"class": BLP, "files": {"model_path": "file_name"}, "extra_args": {"med_config": med_config_path}},
            "imagereward":  {"class": IMGR,"files": {"model_path": "file_name"}, "extra_args": {"med_config": med_config_path}},
            "hpsv21":       {"class": HPS, "files": {"pathname": "file_name"}},
            "pick":         {"class": PICK,"files": {"model_path": "file_name"}},
            "shadowv2":     {"class": SS,  "files": {"model_path": "file_name"}},
            "cafe":         {"class": CAFE,"files": {"model_path": "file_name"}},
            "noai":         {"class": NOAI,"files": {"model_path_class": "class", "model_path_real": "real", "model_path_anime": "anime"}},
            "cityaes":      {"class": CITY,"files": {"pathname": "file_name"}},
            "aestheticv25": {"class": AES25,"files": {"model_path": "file_name"}},
            "luminaflex":  {"class": LUMI,"files": {"model_path": "file_name","config_path": "config_name"}},
            "lumidinov2l": {"class": LUMI,"files": {"model_path": "file_name","config_path": "config_name"}},
            "lumidinov2g": {"class": LUMI,"files": {"model_path": "file_name","config_path": "config_name"}},
            # --- Add other custom scorers like LumiStyle here ---
            # "lumistyle": {
            #     "class": AnatomyScorer if LUMI_ANATOMY_AVAILABLE else None, # Or a different StyleScorer class
            #     "files": {
            #         "model_path": "file_name", # Needs 'file_name' in MODEL_DATA["lumistyle"]
            #         "config_path": "config_name"    # Needs 'config_name' in MODEL_DATA["lumistyle"]
            #      }
            # },
        }
        # --- End Factory Config ---

        # --- Instantiation Loop ---
        for evaluator in self.cfg.scorer_method:
            evaluator_lower = evaluator.lower()
            if evaluator_lower == 'manual': continue

            logger.info(f"Loading instance for scorer: '{evaluator}'")

            if evaluator_lower not in scorer_factory:
                logger.error(f"Unknown scorer '{evaluator}' defined in config but not found in scorer_factory. Skipping.")
                continue

            config = scorer_factory[evaluator_lower]
            ScorerClass = config.get("class")

            if ScorerClass is None:
                logger.error(f"Scorer class for '{evaluator}' not available (possibly failed import). Skipping.")
                continue

            # Prepare constructor arguments
            constructor_args = {}
            file_paths_ok = True

            # 1. Add device
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
                        if not model_data_entry: raise KeyError("MODEL_DATA entry missing")

                        filename = None
                        if model_data_key == "file_name": filename = model_data_entry.get("file_name")
                        elif model_data_key == "config_name": filename = model_data_entry.get("config_name")
                        elif model_data_key in ['class', 'real', 'anime']: filename = model_data_entry.get(model_data_key)
                        else: filename = model_data_entry.get(model_data_key)

                        if not filename:
                            # Special case: aestheticv25 has no file
                            if evaluator_lower == "aestheticv25" and not config["files"]:
                                 logger.debug(f"No file needed for {evaluator_lower}, arg '{arg_name}'.")
                                 continue # Skip adding this arg if no file needed
                            else:
                                 raise KeyError(f"Filename key '{model_data_key}' not found in MODEL_DATA for '{evaluator_lower}'")

                        # Use the Path object stored in self.model_path if it's the primary file, otherwise construct path
                        if arg_name in ["model_path", "pathname"] and evaluator_lower in self.model_path:
                             # Use the primary path object already created
                             file_path = self.model_path[evaluator_lower]
                             # Verify filename matches if needed (optional sanity check)
                             if file_path.name != filename:
                                  logger.warning(f"Filename mismatch for {evaluator_lower} arg {arg_name}: Expected {filename}, Path has {file_path.name}. Using path.")
                        else:
                             # Construct path for secondary files (like config, or NOAI parts)
                             file_path = scorer_model_dir_path / filename

                        # Check existence
                        if not file_path.is_file():
                            logger.error(f"Required file for '{evaluator}', arg '{arg_name}' not found: {file_path}")
                            file_paths_ok = False
                            break
                        constructor_args[arg_name] = str(file_path) # Pass path as string

                    except KeyError as e:
                        logger.error(f"Config error resolving file for '{evaluator}', arg '{arg_name}': {e}")
                        file_paths_ok = False; break
                    except Exception as e_path:
                        logger.error(f"Error resolving path for '{evaluator}', arg '{arg_name}': {e_path}")
                        file_paths_ok = False; break

            if not file_paths_ok: continue # Skip if files missing

            # 3. Add extra arguments
            if "extra_args" in config:
                # Convert Path objects in extra_args to strings if needed by constructor
                resolved_extra_args = {}
                for k, v in config["extra_args"].items():
                     resolved_extra_args[k] = str(v) if isinstance(v, Path) else v
                constructor_args.update(resolved_extra_args)

            # 4. Instantiate
            try:
                logger.debug(f"Instantiating {ScorerClass.__name__} with args: {constructor_args}")
                self.model[evaluator_lower] = ScorerClass(**constructor_args) # Store instance using lowercase key
                logger.info(f"Successfully loaded instance for scorer: '{evaluator}'")
            except Exception as e_init:
                logger.error(f"Failed to initialize instance for '{evaluator}': {e_init}", exc_info=True)
        # --- End Instantiation Loop ---

    async def score(self, image: Image.Image, prompt: str, name: Optional[str] = None) -> float:
        values: List[float] = []
        scorer_weights: List[float] = []
        logger.info("Entering score method.")

        # --- Optional Background Check ---
        # Read settings from config ONCE before the loop
        # Use .get() to safely access nested keys and provide defaults
        background_check_cfg = self.cfg.get("background_check", {})  # Get the whole sub-config or empty dict
        enable_background_check = background_check_cfg.get("enabled", False)  # Default to False if key missing
        payloads_to_check = background_check_cfg.get("payloads", [])  # Get list of payloads or empty list

        # Determine if check applies to THIS image and perform it IF enabled
        run_check_for_this_image = False
        background_check_passed = True  # Assume passes unless check fails
        if enable_background_check and name is not None and name in payloads_to_check:
            logger.debug(f"Background check enabled for payload '{name}'. Performing check...")
            run_check_for_this_image = True
            try:
                background_check_passed = self.check_background_color(image)
                if not background_check_passed:
                    logger.info(f"Image from '{name}' failed background color check.")
                else:
                    logger.debug(f"Image from '{name}' passed background color check.")
            except Exception as e_bc:
                logger.error(f"Error during background check for '{name}': {e_bc}")
                background_check_passed = False  # Treat check error as failure? Or ignore? Let's treat as fail.
        # --- End Optional Background Check ---

        # --- Manual Score Image Display Helper (if needed) ---
        def show_image():
            try:
                image.show()
            except Exception as e:
                logger.error(f"Error displaying image: {e}")

        # --- Scoring Loop ---
        for evaluator in self.cfg.scorer_method:
            individual_eval_score: float = 0.0  # Explicitly type hint
            weight: float = 1.0

            # Check if manual scoring is selected
            if evaluator == 'manual':
                threading.Thread(target=show_image, daemon=True).start()
                individual_eval_score = await asyncio.to_thread(self.get_user_score)
                if individual_eval_score == -1.0:
                    return -1.0  # Exit early on override signal
                weight = self.cfg.scorer_weight.get(evaluator, 1.0)
            else:
                # --- Automatic Scoring ---
                # Check if background check failed FOR THIS specific image
                if run_check_for_this_image and not background_check_passed:
                    logger.info(
                        f"Assigning 0 score for '{evaluator}' due to background check failure for image '{name}'.")
                    individual_eval_score = 0.0
                else:
                    # Proceed with scoring
                    try:
                        # Get the specific scorer instance
                        scorer_instance = self.model.get(evaluator)  # Use .get() for safety
                        if scorer_instance is None:
                            logger.error(f"Scorer instance for '{evaluator}' not found in self.model. Skipping.")
                            individual_eval_score = 0.0  # Assign 0 if scorer missing
                        else:
                            # --- VVV CORRECTED ARGUMENT ORDER VVV ---
                            logger.debug(f"[AestheticScorer] Preparing to call {evaluator}.score.")
                            # Pass image first, then prompt. Use keywords for clarity.
                            individual_eval_score = scorer_instance.score(image=image, prompt=prompt)
                            # --- ^^^ END CORRECTION ^^^ ---
                            logger.debug(f"[AestheticScorer] Call to {evaluator}.score finished.")

                    except Exception as e:
                        logger.error(f"Error scoring image with {evaluator}: {e}", exc_info=True)
                        individual_eval_score = 0.0  # Assign 0 on scoring error

                weight = self.cfg.scorer_weight.get(evaluator, 1.0)
            # --- End Automatic Scoring Path ---

            # Append results for this evaluator
            values.append(individual_eval_score)
            scorer_weights.append(weight)

            if self.cfg.scorer_print_individual:
                score_str = f"{individual_eval_score:.4f}" if isinstance(individual_eval_score, (int, float)) else str(
                    individual_eval_score)
                print(f"{evaluator}:{score_str}")
        # --- End Scoring Loop ---

        # Calculate average only if no override signal was passed up
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
                        return fake_score # Return the validated fake score
                    else:
                        print("\tInvalid score. Please enter a number between 0 and 10.")
                except ValueError:
                    print("\tInvalid input. Please enter a number.")
            else:
                print("\tInput cannot be empty.")

    def average_calc(self, values: List[float], scorer_weights: List[float], average_type: str) -> float:
        # Ensure weights and values match length
        if len(values) != len(scorer_weights):
             logger.error(f"Score calculation error: Mismatched values ({len(values)}) and weights ({len(scorer_weights)}). Using default weights.")
             # Fallback to equal weights if lengths mismatch
             scorer_weights = [1.0] * len(values)

        # Filter out potential None values if errors occurred and weren't handled before
        valid_data = [(v, w) for v, w in zip(values, scorer_weights) if v is not None]
        if not valid_data: return 0.0 # Return 0 if no valid scores

        values, scorer_weights = zip(*valid_data)
        norm = sum(scorer_weights)
        if norm == 0: return 0.0 # Avoid division by zero

        if average_type == 'geometric':
            # Avoid log(0) or negative numbers for geometric mean
            product = 1.0
            total_weight = 0.0
            for value, weight in zip(values, scorer_weights):
                if value > 0:
                    product *= value ** weight
                    total_weight += weight
                else: # Handle non-positive scores - maybe skip or use a floor? Skipping is safer.
                    logger.warning(f"Skipping non-positive score {value} in geometric mean calculation.")
            return product ** (1 / total_weight) if total_weight > 0 else 0.0
        elif average_type == 'arithmetic':
            return sum(value * weight for value, weight in zip(values, scorer_weights)) / norm
        elif average_type == 'quadratic':
            # Ensure values are non-negative for quadratic mean if that's intended
            avg_sq = sum((value ** 2) * weight for value, weight in zip(values, scorer_weights))
            return (avg_sq / norm) ** 0.5 # Use 0.5 for square root
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
                        print(
                            "Make sure to install xdg-open-wsl from here: https://github.com/cpbotha/xdg-open-wsl otherwise the images will NOT open."
                        )
                        self.wsl_instructions_printed = True  # Set a flag to avoid printing multiple times
                subprocess.run(["xdg-open", str(image_path)], check=True)
            elif system == "Darwin":  # macOS
                subprocess.run(["open", str(image_path)], check=True)
            else:
                print(
                    f"Sorry, automatic image opening is not yet supported on '{system}'. Please open the image manually: {image_path}"
                )
        except FileNotFoundError:
            print(
                f"Error: Could not find the default image viewer. Please ensure it's installed and configured correctly.")
        except (subprocess.CalledProcessError, OSError) as e:
            print(f"Error opening image: {e}")
            print(f"Please try opening the image manually: {image_path}")

    @staticmethod
    def get_user_score() -> float:
        while True:
            user_input = input(
                f"\n\tPlease enter the score for the shown image (a number between 0 and 10)\n\t> "
            )

            # Cheat code handling
            if user_input == "OVERRIDE_SCORE":  # Check for the cheat code
                return -1.0  # Signal override to batch_score

            # Input validation
            if not user_input.replace(".", "", 1).isdigit():  # Allow one decimal point
                print("\tInvalid input. Please enter a number between 0 and 10.")
                continue

            try:
                score = float(user_input)
                if 0 <= score <= 10:
                    return score
                else:
                    print("\tInvalid input. Please enter a number between 0 and 10.")
            except ValueError:
                print("\tInvalid input. Please enter a number between 0 and 10.")

    def check_background_color(self, image: Image.Image) -> bool:
        """
        Check if the image has a black (#000000) or white (#ffffff) background.
        Returns True if background is acceptable, False otherwise.
        """
        # Convert image to RGB if it's not already
        if image.mode != 'RGB':
            image = image.convert('RGB')

        width, height = image.size

        # Define sample points (corners and edges)
        sample_points = [
            # Corners
            (0, 0),  # Top-left
            (width - 1, 0),  # Top-right
            (0, height - 1),  # Bottom-left
            (width - 1, height - 1),  # Bottom-right

            # Extra edge samples
            (width // 2, 0),  # Top middle
            (width // 2, height - 1),  # Bottom middle
            (0, height // 2),  # Left middle
            (width - 1, height // 2),  # Right middle
        ]

        # Check sample points
        acceptable_colors = [(0, 0, 0), (255, 255, 255)]  # Black and white
        acceptable_count = 0

        for x, y in sample_points:
            color = image.getpixel((x, y))
            # Allow small tolerance for compression artifacts
            is_black = all(c <= 10 for c in color)
            is_white = all(c >= 245 for c in color)

            if is_black or is_white:
                acceptable_count += 1

        # Consider background acceptable if majority of sample points are black or white
        return acceptable_count >= len(sample_points) * 0.75