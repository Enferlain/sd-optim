import asyncio
import platform
import subprocess
import threading
import requests
import logging

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict
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
}

printWSLFlag = 0


@dataclass
class AestheticScorer:
    cfg: DictConfig
    scorer_model_name: Dict
    model_path: Dict
    model: Dict

    def __post_init__(self):
        self.setup_img_saving()

        with open_dict(self.cfg):  # Make sure scorer_weight exists
            self.cfg.scorer_weight = self.cfg.scorer_weight or {}

        self.setup_evaluator_paths()  # Now safe to call this method
        self.get_models()
        self.load_models()

    def setup_img_saving(self):
        if "manual" in self.cfg.scorer_method:
            self.cfg.save_imgs = True

        if self.cfg.save_imgs:
            self.imgs_dir = Path(HydraConfig.get().runtime.output_dir, "imgs")
            self.imgs_dir.mkdir(exist_ok=True)

    def setup_evaluator_paths(self):
        """Sets up model paths and devices for each evaluator."""
        for evaluator in self.cfg.scorer_method:
            if evaluator in ['manual', 'noai']:
                continue

            self.scorer_model_name[evaluator], self.model_path[evaluator] = self.get_evaluator_model_info(evaluator)

            with open_dict(self.cfg):
                self.cfg.scorer_device = self.cfg.scorer_device or {}
                self.cfg.scorer_device.setdefault(evaluator, self.cfg.scorer_default_device)

                self.cfg.scorer_weight = self.cfg.scorer_weight or {}
                self.cfg.scorer_weight.setdefault(evaluator, 1)

    def get_evaluator_model_info(self, evaluator):
        """Determines the model name and path for each evaluator."""
        alt_location = self.cfg.scorer_alt_location

        if alt_location and evaluator in alt_location:
            model_name = alt_location[evaluator]['model_name']
            model_dir = Path(alt_location[evaluator]['model_dir'])
        else:
            model_name = MODEL_DATA.get(evaluator.upper(), MODEL_DATA.get(evaluator.lower(), {})).get("file_name")
            if not model_name:
                raise KeyError(f"Evaluator '{evaluator}' not found in MODEL_DATA")
            model_dir = Path(self.cfg.scorer_model_dir, model_name)

        return model_name, model_dir

    def get_models(self) -> None:
        """Downloads necessary model files if they do not exist."""
        blip_config = Path(self.cfg.scorer_model_dir, 'med_config.json')

        if not blip_config.is_file():
            self.download_file(
                "https://huggingface.co/THUDM/ImageReward/resolve/main/med_config.json?download=true",
                blip_config,
            )

        for evaluator in self.cfg.scorer_method:
            if evaluator in ['manual', 'noai']:
                continue
            # Skip download if URL is None
            if MODEL_DATA[evaluator]["url"] is None:  # Add this check
                continue  # Aestheticv25 doesn't need downloads
            if not self.model_path[evaluator].is_file():
                print(f"Downloading {evaluator.upper()} model")
                url = MODEL_DATA[evaluator]["url"]
                self.download_file(url, self.model_path[evaluator])

        # Download CLIP model if needed by Laion or Chad scorers
        if any(x in ['laion', 'chad'] for x in self.cfg.scorer_method):
            clip_path = Path(self.cfg.scorer_model_dir, "CLIP-ViT-L-14.pt")
            if not clip_path.is_file():
                print("Downloading CLIP model (required for Laion or Chad scorers)")
                self.download_file(
                    "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt?raw=true",
                    clip_path,
                )

    def download_file(self, url: str, path: Path):
        """Downloads a file from a URL to the specified path."""
        r = requests.get(url)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        print(f"Saved into {path}")

    def load_models(self) -> None:
        """Initializes models for each evaluator."""
        med_config = Path(self.cfg.scorer_model_dir, "med_config.json")
        model_loaders = self.get_model_loaders(med_config)

        for evaluator in self.cfg.scorer_method:
            if evaluator == 'manual':
                continue

            print(f"Loading {self.scorer_model_name[evaluator]}")
            try:
                self.model[evaluator] = model_loaders[evaluator]()
                # Verify model initialization
                if not hasattr(self.model[evaluator], 'score'):
                    raise AttributeError(f"{evaluator} missing score method")
                print(f"Successfully initialized {evaluator}")
            except Exception as e:
                logger.error(f"Failed to initialize {evaluator}: {str(e)}")
                raise RuntimeError(f"Critical model init failure in {evaluator}") from e

    # Now place the get_model_loaders method outside load_models
    def get_model_loaders(self, med_config: Path):
        """Creates a dictionary of evaluators mapped to their respective model loading functions."""
        return {
            "wdaes": lambda: WDA(self.model_path["wdaes"], Path(self.cfg.scorer_model_dir, "CLIP-ViT-B-32.safetensors"), self.cfg.scorer_device["wdaes"]),
            "clip": lambda: CLP(self.model_path["clip"], self.cfg.scorer_device["clip"]),
            "blip": lambda: BLP(self.model_path["blip"], med_config, self.cfg.scorer_device["blip"]),
            "imagereward": lambda: IMGR(self.model_path["imagereward"], med_config, self.cfg.scorer_device["imagereward"]),
            "laion": lambda: AES(self.model_path["laion"], self.model_path['clip'], self.cfg.scorer_device["laion"]),
            "chad": lambda: AES(self.model_path["chad"], self.model_path['clip'], self.cfg.scorer_device["chad"]),
            "hpsv21": lambda: HPS(self.model_path["hpsv21"], self.cfg.scorer_device["hpsv21"]),
            "pick": lambda: PICK(self.model_path["pick"], self.cfg.scorer_device["pick"]),
            "shadowv2": lambda: SS(self.model_path["shadowv2"], self.cfg.scorer_device["shadowv2"]),
            "cafe": lambda: CAFE(self.model_path["cafe"], self.cfg.scorer_device["cafe"]),
            "noai": lambda: NOAI(self.model_path["noai"]['class'], self.model_path["noai"]['real'], self.model_path["noai"]['anime'], device=self.cfg.scorer_device["noai"]),
            "cityaes": lambda: CITY(self.model_path["cityaes"], self.cfg.scorer_device["cityaes"]),
            "aestheticv25": lambda: AES25(self.cfg.scorer_device["aestheticv25"]),
        }

    async def score(self, image: Image.Image, prompt, name=None) -> float:
        values = []
        scorer_weights = []
        logger.info("Entering score method.")

        # Check if this image should undergo background color check
        # Replace with your specific model identifiers
        models_to_check = ["negeuler2",]

        should_check_background = name in models_to_check if name else False
        background_check_passed = True

        if should_check_background:
            background_check_passed = self.check_background_color(image)
            if not background_check_passed:
                logger.info(f"Image from {name} failed background color check")

        # Rest of your existing score method...
        def show_image():
            try:
                image.show()
            except Exception as e:
                logger.error(f"Error displaying image: {e}")

        for evaluator in self.cfg.scorer_method:
            individual_eval_score = 0.0 # Score for this evaluator
            weight = 1.0 # Default weight

            if evaluator == 'manual':
                # --- Run blocking input in a separate thread ---
                # We still start the image display thread
                threading.Thread(target=show_image, daemon=True).start()
                # Await the result from the thread running get_user_score
                individual_eval_score = await asyncio.to_thread(self.get_user_score)
                # --- End modification ---

                if individual_eval_score == -1.0:
                    return -1.0 # Pass the override signal up immediately

                weight = self.cfg.scorer_weight.get(evaluator, 1.0)
            else:
                # Non-manual scoring remains synchronous within this loop iteration
                if should_check_background and not background_check_passed:
                    logger.info(f"Assigning 0 score for {evaluator} due to background check failure")
                    individual_eval_score = 0.0
                else:
                    try:
                        # Note: If any *automatic* scorer also uses blocking I/O,
                        # it would need the asyncio.to_thread treatment too.
                        # Assuming they are CPU/GPU bound or use async I/O internally.
                        individual_eval_score = self.model[evaluator].score(prompt, image)
                    except Exception as e:
                        logger.error(f"Error scoring image with {evaluator}: {e}")
                        individual_eval_score = 0.0
                weight = self.cfg.scorer_weight.get(evaluator, 1.0)

            values.append(individual_eval_score)
            scorer_weights.append(weight)

            if self.cfg.scorer_print_individual:
                print(f"{evaluator}:{values[-1]}")

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