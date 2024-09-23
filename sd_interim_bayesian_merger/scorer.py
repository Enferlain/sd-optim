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
from sd_interim_bayesian_merger.models.Laion import Laion as AES
from sd_interim_bayesian_merger.models.ImageReward import ImageReward as IMGR
from sd_interim_bayesian_merger.models.CLIPScore import CLIPScore as CLP
from sd_interim_bayesian_merger.models.BLIPScore import BLIPScore as BLP
from sd_interim_bayesian_merger.models.HPSv2 import HPSv2 as HPS
from sd_interim_bayesian_merger.models.PickScore import PickScore as PICK
from sd_interim_bayesian_merger.models.WDAes import WDAes as WDA
from sd_interim_bayesian_merger.models.ShadowScore import ShadowScore as SS
from sd_interim_bayesian_merger.models.CafeScore import CafeScore as CAFE
from sd_interim_bayesian_merger.models.NoAIScore import NoAIScore as NOAI
from sd_interim_bayesian_merger.models.CityAesthetics import CityAesthetics as CITY

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
    "hpsv2": {
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
}

printWSLFlag = 0


@dataclass
class AestheticScorer:
    cfg: DictConfig
    scorer_model_name: Dict
    model_path: Dict
    model: Dict

    def __post_init__(self):
        if "manual" in self.cfg.scorer_method:
            self.cfg.save_imgs = True

        if self.cfg.save_imgs:
            self.imgs_dir = Path(HydraConfig.get().runtime.output_dir, "imgs")
            if not self.imgs_dir.exists():
                self.imgs_dir.mkdir()

        for evaluator in self.cfg.scorer_method:
            if evaluator != 'manual':
                if evaluator != 'noai':
                    if self.cfg.scorer_alt_location is not None and evaluator in self.cfg.scorer_alt_location:
                        self.scorer_model_name[evaluator] = self.cfg.scorer_alt_location[evaluator]['model_name']
                        self.model_path[evaluator] = Path(self.cfg.scorer_alt_location[evaluator]['model_dir'])
                    else:
                        # Use MODEL_DATA to get file name, dynamically adjusting casing
                        if evaluator.upper() in MODEL_DATA:
                            self.scorer_model_name[evaluator] = MODEL_DATA[evaluator.upper()]["file_name"]
                            self.model_path[evaluator] = Path(
                                self.cfg.scorer_model_dir,
                                self.scorer_model_name[evaluator],
                            )
                        elif evaluator.lower() in MODEL_DATA:
                            self.scorer_model_name[evaluator] = MODEL_DATA[evaluator.lower()]["file_name"]
                            self.model_path[evaluator] = Path(
                                self.cfg.scorer_model_dir,
                                self.scorer_model_name[evaluator],
                            )
                        else:
                            raise KeyError(f"Evaluator '{evaluator}' not found in MODEL_DATA")

                with open_dict(self.cfg):
                    if self.cfg.scorer_device is None:
                        self.cfg.scorer_device = {}
                    if evaluator not in self.cfg.scorer_device:
                        self.cfg.scorer_device[evaluator] = self.cfg.scorer_default_device
            with open_dict(self.cfg):
                if self.cfg.scorer_weight is None:
                    self.cfg.scorer_weight = {}
                if evaluator not in self.cfg.scorer_weight:
                    self.cfg.scorer_weight[evaluator] = 1
        if 'clip' not in self.cfg.scorer_method and any(
                x in ['laion', 'chad'] for x in self.cfg.scorer_method):
            self.model_path['clip'] = Path(
                self.cfg.scorer_model_dir,
                CLIP_MODEL="CLIP-ViT-L-14.pt",
            )

        self.get_models()
        self.load_models()

    def get_models(self) -> None:
        blip_config = Path(
            self.cfg.scorer_model_dir,
            'med_config.json',
        )
        if not blip_config.is_file():
            url = "https://huggingface.co/THUDM/ImageReward/resolve/main/med_config.json?download=true"

            r = requests.get(url)
            r.raise_for_status()

            with open(blip_config.absolute(), "wb") as f:
                print(f"saved into {blip_config}")
                f.write(r.content)

        for evaluator in self.cfg.scorer_method:
            if evaluator != 'manual':
                if evaluator != 'noai':
                    if not self.model_path[evaluator].is_file():
                        print(f"You do not have the {evaluator.upper()} model, let me download that for you")
                        # Use MODEL_DATA to get URL
                        url = MODEL_DATA[evaluator]["url"]

                        r = requests.get(url)
                        r.raise_for_status()

                        with open(self.model_path[evaluator].absolute(), "wb") as f:
                            print(f"saved into {self.model_path[evaluator]}")
                            f.write(r.content)
                else:
                    for m_path in self.model_path[evaluator]:
                        if not self.model_path[evaluator][m_path].is_file():
                            url = eval(f"{m_path.upper() + '_URL'}")

                            r = requests.get(url)
                            r.raise_for_status()

                            with open(self.model_path[evaluator][m_path].absolute(), "wb") as f:
                                print(f"saved into {self.model_path[evaluator][m_path]}")
                                f.write(r.content)

                if evaluator == 'wdaes':
                    # Use MODEL_DATA to get file name
                    clip_vit_b_32 = Path(
                        self.cfg.scorer_model_dir,
                        MODEL_DATA["wdaes"]["clip_file_name"],
                    )
                    if not clip_vit_b_32.is_file():
                        print(
                            f"You do not have the CLIP-ViT-B-32 necessary for the wdaes model, let me download that for you")
                        url = "https://huggingface.co/openai/clip-vit-base-patch32/resolve/b527df4b30e5cc18bde1cc712833a741d2d8c362/model.safetensors?download=true"

                        r = requests.get(url)
                        r.raise_for_status()

                        with open(clip_vit_b_32.absolute(), "wb") as f:
                            print(f"saved into {clip_vit_b_32}")
                            f.write(r.content)

        if ('clip' not in self.cfg.scorer_method and
                any(x in ['laion', 'chad'] for x in self.cfg.scorer_method)):
            if not self.model_path['clip'].is_file():
                print(f"You do not have the CLIP(which you need) model, let me download that for you")
                url = "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt?raw=true"

                r = requests.get(url)
                r.raise_for_status()

                with open(self.model_path['clip'].absolute(), "wb") as f:
                    print(f"saved into {self.model_path['clip']}")
                    f.write(r.content)

    def load_models(self) -> None:
        med_config = Path(
            self.cfg.scorer_model_dir,
            "med_config.json"
        )

        # Create a dictionary mapping evaluator names to model loading functions
        model_loaders = {
            "wdaes": lambda: WDA(self.model_path["wdaes"], Path(self.cfg.scorer_model_dir, "CLIP-ViT-B-32.safetensors"), self.cfg.scorer_device["wdaes"]),
            "clip": lambda: CLP(self.model_path["clip"], self.cfg.scorer_device["clip"]),
            "blip": lambda: BLP(self.model_path["blip"], med_config, self.cfg.scorer_device["blip"]),
            "imagereward": lambda: IMGR(self.model_path["imagereward"], med_config, self.cfg.scorer_device["imagereward"]),
            "laion": lambda: AES(self.model_path["laion"], self.model_path['clip'], self.cfg.scorer_device["laion"]),
            "chad": lambda: AES(self.model_path["chad"], self.model_path['clip'], self.cfg.scorer_device["chad"]),
            "hpsv2": lambda: HPS(self.model_path["hpsv2"], self.cfg.scorer_device["hpsv2"]),
            "pick": lambda: PICK(self.model_path["pick"], self.cfg.scorer_device["pick"]),
            "shadowv2": lambda: SS(self.model_path["shadowv2"], self.cfg.scorer_device["shadowv2"]),
            "cafe": lambda: CAFE(self.model_path["cafe"], self.cfg.scorer_device["cafe"]),
            "noai": lambda: NOAI(self.model_path["noai"]['class'], self.model_path["noai"]['real'], self.model_path["noai"]['anime'], device=self.cfg.scorer_device["noai"]),
            "cityaes": lambda: CITY(self.model_path["cityaes"], self.model_path['clip'], self.cfg.scorer_device["cityaes"]),
        }

        for evaluator in self.cfg.scorer_method:
            if evaluator != 'manual':
                print(f"Loading {self.scorer_model_name[evaluator]}")
                if evaluator in model_loaders:
                    self.model[evaluator] = model_loaders[evaluator]()  # Call the appropriate loading function

    def score(self, image: Image.Image, prompt) -> float:
        values = []
        scorer_weights = []
        logger.info("Entering score method.")

        def show_image():
            try:
                image.show()
            except Exception as e:
                logger.error(f"Error displaying image: {e}")

        for evaluator in self.cfg.scorer_method:
            scorer_weights.append(int(self.cfg.scorer_weight[evaluator]))
            if evaluator == 'manual':
                # Display the image in a separate thread
                threading.Thread(target=show_image, daemon=True).start()
                values.append(self.get_user_score())
            else:
                try:
                    values.append(self.model[evaluator].score(prompt, image))
                except Exception as e:
                    logger.error(f"Error scoring image with {evaluator}: {e}")
                    values.append(0.0)

            if self.cfg.scorer_print_individual:
                print(f"{evaluator}:{values[-1]}")

        score = self.average_calc(values, scorer_weights, self.cfg.scorer_average_type)
        return score

    def batch_score(
            self,
            images: List[Image.Image],
            payload_names: List[str],
            payloads: Dict,
            it: int,
    ) -> Tuple[List[float], List[float]]:
        logger.info("Entering batch_score method.")
        scores = []
        norm = []  # Restore the norm list
        fake_score = 0.0

        for i, (img, name, payload) in enumerate(zip(images, payload_names, payloads)):
            score = self.score(img, payload["prompt"])
            if self.cfg.save_imgs:
                img_path = self.save_img(img, name, score, it, i, payload)
                if img_path is None:
                    logger.warning(f"Failed to save image for {name}-{i}")
                    continue  # Skip to the next image

            # Check for override signal and handle it
            if score == -1:
                logger.warning("Score override activated! Using fake average score for the entire batch.")
                while True:
                    fake_score_input = input("\tEnter the fake average score (0-10): ")
                    if fake_score_input:
                        try:
                            fake_score = float(fake_score_input)
                            if 0 <= fake_score <= 10:
                                break
                            else:
                                print("\tInvalid fake score. Please enter a number between 0 and 10.")
                        except ValueError:
                            print("\tInvalid input. Please enter a number between 0 and 10.")
                    else:
                        print("\tInput cannot be empty. Please enter a fake average score.")

                scores = [fake_score] * len(images)
                norm = [1.0] * len(images)
                return scores, norm  # Exit early

            # Proceed with normal scoring
            if "score_weight" in payload:
                norm.append(payload["score_weight"])
            else:
                norm.append(1.0)
            scores.append(score)

            print(f"{name}-{i} {score:4.3f}")

        return scores, norm  # Return both scores and norm

    def average_calc(self, values: List[float], scorer_weights: List[float], average_type: str) -> float:
        norm = sum(scorer_weights)  # Calculate norm directly using sum

        if average_type == 'geometric':
            avg = 1
            for value, weight in zip(values, scorer_weights):
                avg *= value ** weight
            return avg ** (1 / norm)
        elif average_type == 'arithmetic':
            return sum(value * weight for value, weight in zip(values, scorer_weights)) / norm
        elif average_type == 'quadratic':
            avg = sum((value ** 2) * weight for value, weight in zip(values, scorer_weights))
            return (avg / norm) ** (1 / 2)
        else:
            raise ValueError(f"Invalid average type: {average_type}")

    def image_path(self, name: str, score: float, it: int, batch_n: int) -> Path:
        return Path(
            self.imgs_dir,
            f"{it:03}-{batch_n:02}-{name}-{score:4.3f}.png",
        )

    def save_img(
            self,
            image: Image.Image,
            name: str,
            score: float,
            it: int,
            batch_n: int,
            payload: Dict,
    ) -> Optional[Path]:
        img_path = self.image_path(name, score, it, batch_n)
        pnginfo = PngImagePlugin.PngInfo()
        for k, v in payload.items():
            pnginfo.add_text(k, str(v))

        try:
            image.save(img_path, pnginfo=pnginfo)
        except (OSError, IOError) as e:
            logger.error(f"Error saving image to {img_path}: {e}")
            return None  # Return None to signal failure
        return img_path

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