import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict
from PIL import Image, PngImagePlugin
from sd_webui_bayesian_merger.models.Laion import Laion as AES
from sd_webui_bayesian_merger.models.ImageReward import ImageReward as IMGR
from sd_webui_bayesian_merger.models.CLIPScore import CLIPScore as CLP
from sd_webui_bayesian_merger.models.BLIPScore import BLIPScore as BLP
from sd_webui_bayesian_merger.models.HPSv2 import HPSv2 as HPS
from sd_webui_bayesian_merger.models.PickScore import PickScore as PICK
from sd_webui_bayesian_merger.models.WDAes import WDAes as WDA
from sd_webui_bayesian_merger.models.ShadowScore import ShadowScore as SS
from sd_webui_bayesian_merger.models.CafeScore import CafeScore as CAFE
from sd_webui_bayesian_merger.models.NoAIScore import NoAIScore as NOAI

MODEL_DATA = {
    "laion": {
        "url": "https://github.com/grexzen/SD-Chad/blob/main/sac+logos+ava1-l14-linearMSE.pth?raw=true",
        "file_name": "Laion.pth",
    },
    "chad": {
        "url": "https://github.com/grexzen/SD-Chad/blob/main/chadscorer.pth?raw=true",
        "file_name": "Chad.pth",
    },
    "WDAES": {
        "url": "https://huggingface.co/hakurei/waifu-diffusion-v1-4/resolve/main/models/aes-B32-v0.pth?download=true",
        "file_name": "WD_Aes.pth",
    },
    "ImageReward": {
        "url": "https://huggingface.co/THUDM/ImageReward/resolve/main/ImageReward.pt?download=true",
        "file_name": "ImageReward.pt",
    },
    "CLIP": {
        "url": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt?raw=true",
        "file_name": "CLIP-ViT-L-14.pt",
    },
    "BLIP": {
        "url": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large.pth?raw=true",
        "file_name": "BLIP_Large.safetensors",
    },
    "HPSV2": {
        "url": "https://huggingface.co/xswu/HPSv2/resolve/main/HPS_v2.1_compressed.pt?download=true",
        "file_name": "HPS_v2.1.pt",
    },
    "PickScore": {
        "url": "https://huggingface.co/yuvalkirstain/PickScore_v1/resolve/main/model.safetensors?download=true",
        "file_name": "Pick-A-Pic.safetensors",
    },
    "ShadowV2": {
        "url": "https://huggingface.co/shadowlilac/aesthetic-shadow-v2/resolve/main/model.safetensors?download=true",
        "file_name": "ShadowV2.safetensors",
    },
    "CAFE": {
        "url": "https://huggingface.co/cafeai/cafe_aesthetic/resolve/3bca27c5c0b6021056b1e84e5a18cf1db9fe5d4c/model.safetensors?download=true",
        "file_name": "Cafe.safetensors",
    },
    "CLASS": {
        "url": "https://huggingface.co/cafeai/cafe_style/resolve/d5ae1a7ac05a12ab84732c25f2ea7225d35ac81b/model.safetensors?download=true",
        "file_name": "CLASS.safetensors",
    },
    "REAL": {
        "url": "https://huggingface.co/Sumsub/Sumsub-ffs-synthetic-2.0/resolve/main/synthetic.pt?download=true",
        "file_name": "REAL.pt",
    },
    "ANIME": {
        "url": "https://huggingface.co/saltacc/anime-ai-detect/resolve/e175bb6b5e19cda40bc6c9ad85b138ee7c7ce23a/model.safetensors?download=true",
        "file_name": "ANIME.safetensors",
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
                        # Use MODEL_DATA to get file name
                        self.scorer_model_name[evaluator] = MODEL_DATA[evaluator]["file_name"]
                        self.model_path[evaluator] = Path(
                            self.cfg.scorer_model_dir,
                            self.scorer_model_name[evaluator],
                        )
                else:
                    self.scorer_model_name[evaluator] = 'NOAI pipeline'
                    self.model_path[evaluator] = {}
                    self.model_path[evaluator]['class'] = Path(
                        self.cfg.scorer_model_dir,
                        "Class.safetensors",
                    )
                    self.model_path[evaluator]['real'] = Path(
                        self.cfg.scorer_model_dir,
                        "Real.pt",
                    )
                    self.model_path[evaluator]['anime'] = Path(
                        self.cfg.scorer_model_dir,
                        "Anime.safetensors",
                    )

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
            "ir": lambda: IMGR(self.model_path["ir"], med_config, self.cfg.scorer_device["ir"]),
            "laion": lambda: AES(self.model_path["laion"], self.model_path['clip'], self.cfg.scorer_device["laion"]),
            "chad": lambda: AES(self.model_path["chad"], self.model_path['clip'], self.cfg.scorer_device["chad"]),
            "hpsv2": lambda: HPS(self.model_path["hpsv2"], self.cfg.scorer_device["hpsv2"]),
            "pick": lambda: PICK(self.model_path["pick"], self.cfg.scorer_device["pick"]),
            "shadow": lambda: SS(self.model_path["shadow"], self.cfg.scorer_device["shadow"]),
            "cafe": lambda: CAFE(self.model_path["cafe"], self.cfg.scorer_device["cafe"]),
            "noai": lambda: NOAI(self.model_path["noai"]['class'], self.model_path["noai"]['real'], self.model_path["noai"]['anime'], device=self.cfg.scorer_device["noai"]),
        }

        for evaluator in self.cfg.scorer_method:
            if evaluator != 'manual':
                print(f"Loading {self.scorer_model_name[evaluator]}")
                if evaluator in model_loaders:
                    self.model[evaluator] = model_loaders[evaluator]()  # Call the appropriate loading function

    def score(self, image: Image.Image, prompt) -> float:
        values = []
        scorer_weights = []
        for evaluator in self.cfg.scorer_method:
            scorer_weights.append(int(self.cfg.scorer_weight[evaluator]))
            if evaluator == 'manual':
                # in manual mode, we save a temp image first then request user input
                tmp_path = Path(Path.cwd(), "tmp.png")
                image.save(tmp_path)
                self.open_image(tmp_path)
                values.append(self.get_user_score())
                tmp_path.unlink()  # remove temporary image
            else:
                values.append(self.model[evaluator].score(prompt, image))

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
    ) -> Tuple[List[float], List[float]]:  # Update return type hint
        scores = []
        norm = []  # Restore the norm list

        for i, (img, name, payload) in enumerate(zip(images, payload_names, payloads)):
            score = self.score(img, payload["prompt"])
            if self.cfg.save_imgs:
                self.save_img(img, name, score, it, i, payload)

            if "score_weight" in payload:
                norm.append(payload["score_weight"])
            else:
                norm.append(1.0)
            scores.append(score)

            print(f"{name}-{i} {score:4.3f}")

        return scores, norm  # Return both scores and norm

    def average_calc(self, values: List[float], scorer_weights: List[float], average_type: str) -> float:
        norm = 0
        for weight in scorer_weights:  # Use scorer_weights in the loop
            norm += weight
        avg = 0
        if average_type == 'geometric':
            avg = 1
        elif average_type == 'arithmetic' or average_type == 'quadratic':
            avg = 0

        for value, weight in zip(values, scorer_weights):
            if average_type == 'arithmetic':
                avg += value * weight
            elif average_type == 'geometric':
                avg *= value ** weight
            elif average_type == 'quadratic':
                avg += (value ** 2) * weight

        if average_type == 'arithmetic':
            avg = avg / norm
        elif average_type == 'geometric':
            avg = avg ** (1 / norm)
        elif average_type == 'quadratic':
            avg = (avg / norm) ** (1 / 2)
        return avg

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
    ) -> Path:
        img_path = self.image_path(name, score, it, batch_n)
        pnginfo = PngImagePlugin.PngInfo()
        for k, v in payload.items():
            pnginfo.add_text(k, str(v))

        image.save(img_path, pnginfo=pnginfo)
        return img_path

    def open_image(self, image_path: Path) -> None:
        system = platform.system()

        if system == "Windows":
            subprocess.run(["start", str(image_path)], shell=True, check=True)
        elif system == "Linux":
            global printWSLFlag
            if ("microsoft-standard" in platform.uname().release) and printWSLFlag == 0:
                print(
                    "Make sure to install xdg-open-wsl from here: https://github.com/cpbotha/xdg-open-wsl otherwise the images will NOT open."
                )
                printWSLFlag = 1
            subprocess.run(["xdg-open", str(image_path)], check=True)
        elif system == "Darwin":  # macOS
            subprocess.run(["open", str(image_path)], check=True)
        else:
            print(
                f"Sorry, we do not support opening images on '{system}' operating system."
            )

    @staticmethod
    def get_user_score() -> float:
        while True:
            try:
                score = float(
                    input(
                        f"\n\tPlease enter the score for the shown image (a number between 0 and 10)\n\t> "
                    )
                )
                if 0 <= score <= 10:
                    return score
                else:
                    print("\tInvalid input. Please enter a number between 0 and 10.")
            except ValueError:
                print("\tInvalid input. Please enter a number between 0 and 10.")
