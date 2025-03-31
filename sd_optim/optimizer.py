# optimizer.py - Version 1.0

import os
import shutil
import asyncio
import logging
import requests

from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict
from PIL import Image, PngImagePlugin

from sd_optim.bounds import ParameterHandler  # Use new class
from sd_optim.generator import Generator
from sd_optim.merger import Merger
from sd_optim.prompter import Prompter
from sd_optim.scorer import AestheticScorer
# from sd_optim import utils  # Removed: No longer needed

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

PathT = os.PathLike


@dataclass
class Optimizer:
    cfg: DictConfig
    best_rolling_score: float = 0.0

    def __post_init__(self) -> None:
        self.bounds_initializer = ParameterHandler(self.cfg)
        self.generator = Generator(self.cfg.url, self.cfg.batch_size, self.cfg.webui)
        self.merger = Merger(self.cfg)
        self.scorer = AestheticScorer(self.cfg, {}, {}, {})
        self.prompter = Prompter(self.cfg)
        self.iteration = 0
        self.best_model_path = None # This seems unused now, merger handles best path
        self.cache = {} # FIXED: Added cache initialization

        # import artist inside
        from sd_optim.artist import Artist
        self.artist = Artist(self)

    def init_params(self) -> Dict:
        with open_dict(self.cfg):
            self.cfg.optimization_guide.setdefault("custom_bounds", {})
            self.cfg.optimization_guide.setdefault("components", [])

        # Check if components are specified
        if not self.cfg.optimization_guide.components:
            raise ValueError("No components specified for optimization in the configuration.")

        return self.bounds_initializer.get_bounds(
            self.cfg.optimization_guide.custom_bounds,
        )

    async def sd_target_function(self, params: Dict) -> float: # Now async and takes Dict
        self.iteration += 1
        iteration_type = (
            "warmup" if self.iteration <= self.cfg.optimizer.init_points else "optimization"
        )

        if self.iteration in {1, self.cfg.optimizer.init_points + 1}:
            logger.info("\n" + "-" * 10 + f" {iteration_type} " + "-" * 10 + ">")

        logger.info(f"\n{iteration_type} - Iteration: {self.iteration}")

        # Assemble parameters using bounds_initializer REMOVED, we use params directly
        logger.info(f"Hyperparameters for Iteration {self.iteration}: {params}")

        # Update the output file name with the current iteration
        self.merger.create_model_out_name(self.iteration)

        # Unload the currently loaded model
        r = requests.post(url=f"{self.cfg.url}/bbwm/unload-model", params={"webui": self.cfg.webui, "url": self.cfg.url})
        r.raise_for_status()

        # Handle different optimization modes
        try:
            if self.cfg.optimization_mode == "merge":
                # Perform model merging, passing only params and cache
                model_path = self.merger.merge(
                    params=params,
                    cache=self.cache
                    # save_best=False implicitly uses self.merger.output_file
                )
            elif self.cfg.optimization_mode == "layer_adjust":
                # Assuming layer_adjust takes params and cfg
                model_path = self.merger.layer_adjust(params, self.cfg)
            elif self.cfg.optimization_mode == "recipe":
                 # Call the dedicated recipe_optimization method if it exists in Merger
                 # Assuming it takes params, output_path, device, cache
                model_path = self.merger.recipe_optimization(
                    params,
                    self.merger.output_file, # Pass the target output path
                    self.cfg.device,
                    self.cache
                )
            else:
                raise ValueError(f"Invalid optimization mode: {self.cfg.optimization_mode}")
        except Exception as e:
            logger.error(f"Error during model processing (merge/adjust/recipe): {e}", exc_info=True) # More specific log
            return 0.0  # Return default score on failure

        # Send a request to the API to load the merged model
        r = requests.post(url=f"{self.cfg.url}/bbwm/load-model",
                      json={"model_path": str(model_path), "webui": self.cfg.webui, "url": self.cfg.url})
        r.raise_for_status()

        # Generate images and score *asynchronously*
        scores = []
        norm = []
        payloads, paths = self.prompter.render_payloads(self.cfg.batch_size)
        payloads_iter = iter(payloads)  # Create an iterator for payloads
        try:
            async for image in self.generator.generate(next(payloads_iter), self.cfg): # Pass cfg
                gen_path = paths[len(scores)]  # Match path with image by current scores length
                payload = payloads[len(scores)]
                try:
                    score = self.scorer.score(image, payload["prompt"])
                except Exception as e:
                    logger.error(f"Error during scoring: {e}", exc_info=True)
                    score = 0.0  # Assign default score

                if self.cfg.save_imgs:
                    img_path = self.save_img(image, gen_path, score, self.iteration, len(scores), payload)
                    if img_path is None:  # Check if save_img failed
                        logger.warning(f"Failed to save image. Skipping...")
                        continue

                if "score_weight" in payload:
                    norm.append(payload["score_weight"])
                else:
                    norm.append(1.0)
                scores.append(score)
                print(f"{gen_path}-{len(scores) -1} {score:4.3f}")
        except StopIteration:
            logger.warning("Ran out of payloads before generating all images.")
        except Exception as e:
            logger.error(f"Error during image generation/scoring: {e}", exc_info=True)
            return 0.0 # Return a default value

        # Calculate the average score.  Handle the case where no images were generated.
        avg_score = self.scorer.average_calc(scores, norm, self.cfg.img_average_type) if scores else 0.0

        # Update best score and handle best model saving
        self.update_best_score(params, avg_score)

        # Collect data for visualization
        self.artist.collect_data(avg_score, params, params) # Modified

        logger.info(f"Average Score for Iteration: {avg_score}")
        return avg_score

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
            return None

        return img_path

    def image_path(self, name: str, score: float, it: int, batch_n: int) -> Path:
        return Path(
            HydraConfig.get().runtime.output_dir,
            "imgs",
            f"{it:03}-{batch_n:02}-{name}-{score:4.3f}.png",
        )

    def update_best_score(self, params: Dict, avg_score: float): # Changed Dict
        logger.info(f"{'-' * 10}\nRun score: {avg_score}")

        for param_name, param_value in params.items():
            logger.info(f"{param_name}: {param_value}")

        if avg_score > self.best_rolling_score:
            logger.info("\n NEW BEST!")
            self.best_rolling_score = avg_score

            # Check if a previous best model exists and delete it BEFORE updating the filename
            if os.path.exists(self.merger.best_output_file):
                os.remove(self.merger.best_output_file)
                logger.info(f"Deleted previous best model: {self.merger.best_output_file}")

            # Update the best model filename
            self.merger.create_best_model_out_name(self.iteration)

            # Move the current model to the new best model filename
            shutil.move(self.merger.output_file, self.merger.best_output_file)
            logger.info(f"Saved new best model as: {self.merger.best_output_file}")

            Optimizer.save_best_log(params, self.iteration) # Changed
        else:
            # Delete the current iteration's model file if it's not the best
            if os.path.exists(self.merger.output_file):
                os.remove(self.merger.output_file)
                logger.info(f"Deleted non-best model: {self.merger.output_file}")

    @abstractmethod
    async def optimize(self) -> None: # Changed
        raise NotImplementedError("Not implemented")

    @abstractmethod
    async def postprocess(self) -> None: # Changed
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def validate_optimizer_config(self) -> bool:
        """Validate optimizer-specific configuration"""
        raise NotImplementedError()

    @abstractmethod
    def get_best_parameters(self) -> Dict:
        """Return best parameters found during optimization."""
        raise NotImplementedError()

    @abstractmethod
    def get_optimization_history(self) -> List[Dict]:
        """Return history of optimization attempts."""
        raise NotImplementedError()

    @staticmethod
    def save_best_log(params: Dict, iteration: int) -> None: # Changed Dict
        """Saves the best hyperparameters and iteration number to a log file."""
        logger.info("Saving best.log")
        with open(
                Path(HydraConfig.get().runtime.output_dir, "best.log"),
                "w",
                encoding="utf-8",
        ) as f:
            f.write(f"Best Iteration: {iteration}.\n\n")
            f.write(str(params)) # Changed
            f.write("\n")
