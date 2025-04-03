# optimizer.py - Version 1.0

import os
import shutil
import asyncio
import logging
import requests

from abc import abstractmethod
from dataclasses import dataclass, field # Added field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from tqdm import tqdm

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict
from PIL import Image, PngImagePlugin

from sd_optim.bounds import ParameterHandler, BoundsInfo
from sd_optim.generator import Generator
from sd_optim.merger import Merger
from sd_optim.prompter import Prompter
from sd_optim.scorer import AestheticScorer
# from sd_optim import utils  # Removed: No longer needed

logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.DEBUG)

PathT = os.PathLike


@dataclass
class Optimizer:
    cfg: DictConfig
    best_rolling_score: float = 0.0
    # Add field for storing the parameter metadata
    param_info: BoundsInfo = field(default_factory=dict, init=False)
    # Add field for bounds specifically formatted for the optimizer lib
    optimizer_pbounds: Dict[str, Union[Tuple[float, float], float, int, List]] = field(default_factory=dict, init=False)


    def __post_init__(self) -> None:
        self.bounds_initializer = ParameterHandler(self.cfg) # Initialize first
        # --- Generate and store parameter info and bounds ---
        self.setup_parameter_space() # Call new helper method

        # --- Initialize other components ---
        self.generator = Generator(self.cfg.url, self.cfg.batch_size, self.cfg.webui)
        self.merger = Merger(self.cfg) # Merger might need cfg for its own init logic now
        self.scorer = AestheticScorer(self.cfg, {}, {}, {}) # Scorer init seems complex, leave as is for now
        self.prompter = Prompter(self.cfg)
        self.iteration = 0
        self.best_model_path = None # Seems unused, merger handles best path naming
        self.cache = {}

        # import artist inside
        from sd_optim.artist import Artist
        self.artist = Artist(self)

    # V1.0 - New method to centralize parameter space setup
    def setup_parameter_space(self):
        """Generates parameter info and extracts bounds for the optimizer."""
        logger.info("Setting up optimization parameter space...")
        # Get the full parameter info including metadata from bounds handler
        # Pass the custom_bounds section from the config
        self.param_info = self.bounds_initializer.get_bounds(
            self.cfg.optimization_guide.get("custom_bounds")
        )

        # Extract only the bounds needed by the optimizer library
        self.optimizer_pbounds = {}
        for param_name, info in self.param_info.items():
            bounds_value = info.get('bounds')
            if bounds_value is None:
                 logger.warning(f"Parameter '{param_name}' missing 'bounds' in info. Skipping for optimizer.")
                 continue
            # Bounds validation happens in ParameterHandler.validate_custom_bounds
            self.optimizer_pbounds[param_name] = bounds_value

        if not self.optimizer_pbounds:
             logger.error("No optimization bounds were generated. Check optimization_guide.yaml and merge method.")
             raise ValueError("Optimization parameter space is empty.")

        logger.info(f"Prepared {len(self.optimizer_pbounds)} parameters for the optimizer.")

    # init_params is no longer needed, setup_parameter_space replaces it
    # def init_params(self) -> Dict: ... REMOVED ...

    # V1.1 - Passes self.param_info to merger.merge
    async def sd_target_function(self, params: Dict[str, Any]) -> float: # params from optimizer
        self.iteration += 1
        iteration_type = (
            "warmup" if self.iteration <= self.cfg.optimizer.init_points else "optimization"
        )
        if self.iteration in {1, self.cfg.optimizer.init_points + 1}:
            logger.info(f"\n{'-'*10} Starting {iteration_type} Phase {'-'*10}>")
        logger.info(f"\n--- {iteration_type} - Iteration: {self.iteration} ---")
        logger.info(f"Optimizer proposed parameters: {params}")

        # Update the output file name for this iteration
        self.merger.create_model_out_name(self.iteration)

        # --- Unload previous model ---
        try:
            # Adjust API call based on latest scripts/api.py endpoint
            api_url = f"{self.cfg.url}/sd_optim/unload-model" # Use prefixed endpoint
            response = requests.post(api_url, params={"webui": self.cfg.webui, "target_url": self.cfg.url if self.cfg.webui == 'swarm' else None})
            response.raise_for_status()
            logger.info("Unload model request sent successfully.")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to send unload request to {api_url}: {e}. Continuing...")
        except Exception as e_unl:
             logger.error(f"Unexpected error during unload request: {e_unl}", exc_info=True)
             # Decide if this is critical? Maybe not.

        # --- Perform Merge / Adjust / Recipe ---
        model_path: Optional[Path] = None
        try:
            start_merge_time = asyncio.get_event_loop().time()
            if self.cfg.optimization_mode == "merge":
                # Call merge, passing optimizer params AND the param_info metadata
                model_path = self.merger.merge(
                    params=params,
                    param_info=self.param_info, # <<< PASS METADATA HERE
                    cache=self.cache
                    # save_best=False implicitly handled by filename logic
                )
            elif self.cfg.optimization_mode == "layer_adjust":
                # layer_adjust likely doesn't need param_info, just params
                model_path = self.merger.layer_adjust(params, self.cfg) # Assuming it's sync for now
            elif self.cfg.optimization_mode == "recipe":
                 # Needs review/update for param_info if recipe params are optimized
                 logger.warning("Recipe optimization mode parameter handling needs review.")
                 # Assuming it takes params, path, device, cache for now
                 model_path = self.merger.recipe_optimization(
                     params,
                     self.merger.output_file,
                     self.cfg.device,
                     self.cache,
                     # param_info=self.param_info # Pass if needed by recipe mode
                 )
            else:
                raise ValueError(f"Invalid optimization mode: {self.cfg.optimization_mode}")
            merge_duration = asyncio.get_event_loop().time() - start_merge_time
            logger.info(f"Model processing took {merge_duration:.2f} seconds.")

        except Exception as e_merge:
            logger.error(f"Error during model processing (mode: {self.cfg.optimization_mode}): {e_merge}", exc_info=True)
            return 0.0 # Return low score on failure

        if not model_path or not model_path.exists():
             logger.error(f"Model processing failed to produce a valid file at {model_path}")
             return 0.0

        # --- Load new model ---
        try:
            # Adjust API call based on latest scripts/api.py endpoint
            api_url = f"{self.cfg.url}/sd_optim/load-model" # Use prefixed endpoint
            load_payload = {
                 "model_path": str(model_path.resolve()), # Send absolute path
                 "webui": self.cfg.webui,
                 "target_url": self.cfg.url if self.cfg.webui == 'swarm' else None
            }
            response = requests.post(api_url, json=load_payload)
            response.raise_for_status()
            logger.info(f"Load model request sent successfully for {model_path.name}.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send load request to {api_url} for {model_path.name}: {e}. Cannot generate images.")
            return 0.0 # Critical failure if load fails
        except Exception as e_load:
             logger.error(f"Unexpected error during load request: {e_load}", exc_info=True)
             return 0.0

        # --- Generate and Score Images ---
        start_gen_score_time = asyncio.get_event_loop().time()
        total_score = 0.0
        images_generated = 0
        scores = []
        norm_weights = [] # Changed variable name for clarity

        # Use prompter to get payloads and target paths
        payloads, target_paths = self.prompter.render_payloads(self.cfg.batch_size)
        if not payloads:
            logger.error("Prompter did not generate any payloads.")
            return 0.0

        img_gen = self.generator.generate(payloads[0], self.cfg) # Start generator for first payload

        try:
            async for i, image in enumerate(img_gen): # Iterate using async for
                if i >= len(payloads): # Safety break if generator yields more than expected
                    logger.warning("Generator yielded more images than payloads requested.")
                    break

                current_payload = payloads[i]
                current_target_base_name = target_paths[i] # Get base name from prompter

                try:
                    # Score the image (assuming scorer is synchronous for now)
                    score = self.scorer.score(image, current_payload["prompt"], name=current_target_base_name) # Pass base name
                except Exception as e_score:
                    logger.error(f"Error scoring image {i}: {e_score}", exc_info=True)
                    score = 0.0 # Assign default score on error

                weight = current_payload.get("score_weight", 1.0)
                scores.append(score)
                norm_weights.append(weight)
                images_generated += 1

                print(f"  Image {i+1}/{len(payloads)} scored: {score:.4f} (Weight: {weight})")

                # Save image (synchronous, maybe make async later if slow)
                if self.cfg.save_imgs:
                    self.save_img(image, current_target_base_name, score, self.iteration, i, current_payload)

                # If generator yields per batch, handle next payload here if needed
                # This assumes generator yields one image at a time matching payloads list

        except Exception as e_gen:
            logger.error(f"Error during image generation/scoring loop: {e_gen}", exc_info=True)
            # Decide return value - maybe partial score or 0.0?
            # Returning 0.0 for now if loop fails significantly.
            return 0.0

        gen_score_duration = asyncio.get_event_loop().time() - start_gen_score_time
        logger.info(f"Image generation & scoring took {gen_score_duration:.2f} seconds for {images_generated} images.")

        # --- Calculate Final Score ---
        if not scores:
            logger.warning("No images were successfully generated or scored.")
            avg_score = 0.0
        else:
            try:
                avg_score = self.scorer.average_calc(scores, norm_weights, self.cfg.img_average_type)
            except Exception as e_avg:
                logger.error(f"Error calculating average score: {e_avg}", exc_info=True)
                avg_score = 0.0

        # --- Update Best Score & Logging ---
        self.update_best_score(params, avg_score)

        # --- Collect Data for Artist ---
        # Pass the proposed params and the final calculated score
        self.artist.collect_data(avg_score, params)

        logger.info(f"Iteration {self.iteration} finished. Average Score: {avg_score:.4f}")
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
