# optimizer.py - Version 1.1 (Concurrent Gen/Score)

import os
import shutil
import asyncio # <<< Import asyncio
import logging
import socket

import aiohttp
import requests
import time # <<< Import time for logging durations

from abc import abstractmethod
from dataclasses import dataclass, field
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

logger = logging.getLogger(__name__)

PathT = os.PathLike

@dataclass
class Optimizer:
    cfg: DictConfig
    best_rolling_score: float = 0.0
    param_info: BoundsInfo = field(default_factory=dict, init=False)
    optimizer_pbounds: Dict[str, Union[Tuple[float, float], float, int, List]] = field(default_factory=dict, init=False)
    optimization_start_time: Optional[float] = None # <<< Add start time tracker

    def __post_init__(self) -> None:
        self.bounds_initializer = ParameterHandler(self.cfg)
        self.setup_parameter_space()
        self.generator = Generator(self.cfg.url, self.cfg.batch_size, self.cfg.webui)
        self.merger = Merger(self.cfg)
        self.scorer = AestheticScorer(self.cfg, {}, {}, {})
        self.prompter = Prompter(self.cfg)
        self.iteration = 0
        self.best_model_path = None
        self.cache = {}
        from sd_optim.artist import Artist
        self.artist = Artist(self)

    def setup_parameter_space(self):
        """Generates parameter info and extracts bounds for the optimizer."""
        logger.info("Setting up optimization parameter space...")
        self.param_info = self.bounds_initializer.get_bounds(
            self.cfg.optimization_guide.get("custom_bounds")
        )
        self.optimizer_pbounds = {}
        for param_name, info in self.param_info.items():
            bounds_value = info.get('bounds')
            if bounds_value is None:
                 logger.warning(f"Parameter '{param_name}' missing 'bounds' in info. Skipping for optimizer.")
                 continue
            self.optimizer_pbounds[param_name] = bounds_value
        if not self.optimizer_pbounds:
             logger.error("No optimization bounds were generated. Check optimization_guide.yaml and merge method.")
             raise ValueError("Optimization parameter space is empty.")
        logger.info(f"Prepared {len(self.optimizer_pbounds)} parameters for the optimizer.")

    # --- MODIFIED: Async Task for Generating Images (Producer) ---
    async def _produce_image_task(
        self,
        payload: Dict,
        path_info: str,
        queue: asyncio.Queue,
        semaphore: asyncio.Semaphore,
        session: aiohttp.ClientSession # <<< ADD session parameter
    ):
        """Async task to generate image(s) for a payload and put them on the queue."""
        # <<< PASS session to generator >>>
        img_gen = self.generator.generate(payload, self.cfg, session) # Pass session here
        image_index_in_payload = 0
        async with semaphore:
            logger.info(f"Starting generation for payload '{path_info}'...")
            try:
                async for image in img_gen:
                    unique_path_info = f"{path_info}_{image_index_in_payload}"
                    logger.debug(f"Putting image for '{unique_path_info}' onto queue.")
                    await queue.put((image, payload, unique_path_info))
                    image_index_in_payload += 1
                logger.info(f"Finished generation task for payload '{path_info}'.")
            except Exception as e_gen_task:
                 logger.error(f"Error in generation task for payload '{path_info}': {e_gen_task}", exc_info=True)
                 # await queue.put((None, payload, path_info)) # Example of signaling failure

    # --- MODIFIED: sd_target_function to create and pass session ---
    async def sd_target_function(self, params: Dict[str, Any]) -> Optional[float]:
        self.iteration += 1
        iteration_start_time = time.time()
        iteration_type = (
            "warmup" if self.iteration <= self.cfg.optimizer.init_points else "optimization"
        )
        if self.iteration in {1, self.cfg.optimizer.init_points + 1}:
            logger.info(f"\n{'-'*10} Starting {iteration_type} Phase {'-'*10}>")
        logger.info(f"\n--- {iteration_type} - Iteration: {self.iteration} ---")
        logger.info(f"Optimizer proposed parameters: {params}")

        # Update the output file name for this iteration
        self.merger.create_model_out_name(self.iteration)

        # --- Unload previous model (keep as is, assumes synchronous is okay here) ---
        try:
            api_url = f"{self.cfg.url}/sd_optim/unload-model"
            response = requests.post(api_url, params={"webui": self.cfg.webui, "target_url": self.cfg.url if self.cfg.webui == 'swarm' else None})
            response.raise_for_status()
            logger.info("Unload model request sent successfully.")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to send unload request to {api_url}: {e}. Continuing...")
        except Exception as e_unl:
             logger.error(f"Unexpected error during unload request: {e_unl}", exc_info=True)

        # --- Perform Merge / Adjust / Recipe (keep as is, assumes synchronous merge) ---
        model_path: Optional[Path] = None
        try:
            start_merge_time = time.time()
            if self.cfg.optimization_mode == "merge":
                model_path = self.merger.merge(
                    params=params, param_info=self.param_info, cache=self.cache, iteration=self.iteration
                )
            elif self.cfg.optimization_mode == "layer_adjust":
                model_path = self.merger.layer_adjust(params, self.cfg)
            elif self.cfg.optimization_mode == "recipe":
                 # Assuming recipe_optimization takes params, path, device, cache
                 logger.warning("Recipe optimization parameter handling needs review for concurrency.")
                 # model_path = self.merger.recipe_optimization(params, self.merger.output_file, self.cfg.device, self.cache)
                 raise NotImplementedError("Recipe optimization concurrency not fully reviewed.")
            else:
                raise ValueError(f"Invalid optimization mode: {self.cfg.optimization_mode}")
            merge_duration = time.time() - start_merge_time
            logger.info(f"Model processing took {merge_duration:.2f} seconds.")

        except Exception as e_merge:
            logger.error(f"Error during model processing (mode: {self.cfg.optimization_mode}): {e_merge}", exc_info=True)
            return 0.0 # Return low score on failure

        if not model_path or not model_path.exists():
             logger.error(f"Model processing failed to produce a valid file at {model_path}")
             return 0.0

        # --- Load new model (keep as is, assumes synchronous is okay) ---
        try:
            api_url = f"{self.cfg.url}/sd_optim/load-model"
            load_payload = {
                 "model_path": str(model_path.resolve()),
                 "webui": self.cfg.webui,
                 "target_url": self.cfg.url if self.cfg.webui == 'swarm' else None
            }
            response = requests.post(api_url, json=load_payload)
            response.raise_for_status()
            logger.info(f"Load model request sent successfully for {model_path.name}.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send load request to {api_url} for {model_path.name}: {e}. Cannot generate images.")
            return 0.0
        except Exception as e_load:
             logger.error(f"Unexpected error during load request: {e_load}", exc_info=True)
             return 0.0

        # --- Setup for Concurrent Generation ---
        start_gen_score_time = time.time()
        scores = []
        norm_weights = []
        payloads, target_paths = self.prompter.render_payloads(self.cfg.batch_size)
        if not payloads: logger.error("Prompter generated no payloads."); return 0.0

        image_queue = asyncio.Queue() # <<< Create the queue
        total_expected_images = len(payloads) # Assumes batch_size handled by prompter

        # Limit concurrent API requests to avoid overwhelming the WebUI
        # Configurable limit, default to a reasonable number like 2 or 3
        concurrency_limit = self.cfg.get("generator_concurrency_limit", 2)
        semaphore = asyncio.Semaphore(concurrency_limit)
        # logger.info(f"Starting concurrent image generation (limit: {concurrency_limit})...")

        # --- V-- MODIFICATION START --V ---

        # --- Configure aiohttp Client (Simplified) ---

        # 1. Configure TCP Connector with Keep-Alive (Simplified)
        keepalive_interval = self.cfg.get("generator_keepalive_interval", 60) # Default 60s
        connector = aiohttp.TCPConnector(
            limit=concurrency_limit + 5,
            limit_per_host=concurrency_limit,
            # REMOVED enable_keepalive=True,
            # Setting keepalive_timeout enables keep-alive automatically
            keepalive_timeout=keepalive_interval, # <<< Set interval here
            # REMOVED socket_options - rely on keepalive_timeout
        )
        logger.info(f"Configured aiohttp TCPConnector: KeepAlive Enabled (Interval: {keepalive_interval}s)")

        # 2. Set a Single High Total Timeout (or None)
        # (This part remains the same)
        total_timeout_seconds = self.cfg.get("generator_total_timeout", 3600)
        client_timeout_setting = None if total_timeout_seconds is None or total_timeout_seconds <= 0 else aiohttp.ClientTimeout(total=total_timeout_seconds)

        if client_timeout_setting:
            logger.info(f"Configured aiohttp ClientSession: Total Timeout = {total_timeout_seconds}s")
        else:
            logger.info("Configured aiohttp ClientSession: Client-side timeout DISABLED.")

        # Create the session context that wraps producer launch and consumer loop
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=client_timeout_setting
        ) as session: # <<< Session created here

            logger.info(f"Starting concurrent image generation (limit: {concurrency_limit})...") # Moved logging here

            # --- Launch Producer Tasks ---
            producer_tasks = []
            for i, payload in enumerate(payloads):
                path_info = target_paths[i]
                task = asyncio.create_task(
                    # <<< Pass the created 'session' object >>>
                    self._produce_image_task(payload, path_info, image_queue, semaphore, session)
                )
                producer_tasks.append(task)

            # --- Consumer Loop (Scoring) ---
            # (The rest of the loop logic remains exactly as you provided)
            images_processed = 0
            images_generated = 0 # Track images successfully put on queue
            avg_score: float = 0.0 # Initialize avg_score
            try:
                logger.info(f"Waiting to score {total_expected_images} images...")
                for i in range(total_expected_images):
                    logger.debug(f"Waiting for image {i+1}/{total_expected_images} from queue...")
                    queue_item = await image_queue.get()
                    # <<< Use images_generated ONLY for successfully dequeued images >>>
                    # images_generated += 1 # Moved this line down

                    if queue_item is None or queue_item[0] is None: # Check for potential error signal
                         logger.warning(f"Received failure signal from queue for item {i+1}. Skipping.")
                         image_queue.task_done()
                         # Don't count as processed if it failed generation
                         continue

                    # Successfully dequeued an image
                    image, current_payload, current_target_base_name = queue_item
                    images_generated += 1 # <<< Increment here

                    logger.info(f"Scoring image {i+1}/{total_expected_images} ('{current_target_base_name}')...")
                    score_start_time = time.time()
                    score = 0.0 # Initialize score for this item
                    try:
                        # Score the image (this might block if manual scoring)
                        score = self.scorer.score(image, current_payload["prompt"], name=current_target_base_name)
                    except Exception as e_score:
                        logger.error(f"Error scoring image '{current_target_base_name}': {e_score}", exc_info=True)
                        score = 0.0 # Assign default score on error
                    finally:
                        image_queue.task_done() # <<< Signal task completion for this image

                    score_duration = time.time() - score_start_time
                    logger.debug(f"Scoring image {i+1} took {score_duration:.2f}s.")

                    # Store score and weight
                    weight = current_payload.get("score_weight", 1.0)
                    scores.append(score)
                    norm_weights.append(weight)
                    images_processed += 1 # <<< Increment processed count here
                    print(f"  Image {i+1}/{total_expected_images} scored: {score:.4f} (Weight: {weight})")

                    # Save image (synchronously for simplicity, could be async)
                    if self.cfg.save_imgs:
                        # Pass unique name from queue item
                        self.save_img(image, current_target_base_name, score, self.iteration, i, current_payload)

            except asyncio.CancelledError:
                 logger.warning("Scoring loop cancelled.")
            except Exception as e_consume:
                logger.error(f"Error during image scoring loop: {e_consume}", exc_info=True)
            finally:
                # --- Wait for Producers and Queue ---
                logger.info("Waiting for all generation tasks to complete...")
                await image_queue.join()
                logger.debug("Image queue joined.")
                results = await asyncio.gather(*producer_tasks, return_exceptions=True)
                for idx, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Generation task {idx} failed: {result}")
                logger.info("All generation tasks finished.")


        # --- ^-- MODIFICATION END --^ ---

        gen_score_duration = time.time() - start_gen_score_time
        # Use 'images_processed' for logging as it reflects successfully scored images
        logger.info(f"Concurrent generation & scoring took {gen_score_duration:.2f} seconds for {images_processed} scored images.")

        # --- Calculate Final Score ---
        if not scores:
            logger.warning("No images were successfully scored.")
            avg_score = 0.0 # Already initialized, but good practice
        else:
            try:
                avg_score = self.scorer.average_calc(scores, norm_weights, self.cfg.img_average_type)
            except Exception as e_avg:
                logger.error(f"Error calculating average score: {e_avg}", exc_info=True)
                avg_score = 0.0

        # --- Update Best Score & Logging ---
        self.update_best_score(params, avg_score)

        # --- Collect Data for Artist ---
        self.artist.collect_data(avg_score, params)

        iteration_duration = time.time() - iteration_start_time
        logger.info(f"Iteration {self.iteration} finished. Average Score: {avg_score:.4f}. Duration: {iteration_duration:.2f}s")
        return avg_score # Return the final calculated or default score

    # --- save_img, image_path, update_best_score remain the same ---
    def save_img(
        self, image: Image.Image, name: str, score: float, it: int, batch_n: int, payload: Dict
    ) -> Optional[Path]:
        # Use batch_n as the image index within the iteration
        img_path = self.image_path(name, score, it, batch_n)
        pnginfo = PngImagePlugin.PngInfo()
        # Ensure payload items are strings for PNG info
        for k, v in payload.items():
            try:
                pnginfo.add_text(str(k), str(v))
            except Exception as e_png:
                logger.warning(f"Could not add key '{k}' to PNG info: {e_png}")

        try:
            # Ensure the directory exists
            img_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(img_path, pnginfo=pnginfo)
            # logger.debug(f"Saved image to {img_path}") # More detailed log if needed
        except (OSError, IOError) as e:
            logger.error(f"Error saving image to {img_path}: {e}")
            return None
        return img_path

    def image_path(self, name: str, score: float, it: int, img_idx: int) -> Path:
        # Use img_idx from the consumer loop
        base_dir = Path(HydraConfig.get().runtime.output_dir)
        imgs_sub_dir = base_dir / "imgs"
        return imgs_sub_dir / f"{it:03}-{img_idx:02}-{name}-{score:4.3f}.png"

    def update_best_score(self, params: Dict, avg_score: float):
        logger.info(f"{'-' * 10}\nRun score: {avg_score}")
        # Format parameters for logging nicely
        param_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items())
        logger.info(f"Parameters: {{{param_str}}}") # Use curly braces for dict-like look

        if avg_score > self.best_rolling_score:
            logger.info("\n NEW BEST!")
            self.best_rolling_score = avg_score

            # Define potential previous best path before updating filename
            potential_prev_best = self.merger.best_output_file

            # Update the best model filename for THIS iteration
            self.merger.create_best_model_out_name(self.iteration)

            # Check if a *different* previous best model exists and delete it
            if potential_prev_best and potential_prev_best != self.merger.best_output_file and potential_prev_best.exists():
                 try:
                      os.remove(potential_prev_best)
                      logger.info(f"Deleted previous best model: {potential_prev_best}")
                 except OSError as e_del:
                      logger.error(f"Error deleting previous best model {potential_prev_best}: {e_del}")

            # Move the current iteration's model to the new best model filename
            try:
                if self.merger.output_file and self.merger.output_file.exists():
                     shutil.move(self.merger.output_file, self.merger.best_output_file)
                     logger.info(f"Saved new best model as: {self.merger.best_output_file}")
                else:
                     logger.warning(f"Output file {self.merger.output_file} does not exist, cannot save as best.")
            except OSError as e_mov:
                 logger.error(f"Error moving {self.merger.output_file} to {self.merger.best_output_file}: {e_mov}")

            # Static method call is correct
            Optimizer.save_best_log(params, self.iteration)
        else:
            # Delete the current iteration's model file if it's not the best and exists
            if self.merger.output_file and self.merger.output_file.exists():
                try:
                    os.remove(self.merger.output_file)
                    logger.info(f"Deleted non-best model: {self.merger.output_file}")
                except OSError as e_del_non:
                    logger.error(f"Error deleting non-best model {self.merger.output_file}: {e_del_non}")

    # --- optimize, postprocess, validate_optimizer_config etc. remain abstract ---
    @abstractmethod
    async def optimize(self) -> None:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    async def postprocess(self) -> None:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def validate_optimizer_config(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def get_best_parameters(self) -> Dict:
        raise NotImplementedError()

    @abstractmethod
    def get_optimization_history(self) -> List[Dict]:
        raise NotImplementedError()

    @staticmethod
    def save_best_log(params: Dict, iteration: int) -> None:
        logger.info("Saving best.log")
        try:
            log_path = Path(HydraConfig.get().runtime.output_dir) / "best.log"
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"Best Iteration: {iteration}.\n\n")
                # Nicer formatting for parameters
                param_lines = [f"{k}: {v}" for k, v in params.items()]
                f.write("\n".join(param_lines))
                f.write("\n")
        except ValueError: # Hydra not initialized
             logger.error("Hydra context not available, cannot save best.log.")
        except Exception as e:
             logger.error(f"Failed to save best.log: {e}")