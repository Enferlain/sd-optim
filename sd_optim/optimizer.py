# optimizer.py - Version 1.1 (Concurrent Gen/Score)

import os
import shutil
import asyncio # <<< Import asyncio
import logging
import socket

import aiohttp
import requests
import time # <<< Import time for logging durations

from contextlib import suppress
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
        self.iteration = -1
        self.best_model_path = None
        self.cache = {}
#        from sd_optim.artist import Artist
#        self.artist = Artist(self)

    def setup_parameter_space(self):
        """Generates parameter info and extracts bounds for the optimizer."""
        logger.info("Setting up optimization parameter space...")
        self.param_info, self.optimizer_pbounds = self.bounds_initializer.get_bounds(
            self.cfg.optimization_guide.get("custom_bounds")
        )
        self.optimizer_pbounds = {}
        for param_name, info in self.param_info.items():
            bounds_value = info.get('bounds')
            if bounds_value is None:
                 logger.warning(f"Parameter '{param_name}' missing 'bounds' in info. Skipping for optimizer.")
                 continue
            self.optimizer_pbounds[param_name] = bounds_value

        # Optional: Check if optimizer_pbounds is empty and raise error
        if not self.optimizer_pbounds:
             logger.error("No optimization bounds were generated for the optimizer. Check optimization_guide.yaml and merge method.")
             # Decide if this should be fatal or just a warning depending on the optimizer
             raise ValueError("Optimization parameter space for the optimizer is empty.")
        logger.info(f"Prepared {len(self.optimizer_pbounds)} parameters for the optimizer with specific bounds.")

    # --- ADDED: Sequential Producer Coroutine ---
    async def _sequential_producer(
            self,
            payloads: List[Dict],
            target_paths: List[str],
            queue: asyncio.Queue,
            session: aiohttp.ClientSession,
            interrupt_event: asyncio.Event # Shared event for interruption
    ):
        """
        Requests image generation sequentially, putting results onto the queue.
        Starts generation N+1 immediately after receiving image N.
        Checks interrupt_event before starting each new generation.
        """
        logger.info("Sequential Producer started.")
        total_payloads = len(payloads)
        for i in range(total_payloads):
            # Check for interruption BEFORE starting generation
            if interrupt_event.is_set():
                logger.warning(f"Producer: Interrupt detected before starting generation {i}. Stopping.")
                break # Stop producing new requests

            current_payload = payloads[i]
            current_target_base_name = target_paths[i]
            generated_image = None

            # --- REMOVED: async with semaphore: block ---
            logger.info(f"Producer: Requesting generation {i + 1}/{total_payloads} ('{current_target_base_name}')...")
            try:
                img_gen = self.generator.generate(current_payload, self.cfg, session)
                async for image in img_gen:
                    logger.debug(f"Producer: Received image {i} ('{current_target_base_name}'). Putting onto queue.")
                    await queue.put((i, image, current_payload, current_target_base_name))
                    generated_image = image
                    break
                if generated_image is None:
                    logger.warning(f"Producer: Generation task {i} ('{current_target_base_name}') yielded no images.")
                    await queue.put((i, None, current_payload, current_target_base_name))

            except asyncio.CancelledError:
                logger.info(f"Producer: Generation task {i} cancelled.")
                break
            except Exception as e_gen_task:
                logger.error(f"Producer: Error during generation task {i} ('{current_target_base_name}'): {e_gen_task}",
                             exc_info=True)
                await queue.put((i, None, current_payload, current_target_base_name))
            # --- End removal of semaphore block ---

            # Extra check after generation i completes
            if interrupt_event.is_set():
                 logger.warning(f"Producer: Interrupt detected after finishing generation {i}. Stopping.")
                 break

        logger.info("Sequential Producer finished.")
        # Optionally signal completion: await queue.put(None)

    # --- MODIFIED: sd_target_function to use sequential producer ---
    async def sd_target_function(self, params: Dict[str, Any]) -> Optional[float]:
        self.iteration += 1
        iteration_start_time = time.time()
        iteration_type = (
            "warmup" if self.iteration <= self.cfg.optimizer.init_points else "optimization"
        )
        if self.iteration in {1, self.cfg.optimizer.init_points + 1}:
          logger.info(f"\n{'-' * 10} Starting {iteration_type} Phase {'-' * 10}>")

        logger.info(f"\n--- {iteration_type} - Iteration: {self.iteration} ---")
        logger.info(f"Optimizer proposed parameters: {params}")

        # Update the output file name for this iteration
        self.merger.create_model_out_name(self.iteration)

        # --- Unload / Merge / Load (Synchronous parts remain similar) ---
        try:
            api_url = f"{self.cfg.url}/sd_optim/unload-model"
            # Add timeout to synchronous requests
            response = requests.post(api_url, params={"webui": self.cfg.webui,
                                                      "target_url": self.cfg.url if self.cfg.webui == 'swarm' else None},
                                     timeout=30)
            response.raise_for_status()
            logger.info("Unload model request sent successfully.")
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout sending unload request to {api_url}. Continuing...")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to send unload request to {api_url}: {e}. Continuing...")
        except Exception as e_unl:
            logger.error(f"Unexpected error during unload request: {e_unl}", exc_info=True)

        model_path: Optional[Path] = None
        try:
            start_merge_time = time.time()
            if self.cfg.optimization_mode == "merge":
                model_path = self.merger.merge(
                    params=params,
                    param_info=self.param_info, # <<< PASS param_info HERE
                    cache=self.cache,
                    iteration=self.iteration
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

        # Determine queue size: number of concurrent generators + maybe 1 buffer slot
        concurrency_limit = self.cfg.get("generator_concurrency_limit", 2)
        image_queue = asyncio.Queue(maxsize=concurrency_limit)
        interrupt_event = asyncio.Event() # Event for interrupt signal
        total_expected_images = len(payloads)
        final_score_for_optimizer = 0.0 # Default score
        interrupt_triggered = False
        fake_score_value = 0.0 # Value entered by user on override

        # Configure aiohttp session (remains the same)
        keepalive_interval = self.cfg.get("generator_keepalive_interval", 60)
        connector = aiohttp.TCPConnector(
            limit=concurrency_limit,
            limit_per_host=concurrency_limit,
            keepalive_timeout=keepalive_interval,
        )
        logger.info(f"Configured aiohttp TCPConnector: KeepAlive Enabled (Interval: {keepalive_interval}s)")
        total_timeout_seconds = self.cfg.get("generator_total_timeout", 3600) # Timeout for queue.get
        client_timeout_setting = None if total_timeout_seconds is None or total_timeout_seconds <= 0 else aiohttp.ClientTimeout(total=total_timeout_seconds)
        if client_timeout_setting: logger.info(f"Configured aiohttp ClientSession: Total Timeout = {total_timeout_seconds}s")
        else: logger.info("Configured aiohttp ClientSession: Client-side timeout DISABLED.")

        producer_task = None

        try:
            async with aiohttp.ClientSession(
                connector=connector,
                timeout=client_timeout_setting
            ) as session:

                # Launch ONE Sequential Producer Task
                producer_task = asyncio.create_task(
                    self._sequential_producer(
                        payloads,
                        target_paths,
                        image_queue,
                        session,
                        interrupt_event
                    )
                )

                # --- Consumer Loop ---
                images_processed = 0
                logger.info(f"Consumer: Waiting to receive and score up to {total_expected_images} images sequentially...")

                for i in range(total_expected_images):
                    if interrupt_event.is_set():
                         logger.warning(f"Consumer: Interrupt detected before waiting for image {i}. Stopping consumption.")
                         interrupt_triggered = True
                         break

                    logger.debug(f"Consumer: Waiting for image {i} from queue...")
                    try:
                        # Use a timeout slightly longer than typical generation if possible, or the total timeout
                        effective_timeout = total_timeout_seconds if total_timeout_seconds else 3600 # Default 1hr
                        queue_item = await asyncio.wait_for(image_queue.get(), timeout=effective_timeout)
                    except asyncio.TimeoutError:
                         logger.error(f"Consumer: Timeout waiting for image {i} from queue. Stopping.")
                         interrupt_triggered = True
                         interrupt_event.set()
                         break

                    if queue_item is None:
                        logger.info("Consumer: Received end signal from producer.")
                        break # Optional: Handle producer end signal

                    order_index, image, current_payload, current_target_base_name = queue_item

                    if order_index != i:
                         logger.error(f"Consumer: Order mismatch! Expected index {i}, got {order_index}. Stopping.")
                         interrupt_triggered = True
                         interrupt_event.set()
                         image_queue.task_done()
                         break

                    if image is None:
                        logger.warning(f"Consumer: Received failure signal for image {i} ('{current_target_base_name}'). Skipping scoring.")
                        image_queue.task_done()
                        continue

                    # --- Score the received image ---
                    logger.info(f"Consumer: Scoring image {i+1}/{total_expected_images} ('{current_target_base_name}')...")
                    score_start_time = time.time()
                    individual_score = 0.0
                    processed_item = False # Flag to track if we should call task_done
                    try:
                        individual_score = await self.scorer.score(image, current_payload["prompt"], name=current_target_base_name)

                        if individual_score == -1.0:
                            logger.warning(f"Consumer: OVERRIDE_SCORE detected during scoring of image {i}.")
                            interrupt_event.set()
                            fake_score_value = self.scorer.handle_override_prompt()
                            interrupt_triggered = True
                            # --- DO NOT call task_done here anymore ---
                            break # Exit loop, finally block will be skipped for this item

                        # --- Normal Score Processing ---
                        score_duration = time.time() - score_start_time
                        logger.debug(f"Scoring index {i} took {score_duration:.2f}s.")

                        weight = current_payload.get("score_weight", 1.0)
                        scores.append(individual_score)
                        norm_weights.append(weight)
                        print(f"  Image {i+1}/{total_expected_images} scored: {individual_score:.4f} (Weight: {weight})")

                        if self.cfg.save_imgs:
                            self.save_img(image, current_target_base_name, individual_score, self.iteration, i, current_payload)

                        images_processed += 1
                        processed_item = True # Mark that we processed normally

                    except Exception as e_score:
                        logger.error(f"Consumer: Error scoring image '{current_target_base_name}' (index {i}): {e_score}", exc_info=True)
                        # Decide if task_done should be called on error? Usually yes.
                        processed_item = True # Mark as processed even on error? Or handle differently? Let's mark done.
                    finally:
                        # --- Call task_done() ONLY if the loop didn't break early ---
                        if processed_item:
                            image_queue.task_done()
                        # If 'break' happened, processed_item is False, so task_done is skipped here.

                # End of consumer loop

        except asyncio.CancelledError:
             logger.warning("Main task cancelled.")
             if interrupt_event: interrupt_event.set() # Ensure producer stops
        except Exception as e_main:
            logger.error(f"Error during concurrent generation/scoring: {e_main}", exc_info=True)
            if interrupt_event: interrupt_event.set() # Signal producer on error
        finally:
            # --- Cleanup ---
            if producer_task and not producer_task.done():
                logger.info("Cancelling producer task...")
                producer_task.cancel()
                with suppress(asyncio.CancelledError):
                    await producer_task
                logger.info("Producer task cancelled.")
            # Session closes automatically with 'async with'

        gen_score_duration = time.time() - start_gen_score_time
        logger.info(f"Generation & scoring phase took {gen_score_duration:.2f} seconds.")

        # --- Calculate Final Score ---
        if interrupt_triggered:
            logger.info(f"Iteration interrupted by override. Using final score: {fake_score_value:.4f}")
            avg_score = fake_score_value
        elif not scores:
            logger.warning("No images were successfully scored.")
            avg_score = 0.0
        else:
            try:
                avg_score = self.scorer.average_calc(scores, norm_weights, self.cfg.img_average_type)
                logger.info(f"Calculated average score: {avg_score:.4f}")
            except Exception as e_avg:
                logger.error(f"Error calculating average score: {e_avg}", exc_info=True)
                avg_score = 0.0

        # --- Update Best Score & Logging ---
        self.update_best_score(params, avg_score)

        # --- Collect Data for Artist ---
 #       self.artist.collect_data(avg_score, params)

        iteration_duration = time.time() - iteration_start_time
        logger.info(f"Iteration {self.iteration} finished. Final Score for Optimizer: {avg_score:.4f}. Duration: {iteration_duration:.2f}s")

        return avg_score

    # --- save_img, image_path, update_best_score remain the same ---
    def save_img(
        self,
        image: Image.Image,
        name: str, # This is the original payload name base (e.g., 'noob6')
        score: float,
        it: int,
        img_order_index: int, # <<< Use the order index here
        payload: Dict,
    ) -> Optional[Path]:
        # Use batch_n as the image index within the iteration
        img_path = self.image_path(name, score, it, img_order_index)

        pnginfo = PngImagePlugin.PngInfo()
        for k, v in payload.items():
            try: pnginfo.add_text(str(k), str(v))
            except Exception as e_png: logger.warning(f"Could not add key '{k}' to PNG info: {e_png}")
        try:
            img_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(img_path, pnginfo=pnginfo)
        except (OSError, IOError) as e: logger.error(f"Error saving image to {img_path}: {e}"); return None
        return img_path

    def image_path(self, name: str, score: float, it: int, img_order_index: int) -> Path: # <<< Use order index
        base_dir = Path(HydraConfig.get().runtime.output_dir)
        imgs_sub_dir = base_dir / "imgs"
        # Use img_order_index as the sequence number within the iteration
        return imgs_sub_dir / f"{it:03}-{img_order_index:02}-{name}-{score:4.3f}.png"

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
