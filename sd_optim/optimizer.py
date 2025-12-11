# optimizer.py - Version 1.1 (Concurrent Gen/Score)
import gc
import os
import shutil
import logging
import aiohttp
import asyncio  # <<< Import asyncio
import time  # <<< Import time for logging durations
import torch

from contextlib import suppress
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from tqdm import tqdm
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict
from PIL import Image, PngImagePlugin

import sd_mecha

from sd_mecha.recipe_nodes import ModelRecipeNode
from sd_optim import utils
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
    optimization_start_time: Optional[float] = None  # Add start time tracker
    completed_trials: int = 0  # To track trials from resumed studies

    def __post_init__(self) -> None:
        # --- STAGE 1: VALIDATE THE ENTIRE CONFIG FIRST ---
        # This is now a clean, clear gatekeeper step.
        utils.validate_run_config(self.cfg)

        # --- STAGE 2: CENTRALIZED CONFIG LOADING ---
        logger.info("Optimizer starting up: Performing centralized config loading...")

        # Now that validation is passed, we can safely get the models_dir.
        # This logic is now explicit and clear right where it's needed.
        models_dir = Path(self.cfg.models_dir).resolve()
        logger.info(f"Using primary models directory from config: {models_dir}")

        # The rest of the __post_init__ is exactly the same as before.
        # It just uses the 'models_dir' variable we defined right here.
        if not self.cfg.model_paths:
            raise ValueError("'model_paths' cannot be empty.")

        representative_model_name = self.cfg.model_paths[0]
        representative_model_path = models_dir / representative_model_name
        if not representative_model_path.exists():
            # Fallback to check if an absolute path was given in the list
            absolute_path_check = Path(representative_model_name)
            if absolute_path_check.is_absolute() and absolute_path_check.exists():
                representative_model_path = absolute_path_check
            else:
                raise FileNotFoundError(
                    f"Representative model not found in models_dir or as an absolute path: {representative_model_path}")

        logger.info(f"Inferring base ModelConfig from: {representative_model_path}")
        rep_model_node = sd_mecha.model(str(representative_model_path))

        # We assert that the node is the specific type we need. This makes the linter happy and the code safer!
        assert isinstance(rep_model_node,
                          ModelRecipeNode), "The representative model must be a file path, not a literal dict."

        with sd_mecha.open_input_dicts(rep_model_node, [models_dir]):
            # Now the linter knows rep_model_node has .state_dict because of the assertion above.
            inferred_sets = sd_mecha.infer_model_configs(rep_model_node.state_dict.keys())
            if not inferred_sets:
                raise ValueError(f"Could not infer a ModelConfig for {representative_model_path}.")
            base_model_config = next(iter(inferred_sets[0]))
            logger.info(f"Inferred base ModelConfig: {base_model_config.identifier}")

        # 1c. Load the custom block config ONCE
        custom_block_config_id = self.cfg.optimization_guide.get("custom_block_config_id")
        custom_block_config = None
        if custom_block_config_id:
            try:
                custom_block_config = sd_mecha.extensions.model_configs.resolve(custom_block_config_id)
                logger.info(f"Successfully loaded custom block config: '{custom_block_config_id}'")
            except ValueError as e:
                logger.warning(f"Could not resolve custom block config '{custom_block_config_id}': {e}")

        # --- STAGE 2: INITIALIZE HELPERS WITH LOADED CONFIGS ---
        logger.info("Initializing helpers with centrally loaded configs...")

        # 2a. Initialize Merger with the configs
        self.merger = Merger(
            cfg=self.cfg,
            base_model_config=base_model_config,
            custom_block_config=custom_block_config,
            models_dir=models_dir
        )

        # 2b. Initialize ParameterHandler with the configs
        self.bounds_initializer = ParameterHandler(
            cfg=self.cfg,
            base_model_config=base_model_config,
            custom_block_config=custom_block_config
        )

        # --- STAGE 3: COMPLETE THE REST OF THE SETUP ---
        self.setup_parameter_space()
        self.generator = Generator(self.cfg.url, self.cfg.batch_size, self.cfg.webui)
        self.scorer = AestheticScorer(self.cfg)
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
            logger.error(
                "No optimization bounds were generated for the optimizer. Check optimization_guide.yaml and merge method.")
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
            interrupt_event: asyncio.Event  # Shared event for interruption
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
                break  # Stop producing new requests

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

    async def sd_target_function(self, params: Dict[str, Any]) -> Optional[float]:
        self.iteration += 1
        # Adjust iteration number for resumed runs ---
        effective_iteration = self.iteration + self.completed_trials
        iteration_start_time = time.time()
        
        iteration_type = (
            "warmup" if effective_iteration <= self.cfg.optimizer.init_points else "optimization"
        )
        if effective_iteration in {1, self.cfg.optimizer.init_points + 1}:
            logger.info(f"\n{'-' * 10} Starting {iteration_type} Phase {'-' * 10}>")

        logger.info(f"\n--- {iteration_type} - Iteration: {effective_iteration} ---")
        logger.info(f"Optimizer proposed parameters: {params}")

        # --- NEW: Configure Session for the entire trial ---
        concurrency_limit = self.cfg.get("generator_concurrency_limit", 2)
        keepalive_interval = self.cfg.get("generator_keepalive_interval", 60)
        total_timeout_seconds = self.cfg.get("generator_total_timeout", 3600)
        
        connector = aiohttp.TCPConnector(limit=concurrency_limit, keepalive_timeout=keepalive_interval)
        timeout_settings = aiohttp.ClientTimeout(total=total_timeout_seconds) if total_timeout_seconds > 0 else None

        # We wrap the ENTIRE trial in this session
        async with aiohttp.ClientSession(connector=connector, timeout=timeout_settings) as session:

            # --- STEP 1: ASYNC UNLOAD ---
            try:
                # OLD: requests.post(...)
                # NEW: Delegate to generator -> adapter
                await self.generator.unload_model(session)
                logger.info("Unload model request processed.")
            except Exception as e_unl:
                # OLD: requests.exceptions handling
                # NEW: Generic catch for async errors
                logger.warning(f"Unload request failed (continuing): {e_unl}")

            model_path: Optional[Path] = None
            
            # --- INDENTATION START: Everything below is now inside 'async with session' ---
            try:
                start_merge_time = time.time()

                effective_iteration = self.iteration + self.completed_trials
                if self.cfg.optimization_mode == "merge":
                    self.merger.output_file = self.merger.create_model_output_name(iteration=effective_iteration)
                    model_path = self.merger.merge(
                        params=params,
                        param_info=self.param_info,
                        cache=self.cache,
                        iteration=effective_iteration
                    )
                elif self.cfg.optimization_mode == "layer_adjust":
                    self.merger.output_file = self.merger.create_model_output_name(iteration=effective_iteration)
                    model_path = self.merger.layer_adjust(params, self.cfg)

                elif self.cfg.optimization_mode == "recipe":
                    model_path = self.merger.recipe_optimization(
                        params=params,
                        param_info=self.param_info,
                        cache=self.cache,
                        iteration=effective_iteration
                    )

                else:
                    raise ValueError(f"Invalid optimization mode: {self.cfg.optimization_mode}")

                merge_duration = time.time() - start_merge_time
                logger.info(f"Model processing took {merge_duration:.2f} seconds.")

            except (ValueError, TypeError, FileNotFoundError) as config_error:
                # These errors indicate a fundamental problem with the user's setup or config.
                logger.error(f"FATAL CONFIGURATION ERROR: {config_error}", exc_info=True)
                logger.error("Halting optimization due to unrecoverable setup error.")
                raise config_error

            except Exception as e:
                # --- THIS IS THE PART WE CHANGE ---
                logger.error(f"A runtime error occurred during the trial: {e}", exc_info=True)
                logger.error("Halting optimization because fail_on_error is enabled.")
                # Instead of returning 0.0, we re-raise the exception.
                raise e
                # --- END OF CHANGE ---

            if not model_path or not model_path.exists():
                error_message = f"CRITICAL: Model processing finished but the output file was not created at '{model_path}'. Halting."
                logger.error(error_message)
                raise RuntimeError(error_message)

            # This is the most critical point to free up VRAM.
            logger.info("Performing immediate post-merge memory cleanup before image generation...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("PyTorch CUDA cache cleared.")

            # --- STEP 3: ASYNC LOAD ---
            try:
                # OLD: requests.post(api_url, ...)
                # NEW: Delegate to generator -> adapter
                await self.generator.load_model(model_path, session)
                logger.info(f"Load model request processed for {model_path.name}.")
            except Exception as e_load:
                logger.error(f"Failed to load model: {e_load}. Cannot generate.")
                raise RuntimeError(f"Critical Load Failure: {e_load}") from e_load

            # --- Setup for Concurrent Generation ---
            start_gen_score_time = time.time()
            scores = []
            norm_weights = []
            payloads, target_paths = self.prompter.render_payloads(self.cfg.batch_size)

            if not payloads:
                logger.error("Prompter generated no payloads.")
                raise RuntimeError("Prompter failed to generate any payloads.")

            # Determine queue size: number of concurrent generators + maybe 1 buffer slot
            image_queue = asyncio.Queue(maxsize=concurrency_limit)
            interrupt_event = asyncio.Event()  # Event for interrupt signal
            total_expected_images = len(payloads)
            final_score_for_optimizer = 0.0  # Default score
            interrupt_triggered = False
            fake_score_value = 0.0  # Value entered by user on override

            # (Connector creation removed here as we reused the outer one)
            
            producer_task = None

            try:
                # REMOVED: async with aiohttp.ClientSession... (We use the outer 'session')

                # Launch ONE Sequential Producer Task
                producer_task = asyncio.create_task(
                    self._sequential_producer(
                        payloads,
                        target_paths,
                        image_queue,
                        session, # <<< PASS THE OUTER SESSION
                        interrupt_event
                    )
                )

                # --- Consumer Loop ---
                images_processed = 0
                logger.info(
                    f"Consumer: Waiting to receive and score up to {total_expected_images} images sequentially...")

                for i in range(total_expected_images):
                    if interrupt_event.is_set():
                        logger.warning(
                            f"Consumer: Interrupt detected before waiting for image {i}. Stopping consumption.")
                        interrupt_triggered = True
                        break

                    logger.debug(f"Consumer: Waiting for image {i} from queue...")
                    try:
                        # Use a timeout slightly longer than typical generation if possible, or the total timeout
                        effective_timeout = total_timeout_seconds if total_timeout_seconds else 3600  # Default 1hr
                        queue_item = await asyncio.wait_for(image_queue.get(), timeout=effective_timeout)
                    except asyncio.TimeoutError:
                        logger.error(f"Consumer: Timeout waiting for image {i} from queue. Stopping.")
                        interrupt_triggered = True
                        interrupt_event.set()
                        break

                    if queue_item is None:
                        logger.info("Consumer: Received end signal from producer.")
                        break

                    order_index, image, current_payload, current_target_base_name = queue_item

                    if order_index != i:
                        logger.error(f"Consumer: Order mismatch! Expected index {i}, got {order_index}. Stopping.")
                        interrupt_triggered = True
                        interrupt_event.set()
                        image_queue.task_done()
                        break

                    if image is None:
                        logger.warning(
                            f"Consumer: Received failure signal for image {i} ('{current_target_base_name}'). Skipping scoring.")
                        image_queue.task_done()
                        continue

                    # --- Score the received image ---
                    logger.info(
                        f"Consumer: Scoring image {i + 1}/{total_expected_images} ('{current_target_base_name}')...")
                    score_start_time = time.time()
                    individual_score = 0.0
                    processed_item = False
                    try:
                        # --- FIX: Safely get the prompt, defaulting to "" if not in payload ---
                        prompt_for_scorer = current_payload.get("prompt", "")
                        individual_score = await self.scorer.score(image, prompt_for_scorer,
                                                                   name=current_target_base_name)

                        if individual_score == -1.0:
                            logger.warning(f"Consumer: OVERRIDE_SCORE detected during scoring of image {i}.")
                            interrupt_event.set()
                            fake_score_value = self.scorer.handle_override_prompt()
                            interrupt_triggered = True
                            break

                        # --- Normal Score Processing ---
                        score_duration = time.time() - score_start_time
                        logger.debug(f"Scoring index {i} took {score_duration:.2f}s.")

                        weight = current_payload.get("score_weight", 1.0)
                        scores.append(individual_score)
                        norm_weights.append(weight)
                        print(
                            f"  Image {i + 1}/{total_expected_images} scored: {individual_score:.4f} (Weight: {weight})")

                        if self.cfg.save_imgs:
                            effective_iteration = self.iteration + self.completed_trials
                            self.save_img(image, current_target_base_name, individual_score, effective_iteration, i,
                                          current_payload)

                        images_processed += 1
                        processed_item = True

                    except Exception as e_score:
                        logger.error(
                            f"Consumer: Error scoring image '{current_target_base_name}' (index {i}): {e_score}",
                            exc_info=True)
                        processed_item = True
                    finally:
                        if processed_item:
                            image_queue.task_done()

                # End of consumer loop

            except asyncio.CancelledError:
                logger.warning("Main task cancelled.")
                if interrupt_event:
                    interrupt_event.set()
            except Exception as e_main:
                logger.error(f"Error during concurrent generation/scoring: {e_main}", exc_info=True)
                if interrupt_event:
                    interrupt_event.set()
            finally:
                # --- Cleanup ---
                if producer_task and not producer_task.done():
                    logger.info("Cancelling producer task...")
                    producer_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await producer_task
                    logger.info("Producer task cancelled.")
            
            # --- INDENTATION END --- (The 'async with session' closes here)

        gen_score_duration = time.time() - start_gen_score_time
        logger.info(f"Generation & scoring phase took {gen_score_duration:.2f} seconds.")

        # --- Calculate Final Score ---
        if interrupt_triggered:
            logger.info(f"Iteration interrupted by override. Using final score: {fake_score_value:.4f}")
            avg_score = fake_score_value
        elif not scores:
            logger.warning("No images were successfully scored.")
            raise RuntimeError("Generation failed: No images were produced or scored.")
        else:
            try:
                avg_score = self.scorer.average_calc(scores, norm_weights, self.cfg.img_average_type)
                logger.info(f"Calculated average score: {avg_score:.4f}")
            except Exception as e_avg:
                logger.error(f"Error calculating average score: {e_avg}", exc_info=True)
                raise RuntimeError(f"Score calculation error: {e_avg}")

        # --- Update Best Score & Logging ---
        self.update_best_score(params, avg_score)
        
        # --- Unload any lazy-loaded models ---
        self.scorer.unload_lazy_models()


        # --- Collect Data for Artist ---
        #       self.artist.collect_data(avg_score, params)

        iteration_duration = time.time() - iteration_start_time
        logger.info(
            f"Iteration {self.iteration} finished. Final Score for Optimizer: {avg_score:.4f}. Duration: {iteration_duration:.2f}s")

        return avg_score

    # --- save_img, image_path, update_best_score remain the same ---
    def save_img(
            self,
            image: Image.Image,
            name: str,  # This is the original payload name base (e.g., 'noob6')
            score: float,
            it: int,
            img_order_index: int,  # <<< Use the order index here
            payload: Dict,
    ) -> Optional[Path]:
        # Use batch_n as the image index within the iteration
        img_path = self.image_path(name, score, it, img_order_index)

        pnginfo = PngImagePlugin.PngInfo()
        for k, v in payload.items():
            try:
                pnginfo.add_text(str(k), str(v))
            except Exception as e_png:
                logger.warning(f"Could not add key '{k}' to PNG info: {e_png}")
        try:
            img_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(img_path, pnginfo=pnginfo)
        except (OSError, IOError) as e:
            logger.error(f"Error saving image to {img_path}: {e}"); return None
        return img_path

    def image_path(self, name: str, score: float, it: int, img_order_index: int) -> Path:  # <<< Use order index
        base_dir = Path(HydraConfig.get().runtime.output_dir)
        imgs_sub_dir = base_dir / "imgs"
        # Use img_order_index as the sequence number within the iteration
        return imgs_sub_dir / f"{it:03}-{img_order_index:02}-{name}-{score:4.3f}.png"

    def update_best_score(self, params: Dict, avg_score: float):
        logger.info(f"{'-' * 10}\nRun score: {avg_score}")
        # Format parameters for logging nicely
        param_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items())
        logger.info(f"Parameters: {{{param_str}}}")  # Use curly braces for dict-like look

        if avg_score > self.best_rolling_score:
            logger.info("\n NEW BEST!")
            self.best_rolling_score = avg_score

            # The current model's path is already set in self.merger.output_file
            current_model_path = self.merger.output_file

            # Instead of calling a naming function, we just derive the "best" name
            # from the current file's name. It's simple and has no dependencies!
            if current_model_path:
                new_best_path = current_model_path.with_name(
                    current_model_path.stem + "_best" + current_model_path.suffix
                )
            else:
                logger.error("Cannot determine new best path because output_file is not set.")
                return

            # Check if a different previous best model exists and delete it
            if self.merger.best_output_file and self.merger.best_output_file.exists() and self.merger.best_output_file != new_best_path:
                try:
                    os.remove(self.merger.best_output_file)
                    logger.info(f"Deleted previous best model: {self.merger.best_output_file}")
                except OSError as e_del:
                    logger.error(f"Error deleting previous best model: {e_del}")

            # Update the best model path in the merger
            self.merger.best_output_file = new_best_path

            # Move the current model to the new "best" path
            try:
                if self.merger.output_file and self.merger.output_file.exists():
                    shutil.move(self.merger.output_file, self.merger.best_output_file)
                    logger.info(f"Saved new best model as: {self.merger.best_output_file}")
                else:
                    logger.warning(f"Output file {self.merger.output_file} does not exist, cannot save as best.")
            except OSError as e_mov:
                logger.error(f"Error moving {self.merger.output_file} to {self.merger.best_output_file}: {e_mov}")

            # Static method call is correct
            effective_iteration = self.iteration + self.completed_trials
            Optimizer.save_best_log(params, effective_iteration)
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
        except ValueError:  # Hydra not initialized
            logger.error("Hydra context not available, cannot save best.log.")
        except Exception as e:
            logger.error(f"Failed to save best.log: {e}")
