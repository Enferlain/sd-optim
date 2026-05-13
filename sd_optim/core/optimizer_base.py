# optimizer.py - Version 1.1 (Concurrent Gen/Score)
import os
import logging
import time  # <<< Import time for logging durations
import json

from contextlib import suppress
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from PIL import Image, PngImagePlugin
from hydra import utils as hydra_utils

import sd_mecha

from sd_optim.core.optimizer_artifacts import (
    build_best_output_path,
    remove_model_file,
    replace_best_model,
    save_best_log as persist_best_log,
)
from sd_optim.core.optimizer_cache import (
    calculate_image_hash,
    compute_generation_setup_fingerprint,
    compute_scorer_setup_fingerprint,
    reuse_cached_results_enabled,
)
from sd_optim.core.optimizer_cache_io import (
    build_image_output_path,
    build_run_manifest_entry,
    load_history_cache,
)
from sd_optim.core.optimizer_runtime import run_trial_iteration, sequential_producer
from sd_mecha.recipe_nodes import ModelRecipeNode
from sd_optim.bounds import ParameterHandler, BoundsInfo
from sd_optim.generator import Generator
from sd_optim.merger import Merger
from sd_optim.prompter import Prompter
from sd_optim.scorer import Scorer
from sd_optim.utils.config import validate_run_config
from sd_optim.utils.recipes import get_model_config_candidates

logger = logging.getLogger(__name__)

# Suppress PIL's verbose debug logging when scanning PNG metadata
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)

PathT = os.PathLike


@dataclass
class Optimizer(ABC):
    cfg: DictConfig
    best_rolling_score: float = 0.0
    param_info: BoundsInfo = field(default_factory=dict, init=False)
    optimizer_pbounds: dict[str, tuple[float, float] | float | int | list] = field(default_factory=dict, init=False)
    optimization_start_time: float | None = None  # Add start time tracker
    completed_trials: int = 0  # To track trials from resumed studies
    scorer_setup_fp: str = field(default="", init=False)
    generation_setup_fp: str = field(default="", init=False)

    def __post_init__(self) -> None:
        # --- STAGE 1: VALIDATE THE ENTIRE CONFIG FIRST ---
        # This is now a clean, clear gatekeeper step.
        validate_run_config(self.cfg)

        # --- STAGE 2: CENTRALIZED CONFIG LOADING ---
        logger.info("Optimizer starting up: Performing centralized config loading...")

        # Now that validation is passed, we can safely get the models_dir.
        # This logic is now explicit and clear right where it's needed.
        models_dir = Path(self.cfg.paths.models_dir).resolve()
        logger.info(f"Using primary models directory from config: {models_dir}")

        # The rest of the __post_init__ is exactly the same as before.
        # It just uses the 'models_dir' variable we defined right here.
        if not self.cfg.merge.model_paths:
            raise ValueError("'model_paths' cannot be empty.")

        representative_model_name = self.cfg.merge.model_paths[0]
        representative_model_path = models_dir / representative_model_name
        if not representative_model_path.exists():
            # Fallback to check if an absolute path was given in the list
            absolute_path_check = Path(representative_model_name)
            if absolute_path_check.is_absolute() and absolute_path_check.exists():
                representative_model_path = absolute_path_check
            else:
                raise FileNotFoundError(f"Representative model not found in models_dir or as an absolute path: {representative_model_path}")

        logger.info(f"Inferring base ModelConfig from: {representative_model_path}")
        rep_model_node = sd_mecha.model(str(representative_model_path))

        # We assert that the node is the specific type we need. This makes the linter happy and the code safer!
        assert isinstance(rep_model_node, ModelRecipeNode), "The representative model must be a file path, not a literal dict."

        inferred_candidates = get_model_config_candidates(rep_model_node, [models_dir])
        if not inferred_candidates:
            raise ValueError(f"Could not infer a ModelConfig for {representative_model_path}.")
        base_model_config = inferred_candidates[0]
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
            models_dir=models_dir,
        )

        # 2b. Initialize ParameterHandler with the configs
        self.bounds_initializer = ParameterHandler(
            cfg=self.cfg,
            base_model_config=base_model_config,
            custom_block_config=custom_block_config,
        )

        # --- STAGE 3: COMPLETE THE REST OF THE SETUP ---
        self.setup_parameter_space()
        self.generator = Generator(self.cfg.url, self.cfg.generation.batch_size, self.cfg.webui)
        self.scorer = Scorer(self.cfg)
        self.scorer_setup_fp = compute_scorer_setup_fingerprint(self.cfg)
        self.generation_setup_fp = compute_generation_setup_fingerprint(self.cfg)
        self.prompter = Prompter(self.cfg)
        self.iteration = -1
        self.best_model_path = None
        self.cache = {}
        self.last_trial_scorer_summary: dict[str, Any] = {"aggregate": {}, "payloads": []}

        # --- REUSE CACHE SETUP ---
        # Maps image hash -> {full_path, scores, final_score}
        self.history_cache: dict[str, dict] = {}
        # Maps image hash -> {path (relative), scores, final_score} for current run
        self.current_run_manifest: dict[str, dict] = {}
        try:
            project_root = Path(hydra_utils.get_original_cwd())
        except ValueError:
            project_root = Path.cwd()

        logs_dir = project_root / "logs"
        if not logs_dir.exists():
            logger.debug("No logs directory found, skipping history cache load.")
        elif not reuse_cached_results_enabled(self.cfg):
            logger.info("Universal Reuse: Disabled by config; skipping cached history scan.")
        else:
            start_time = time.time()
            logger.info("Universal Reuse: Scanning logs for cached results...")
            cache_load_result = load_history_cache(
                logs_dir,
                scan_legacy_pngs=bool(self.cfg.reuse_scan_legacy_pngs),
            )
            self.history_cache = cache_load_result.entries
            elapsed = time.time() - start_time
            total_hits = cache_load_result.manifest_hits + cache_load_result.png_hits
            if total_hits > 0:
                logger.info("Universal Reuse: Loaded %s cached results in %.2fs", total_hits, elapsed)
            else:
                logger.info("Universal Reuse: No cached results found (%.2fs)", elapsed)

    #        from sd_optim.artist import Artist
    #        self.artist = Artist(self)

    def setup_parameter_space(self):
        """Generates parameter info and extracts bounds for the optimizer."""
        logger.info("Setting up optimization parameter space...")
        self.param_info, self.optimizer_pbounds = self.bounds_initializer.get_bounds(self.cfg.optimization_guide.get("custom_bounds"))
        self.optimizer_pbounds = {}
        for param_name, info in self.param_info.items():
            bounds_value = info.get("bounds")
            if bounds_value is None:
                logger.warning(f"Parameter '{param_name}' missing 'bounds' in info. Skipping for optimizer.")
                continue
            self.optimizer_pbounds[param_name] = bounds_value

        # Optional: Check if optimizer_pbounds is empty and raise error
        if not self.optimizer_pbounds:
            logger.error("No optimization bounds were generated for the optimizer. Check optimization_guide.yaml and merge method.")
            # Decide if this should be fatal or just a warning depending on the optimizer
            raise ValueError("Optimization parameter space for the optimizer is empty.")
        logger.debug("Prepared %s parameters for the optimizer with specific bounds.", len(self.optimizer_pbounds))

    async def _sequential_producer(
        self,
        payloads: list[dict],
        target_paths: list[str],
        queue: Any,
        session: Any,
        interrupt_event: Any,
    ) -> None:
        await sequential_producer(self, payloads, target_paths, queue, session, interrupt_event)

    async def sd_target_function(self, params: dict[str, Any]) -> float | None:
        return await run_trial_iteration(self, params)

    # --- save_img, image_path, update_best_score remain the same ---
    def save_img(
        self,
        image: Image.Image,
        name: str,  # This is the original payload name base (e.g., 'noob6')
        score: float,
        it: int,
        img_order_index: int,
        payload: dict,
        params: dict | None = None,  # Required for hash calculation
        scorer_results: dict[str, float] | None = None,  # Individual scorer results
    ) -> Path | None:
        """
        Saves the image with comprehensive metadata:
        1. Preserves existing backend metadata (ComfyUI workflow, etc.)
        2. Adds payload generation parameters
        3. Injects sd_optim hash and scores for future reuse
        """
        output_dir = Path(HydraConfig.get().runtime.output_dir)
        img_path = build_image_output_path(
            output_dir=output_dir,
            name=name,
            score=score,
            iteration=it,
            img_order_index=img_order_index,
        )

        pnginfo = PngImagePlugin.PngInfo()

        # --- PHASE 1: Preserve existing metadata (ComfyUI workflow, Swarm data, etc.) ---
        if hasattr(image, "info") and image.info:
            for k, v in image.info.items():
                if isinstance(v, str):
                    with suppress(Exception):
                        pnginfo.add_text(str(k), v)

        # --- PHASE 2: Add payload generation parameters ---
        for k, v in payload.items():
            try:
                pnginfo.add_text(str(k), str(v))
            except Exception as e_png:
                logger.warning(f"Could not add key '{k}' to PNG info: {e_png}")

        # --- PHASE 3: Inject reuse/identity metadata ---
        if params is not None:
            img_hash = calculate_image_hash(params, payload, self.generation_setup_fp)
            pnginfo.add_text("sd_optim_hash", img_hash)

            # Store individual scorer results if available
            if scorer_results:
                pnginfo.add_text("sd_optim_scores", json.dumps(scorer_results))
            pnginfo.add_text("sd_optim_final_score", str(score))

            # Update manifest for this run
            with suppress(Exception):
                self.current_run_manifest[img_hash] = build_run_manifest_entry(
                    image_path=img_path,
                    output_dir=output_dir,
                    scorer_results=scorer_results or {"combined": score},
                    final_score=score,
                    scorer_setup_fp=self.scorer_setup_fp,
                )

        # --- PHASE 4: Save the image ---
        try:
            img_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(img_path, pnginfo=pnginfo)
        except OSError as e:
            logger.error(f"Error saving image to {img_path}: {e}")
            return None
        return img_path

    def update_best_score(self, params: dict, avg_score: float):
        logger.info(f"{'-' * 10}\nRun score: {avg_score}")
        # Format parameters for logging nicely
        param_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items())
        logger.info(f"Parameters: {{{param_str}}}")  # Use curly braces for dict-like look

        if avg_score > self.best_rolling_score:
            logger.info("\n NEW BEST!")
            self.best_rolling_score = avg_score

            current_model_path = self.merger.output_file
            if current_model_path:
                new_best_path = build_best_output_path(current_model_path)
            else:
                logger.error("Cannot determine new best path because output_file is not set.")
                return

            if self.merger.best_output_file and self.merger.best_output_file.exists() and self.merger.best_output_file != new_best_path:
                logger.info(f"Deleted previous best model: {self.merger.best_output_file}")

            try:
                if self.merger.output_file and self.merger.output_file.exists():
                    self.merger.best_output_file = replace_best_model(
                        current_model_path=self.merger.output_file,
                        previous_best_path=self.merger.best_output_file,
                    )
                    logger.info(f"Saved new best model as: {self.merger.best_output_file}")
                else:
                    logger.warning(f"Output file {self.merger.output_file} does not exist, cannot save as best.")
            except OSError as e_mov:
                logger.error(f"Error moving {self.merger.output_file} to {self.merger.best_output_file}: {e_mov}")

            effective_iteration = self.iteration + self.completed_trials
            type(self).save_best_log(params, effective_iteration)
        else:
            try:
                if remove_model_file(self.merger.output_file):
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
    def get_best_parameters(self) -> dict:
        raise NotImplementedError()

    @abstractmethod
    def get_optimization_history(self) -> list[dict]:
        raise NotImplementedError()

    @staticmethod
    def save_best_log(params: dict, iteration: int) -> None:
        logger.info("Saving best.log")
        try:
            output_dir = Path(HydraConfig.get().runtime.output_dir)
            persist_best_log(output_dir=output_dir, params=params, iteration=iteration)
        except ValueError:  # Hydra not initialized
            logger.error("Hydra context not available, cannot save best.log.")
        except Exception as e:
            logger.error(f"Failed to save best.log: {e}")
