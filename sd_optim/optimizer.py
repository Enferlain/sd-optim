# optimizer.py - Version 1.1 (Concurrent Gen/Score)
import gc
import os
import shutil
import logging
import aiohttp
import asyncio  # <<< Import asyncio
import time  # <<< Import time for logging durations
import torch
import hashlib
import json

from contextlib import suppress
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from PIL import Image, PngImagePlugin
from hydra import utils as hydra_utils

import sd_mecha

from sd_mecha.recipe_nodes import ModelRecipeNode
from sd_optim import utils
from sd_optim.bounds import ParameterHandler, BoundsInfo
from sd_optim.generator import Generator
from sd_optim.merger import Merger
from sd_optim.prompter import Prompter
from sd_optim.scorer import AestheticScorer
from sd_optim.trial_scorer_summary import build_trial_scorer_summary

logger = logging.getLogger(__name__)

# Suppress PIL's verbose debug logging when scanning PNG metadata
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)

PathT = os.PathLike


def _compute_scorer_setup_fingerprint(cfg: DictConfig) -> str:
    """
    Fingerprint the scoring objective so cached `final_score` is only reused when
    scorer configuration is effectively identical.
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        cfg_dict = {}

    scorer_method_raw = cfg_dict.get("scorer_method", []) or []
    scorer_method = [str(s).lower() for s in scorer_method_raw]

    scorer_weight_raw = cfg_dict.get("scorer_weight", {}) or {}
    if not isinstance(scorer_weight_raw, dict):
        scorer_weight_raw = {}
    scorer_weight = {name: scorer_weight_raw.get(name, scorer_weight_raw.get(name.lower(), 1.0)) for name in scorer_method}

    scorer_filters_raw = cfg_dict.get("scorer_filters", {}) or {}
    if not isinstance(scorer_filters_raw, dict):
        scorer_filters_raw = {}
    scorer_filters = {name: scorer_filters_raw.get(name, scorer_filters_raw.get(name.lower(), {})) for name in scorer_method}

    per_scorer_cfg: dict[str, dict[str, Any]] = {}
    for name in scorer_method:
        prefix = f"{name}_"
        per_scorer_cfg[name] = {k: v for k, v in cfg_dict.items() if isinstance(k, str) and k.lower().startswith(prefix)}

    fingerprint_input = {
        "v": 1,
        "scorer_method": scorer_method,
        "scorer_average_type": cfg_dict.get("scorer_average_type"),
        "scorer_weight": scorer_weight,
        "scorer_filters": scorer_filters,
        "per_scorer_cfg": per_scorer_cfg,
        "scorer_model_dir": cfg_dict.get("scorer_model_dir"),
    }
    recipe_json = json.dumps(fingerprint_input, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(recipe_json.encode("utf-8")).hexdigest()


@dataclass
class Optimizer:
    cfg: DictConfig
    best_rolling_score: float = 0.0
    param_info: BoundsInfo = field(default_factory=dict, init=False)
    optimizer_pbounds: dict[str, tuple[float, float] | float | int | list] = field(default_factory=dict, init=False)
    optimization_start_time: float | None = None  # Add start time tracker
    completed_trials: int = 0  # To track trials from resumed studies
    scorer_setup_fp: str = field(default="", init=False)

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
                raise FileNotFoundError(f"Representative model not found in models_dir or as an absolute path: {representative_model_path}")

        logger.info(f"Inferring base ModelConfig from: {representative_model_path}")
        rep_model_node = sd_mecha.model(str(representative_model_path))

        # We assert that the node is the specific type we need. This makes the linter happy and the code safer!
        assert isinstance(rep_model_node, ModelRecipeNode), "The representative model must be a file path, not a literal dict."

        inferred_candidates = utils.get_model_config_candidates(rep_model_node, [models_dir])
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
        self.generator = Generator(self.cfg.url, self.cfg.batch_size, self.cfg.webui)
        self.scorer = AestheticScorer(self.cfg)
        self.scorer_setup_fp = _compute_scorer_setup_fingerprint(self.cfg)
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
        self._load_history_cache()

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
        logger.info(f"Prepared {len(self.optimizer_pbounds)} parameters for the optimizer with specific bounds.")

    # =========================================================================
    # UNIVERSAL IMAGE REUSE METHODS
    # =========================================================================

    def _load_history_cache(self):
        """
        Scans logs/ for run_manifest.json files and builds an in-memory lookup.
        Also checks PNG metadata for legacy runs without manifests.
        """
        import time as _time  # Local import to avoid shadowing

        try:
            project_root = Path(hydra_utils.get_original_cwd())
        except ValueError:
            # Hydra not initialized yet (e.g., during testing)
            project_root = Path.cwd()

        logs_dir = project_root / "logs"
        if not logs_dir.exists():
            logger.debug("No logs directory found, skipping history cache load.")
            return

        start_time = _time.time()
        logger.info("Universal Reuse: Scanning logs for cached results...")

        manifest_hits = 0
        png_hits = 0
        legacy_dirs = []

        # Phase 1: High-fidelity manifests (preferred) - sorted newest first
        manifest_paths = sorted(
            logs_dir.rglob("run_manifest.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,  # Newest first
        )
        for manifest_path in manifest_paths:
            try:
                run_dir = manifest_path.parent
                with open(manifest_path, encoding="utf-8") as f:
                    manifest_data = json.load(f)
                for img_hash, data in manifest_data.items():
                    if img_hash in self.history_cache:
                        continue  # Skip duplicates (newer already found)
                    rel_path = data.get("path")
                    if rel_path:
                        abs_path = run_dir / rel_path
                        if abs_path.exists():
                            data["full_path"] = abs_path
                            self.history_cache[img_hash] = data
                            manifest_hits += 1
            except Exception as e:
                logger.debug(f"Could not load manifest {manifest_path}: {e}")

        # Phase 2: Legacy PNG scan - DISABLED by default (old PNGs don't have our hash)
        # Enable via config if needed: reuse_scan_legacy_pngs: true
        if self.cfg.get("reuse_scan_legacy_pngs", False):
            legacy_dirs = []
            for img_dir in logs_dir.rglob("imgs"):
                run_dir = img_dir.parent
                if not (run_dir / "run_manifest.json").exists():
                    legacy_dirs.append(img_dir)

            if legacy_dirs:
                legacy_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                logger.info(f"  Scanning {len(legacy_dirs)} legacy run(s) for PNG metadata...")
                for img_dir in legacy_dirs:
                    for png_path in img_dir.glob("*.png"):
                        try:
                            with Image.open(png_path) as img:
                                meta_hash = img.info.get("sd_optim_hash")
                                if meta_hash and meta_hash not in self.history_cache:
                                    self.history_cache[meta_hash] = {
                                        "full_path": png_path,
                                        "scores": json.loads(img.info.get("sd_optim_scores", "{}")),
                                        "final_score": float(img.info.get("sd_optim_final_score", 0)),
                                    }
                                    png_hits += 1
                        except Exception:
                            continue

        elapsed = _time.time() - start_time
        total = manifest_hits + png_hits
        if total > 0:
            logger.info(f"Universal Reuse: Loaded {total} cached results in {elapsed:.2f}s")
        else:
            logger.info(f"Universal Reuse: No cached results found ({elapsed:.2f}s)")

    def _save_run_manifest(self):
        """Writes current run's manifest to the Hydra output directory."""
        if not self.current_run_manifest:
            return

        try:
            output_dir = Path(HydraConfig.get().runtime.output_dir)
            manifest_path = output_dir / "run_manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(self.current_run_manifest, f, indent=2, sort_keys=True)
            logger.debug(f"Saved run manifest with {len(self.current_run_manifest)} entries to {manifest_path}")
        except Exception as e:
            logger.warning(f"Could not save run manifest: {e}")

    @staticmethod
    def calculate_image_hash(params: dict, payload: dict) -> str:
        """
        Creates a deterministic SHA256 hash from generation recipe.
        This fingerprint uniquely identifies an image based on:
        - Merge/optimization parameters
        - Key generation settings (prompt, seed, dimensions, etc.)
        """
        # Keys that affect the generated image
        gen_keys = [
            "prompt",
            "negative_prompt",
            "seed",
            "steps",
            "cfg",
            "sampler_name",
            "scheduler",
            "width",
            "height",
            "workflow_json",  # For ComfyUI
        ]

        # Build stable representation
        stable_params = {k: params[k] for k in sorted(params.keys())}
        stable_payload = {k: payload.get(k) for k in gen_keys if k in payload}

        recipe = {"params": stable_params, "payload": stable_payload}
        recipe_json = json.dumps(recipe, sort_keys=True, default=str)
        return hashlib.sha256(recipe_json.encode("utf-8")).hexdigest()

    # --- ADDED: Sequential Producer Coroutine ---
    async def _sequential_producer(
        self,
        payloads: list[dict],
        target_paths: list[str],
        queue: asyncio.Queue,
        session: aiohttp.ClientSession,
        interrupt_event: asyncio.Event,  # Shared event for interruption
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
                    if generated_image is not None:
                        # We only expect one image per generation in this context
                        continue
                    logger.debug(f"Producer: Received image {i} ('{current_target_base_name}'). Putting onto queue.")
                    await queue.put((i, image, current_payload, current_target_base_name))
                    generated_image = image
                    # --- REMOVED: break ---
                    # Removing this break prevents 'RuntimeError: async generator ignored GeneratorExit'
                    # by allowing the generator to finish naturally.
                if generated_image is None:
                    logger.warning(f"Producer: Generation task {i} ('{current_target_base_name}') yielded no images.")
                    await queue.put((i, None, current_payload, current_target_base_name))

            except asyncio.CancelledError:
                logger.info(f"Producer: Generation task {i} cancelled.")
                break
            except Exception as e_gen_task:
                logger.error(
                    f"Producer: Error during generation task {i} ('{current_target_base_name}'): {e_gen_task}",
                    exc_info=True,
                )
                await queue.put((i, None, current_payload, current_target_base_name))
            # --- End removal of semaphore block ---

            # Extra check after generation i completes
            if interrupt_event.is_set():
                logger.warning(f"Producer: Interrupt detected after finishing generation {i}. Stopping.")
                break

        logger.info("Sequential Producer finished.")
        # Optionally signal completion: await queue.put(None)

    async def sd_target_function(self, params: dict[str, Any]) -> float | None:
        self.iteration += 1
        self.last_trial_scorer_summary = {"aggregate": {}, "payloads": []}
        # Adjust iteration number for resumed runs ---
        effective_iteration = self.iteration + self.completed_trials
        iteration_start_time = time.time()

        iteration_type = "warmup" if effective_iteration <= self.cfg.optimizer.init_points else "optimization"
        if effective_iteration in {1, self.cfg.optimizer.init_points + 1}:
            logger.info(f"\n{'-' * 10} Starting {iteration_type} Phase {'-' * 10}>")

        logger.info(f"\n--- {iteration_type} - Iteration: {effective_iteration} ---")
        logger.info(f"Optimizer proposed parameters: {params}")

        # =========================================================================
        # EARLY CACHE CHECK - 3 tiers:
        #   1. FULL HIT:    hash found + scorers match → return cached score
        #   2. PARTIAL HIT: hash found + image exists but scorers differ → re-score only
        #   3. FULL MISS:   hash not found → merge + generate + score
        # =========================================================================
        payloads, target_paths = self.prompter.render_payloads(self.cfg.batch_size)
        if not payloads:
            logger.error("Prompter generated no payloads.")
            raise RuntimeError("Prompter failed to generate any payloads.")

        # Determine current scorer set for validation
        current_scorers = set(s.lower() for s in self.cfg.scorer_method)
        scorer_setup_fp = self.scorer_setup_fp

        # Calculate hashes and check cache for all payloads
        cache_results = []  # list of (hash, cached_data, tier) tuples
        overall_tier = "full_hit"  # optimistic, downgrade as needed

        for i, payload in enumerate(payloads):
            img_hash = self.calculate_image_hash(params, payload)
            cached = self.history_cache.get(img_hash)

            if cached and cached.get("final_score") is not None:
                # Hash found — check if scorers match
                cached_scorers = set(cached.get("scores", {}).keys()) - {"combined"}
                cached_fp = cached.get("scorer_setup_fp")
                if (
                    cached_scorers
                    and cached_scorers == current_scorers
                    and isinstance(cached_fp, str)
                    and cached_fp == scorer_setup_fp
                ):
                    cache_results.append((img_hash, cached, payload, "full_hit"))
                elif cached.get("full_path") and Path(cached["full_path"]).exists():
                    # Image exists but scorer data doesn't match — need re-scoring
                    cache_results.append((img_hash, cached, payload, "partial_hit"))
                    overall_tier = "partial_hit"
                else:
                    overall_tier = "full_miss"
                    break
            elif cached and cached.get("full_path") and Path(cached["full_path"]).exists():
                # Hash found, no score, but image exists — re-score
                cache_results.append((img_hash, cached, payload, "partial_hit"))
                overall_tier = "partial_hit"
            else:
                overall_tier = "full_miss"
                break

        # --- Tier 1: FULL HIT — all images cached with matching scorers ---
        if overall_tier == "full_hit" and cache_results:
            cached_scores = [c[1]["final_score"] for c in cache_results]
            cached_weights = [c[2].get("score_weight", 1.0) for c in cache_results]
            avg_score = self.scorer.average_calc(cached_scores, cached_weights, self.cfg.img_average_type)
            payload_entries: list[dict[str, Any]] = []
            for idx, (_, cached, payload, _) in enumerate(cache_results):
                payload_name = target_paths[idx] if idx < len(target_paths) else f"payload_{idx}"
                payload_entries.append(
                    {
                        "name": payload_name,
                        "weight": payload.get("score_weight", 1.0),
                        "scores": dict(cached.get("scores", {})),
                        "combined": cached.get("final_score"),
                    }
                )
            self.last_trial_scorer_summary = build_trial_scorer_summary(
                payload_entries,
                avg_score,
                lambda values, weights: self.scorer.average_calc(values, weights, self.cfg.img_average_type),
            )
            elapsed = time.time() - iteration_start_time
            logger.info(f"CACHE HIT: All {len(cached_scores)} images reused. Score: {avg_score:.4f} ({elapsed:.2f}s)")
            return avg_score

        # --- Tier 2: PARTIAL HIT — images exist but need re-scoring ---
        if overall_tier == "partial_hit" and cache_results:
            logger.info(
                f"PARTIAL CACHE HIT: {len(cache_results)} images found, "
                f"re-scoring with current scorers ({', '.join(sorted(current_scorers))})"
            )
            rescored_scores = []
            rescored_weights = []
            payload_entries: list[dict[str, Any]] = []
            for idx, (img_hash, cached, payload, tier) in enumerate(cache_results):
                payload_name = target_paths[idx] if idx < len(target_paths) else f"payload_{idx}"
                if tier == "full_hit":
                    # This one already has matching scores
                    rescored_scores.append(cached["final_score"])
                    rescored_weights.append(payload.get("score_weight", 1.0))
                    payload_entries.append(
                        {
                            "name": payload_name,
                            "weight": payload.get("score_weight", 1.0),
                            "scores": dict(cached.get("scores", {})),
                            "combined": cached.get("final_score"),
                        }
                    )
                    continue

                # Load image from disk and re-score
                try:
                    image = Image.open(cached["full_path"])
                    prompt_for_scorer = payload.get("prompt", "")
                    individual_score = await self.scorer.score(
                        image,
                        prompt_for_scorer,
                        name=payload_name,
                    )
                    rescored_scores.append(individual_score)
                    rescored_weights.append(payload.get("score_weight", 1.0))

                    # Update cache entry with new scorer results
                    scorer_results = dict(self.scorer.last_scorer_results)
                    scorer_results["combined"] = individual_score
                    cached["scores"] = scorer_results
                    cached["final_score"] = individual_score
                    cached["scorer_setup_fp"] = scorer_setup_fp

                    # Update manifest for this run
                    try:
                        output_dir = Path(HydraConfig.get().runtime.output_dir)
                        self.current_run_manifest[img_hash] = {
                            "path": str(Path(cached["full_path"]).relative_to(output_dir)),
                            "scores": scorer_results,
                            "final_score": individual_score,
                            "scorer_setup_fp": scorer_setup_fp,
                        }
                    except Exception:
                        # Image from different run dir — store absolute
                        self.current_run_manifest[img_hash] = {
                            "path": str(cached["full_path"]),
                            "scores": scorer_results,
                            "final_score": individual_score,
                            "scorer_setup_fp": scorer_setup_fp,
                        }

                    payload_entries.append(
                        {
                            "name": payload_name,
                            "weight": payload.get("score_weight", 1.0),
                            "scores": scorer_results,
                            "combined": individual_score,
                        }
                    )
                    logger.info(f"  Re-scored: {individual_score:.4f}")
                    image.close()
                except Exception as e:
                    logger.warning(f"Failed to re-score cached image: {e}")
                    overall_tier = "full_miss"
                    break

            if overall_tier == "partial_hit" and rescored_scores:
                avg_score = self.scorer.average_calc(rescored_scores, rescored_weights, self.cfg.img_average_type)
                self.last_trial_scorer_summary = build_trial_scorer_summary(
                    payload_entries,
                    avg_score,
                    lambda values, weights: self.scorer.average_calc(values, weights, self.cfg.img_average_type),
                )
                elapsed = time.time() - iteration_start_time
                logger.info(f"PARTIAL HIT complete: Score: {avg_score:.4f} (re-scored in {elapsed:.2f}s, skipped merge+gen)")
                self._save_run_manifest()
                return avg_score

        # --- Tier 3: FULL MISS — proceed with merge + generation + scoring ---

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

            model_path: Path | None = None

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
                        iteration=effective_iteration,
                    )
                elif self.cfg.optimization_mode == "layer_adjust":
                    self.merger.output_file = self.merger.create_model_output_name(iteration=effective_iteration)
                    model_path = self.merger.layer_adjust(params, self.cfg)

                elif self.cfg.optimization_mode == "recipe":
                    model_path = self.merger.recipe_optimization(
                        params=params,
                        param_info=self.param_info,
                        cache=self.cache,
                        iteration=effective_iteration,
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
            payload_entries = []
            # NOTE: payloads and target_paths already rendered in early cache check section

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
                        session,  # <<< PASS THE OUTER SESSION
                        interrupt_event,
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
                        effective_timeout = total_timeout_seconds if total_timeout_seconds else 3600  # Default 1hr
                        queue_item = await asyncio.wait_for(image_queue.get(), timeout=effective_timeout)
                    except TimeoutError:
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
                        logger.warning(f"Consumer: Received failure signal for image {i} ('{current_target_base_name}'). Skipping scoring.")
                        image_queue.task_done()
                        continue

                    # --- Score the received image ---
                    logger.info(f"Consumer: Scoring image {i + 1}/{total_expected_images} ('{current_target_base_name}')...")
                    score_start_time = time.time()
                    individual_score = 0.0
                    processed_item = False
                    try:
                        # --- FIX: Safely get the prompt, defaulting to "" if not in payload ---
                        prompt_for_scorer = current_payload.get("prompt", "")
                        individual_score = await self.scorer.score(image, prompt_for_scorer, name=current_target_base_name)

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
                        scorer_results = dict(self.scorer.last_scorer_results)
                        scores.append(individual_score)
                        norm_weights.append(weight)
                        payload_entries.append(
                            {
                                "name": current_target_base_name,
                                "weight": weight,
                                "scores": scorer_results,
                                "combined": individual_score,
                            }
                        )
                        logger.info(
                            "Image %s/%s scored: %.4f (Weight: %.3f)",
                            i + 1,
                            total_expected_images,
                            individual_score,
                            weight,
                        )

                        if self.cfg.save_imgs:
                            effective_iteration = self.iteration + self.completed_trials
                            scorer_results["combined"] = individual_score
                            self.save_img(
                                image,
                                current_target_base_name,
                                individual_score,
                                effective_iteration,
                                i,
                                current_payload,
                                params=params,
                                scorer_results=scorer_results,
                            )

                        images_processed += 1
                        processed_item = True

                    except Exception as e_score:
                        logger.error(
                            f"Consumer: Error scoring image '{current_target_base_name}' (index {i}): {e_score}",
                            exc_info=True,
                        )
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
                logger.error(
                    f"Error during concurrent generation/scoring: {e_main}",
                    exc_info=True,
                )
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

        self.last_trial_scorer_summary = build_trial_scorer_summary(
            payload_entries,
            avg_score,
            lambda values, weights: self.scorer.average_calc(values, weights, self.cfg.img_average_type),
        )

        # --- Update Best Score & Logging ---
        self.update_best_score(params, avg_score)

        # --- Unload any lazy-loaded models ---
        self.scorer.unload_lazy_models()

        # --- Collect Data for Artist ---
        #       self.artist.collect_data(avg_score, params)

        iteration_duration = time.time() - iteration_start_time
        logger.info(f"Iteration {self.iteration} finished. Final Score for Optimizer: {avg_score:.4f}. Duration: {iteration_duration:.2f}s")

        # --- Save run manifest (for future reuse) ---
        self._save_run_manifest()

        return avg_score

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
        img_path = self.image_path(name, score, it, img_order_index)

        pnginfo = PngImagePlugin.PngInfo()

        # --- PHASE 1: Preserve existing metadata (ComfyUI workflow, Swarm data, etc.) ---
        if hasattr(image, "info") and image.info:
            for k, v in image.info.items():
                if isinstance(v, str):
                    try:
                        pnginfo.add_text(str(k), v)
                    except Exception:
                        pass  # Skip non-serializable keys

        # --- PHASE 2: Add payload generation parameters ---
        for k, v in payload.items():
            try:
                pnginfo.add_text(str(k), str(v))
            except Exception as e_png:
                logger.warning(f"Could not add key '{k}' to PNG info: {e_png}")

        # --- PHASE 3: Inject reuse/identity metadata ---
        if params is not None:
            img_hash = self.calculate_image_hash(params, payload)
            pnginfo.add_text("sd_optim_hash", img_hash)

            # Store individual scorer results if available
            if scorer_results:
                pnginfo.add_text("sd_optim_scores", json.dumps(scorer_results))
            pnginfo.add_text("sd_optim_final_score", str(score))

            # Update manifest for this run
            try:
                output_dir = Path(HydraConfig.get().runtime.output_dir)
                self.current_run_manifest[img_hash] = {
                    "path": str(img_path.relative_to(output_dir)),
                    "scores": scorer_results or {"combined": score},
                    "final_score": score,
                    "scorer_setup_fp": self.scorer_setup_fp,
                }
            except Exception:
                pass  # Non-critical, continue saving

        # --- PHASE 4: Save the image ---
        try:
            img_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(img_path, pnginfo=pnginfo)
        except OSError as e:
            logger.error(f"Error saving image to {img_path}: {e}")
            return None
        return img_path

    def image_path(self, name: str, score: float, it: int, img_order_index: int) -> Path:  # <<< Use order index
        base_dir = Path(HydraConfig.get().runtime.output_dir)
        imgs_sub_dir = base_dir / "imgs"
        # Use img_order_index as the sequence number within the iteration
        return imgs_sub_dir / f"{it:03}-{img_order_index:02}-{name}-{score:4.3f}.png"

    def update_best_score(self, params: dict, avg_score: float):
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
                new_best_path = current_model_path.with_name(current_model_path.stem + "_best" + current_model_path.suffix)
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
    def get_best_parameters(self) -> dict:
        raise NotImplementedError()

    @abstractmethod
    def get_optimization_history(self) -> list[dict]:
        raise NotImplementedError()

    @staticmethod
    def save_best_log(params: dict, iteration: int) -> None:
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
