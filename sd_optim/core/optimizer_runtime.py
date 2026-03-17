from __future__ import annotations

import asyncio
import gc
import logging
import time
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiohttp
import torch
from hydra.core.hydra_config import HydraConfig
from PIL import Image

from sd_optim.core.optimizer_cache import calculate_image_hash, fail_on_error_enabled
from sd_optim.core.optimizer_cache_io import build_run_manifest_entry, save_run_manifest
from sd_optim.core.trial_scorer_summary import build_trial_scorer_summary

if TYPE_CHECKING:
    from sd_optim.core.optimizer_base import Optimizer

logger = logging.getLogger(__name__)


async def sequential_producer(
    optimizer: Optimizer,
    payloads: list[dict[str, Any]],
    target_paths: list[str],
    queue: asyncio.Queue,
    session: aiohttp.ClientSession,
    interrupt_event: asyncio.Event,
) -> None:
    """Generate images sequentially and enqueue them for scoring."""
    logger.info("Sequential Producer started.")
    total_payloads = len(payloads)
    for i in range(total_payloads):
        if interrupt_event.is_set():
            logger.warning("Producer: Interrupt detected before starting generation %s. Stopping.", i)
            break

        current_payload = payloads[i]
        current_target_base_name = target_paths[i]
        generated_image = None

        logger.info("Producer: Requesting generation %s/%s ('%s')...", i + 1, total_payloads, current_target_base_name)
        try:
            img_gen = optimizer.generator.generate(current_payload, optimizer.cfg, session)
            async for image in img_gen:
                if generated_image is not None:
                    continue
                logger.debug(
                    "Producer: Received image %s ('%s'). Putting onto queue.",
                    i,
                    current_target_base_name,
                )
                await queue.put((i, image, current_payload, current_target_base_name))
                generated_image = image
            if generated_image is None:
                logger.warning(
                    "Producer: Generation task %s ('%s') yielded no images.",
                    i,
                    current_target_base_name,
                )
                await queue.put((i, None, current_payload, current_target_base_name))

        except asyncio.CancelledError:
            logger.info("Producer: Generation task %s cancelled.", i)
            break
        except Exception as error:
            logger.error(
                "Producer: Error during generation task %s ('%s'): %s",
                i,
                current_target_base_name,
                error,
                exc_info=True,
            )
            await queue.put((i, None, current_payload, current_target_base_name))

        if interrupt_event.is_set():
            logger.warning("Producer: Interrupt detected after finishing generation %s. Stopping.", i)
            break

    logger.info("Sequential Producer finished.")


def _save_run_manifest_if_needed(optimizer: Optimizer) -> None:
    if not optimizer.current_run_manifest:
        return

    try:
        output_dir = Path(HydraConfig.get().runtime.output_dir)
        manifest_path = save_run_manifest(output_dir, optimizer.current_run_manifest)
        logger.debug(
            "Saved run manifest with %s entries to %s",
            len(optimizer.current_run_manifest),
            manifest_path,
        )
    except Exception as error:
        logger.warning("Could not save run manifest: %s", error)


async def run_trial_iteration(optimizer: Optimizer, params: dict[str, Any]) -> float | None:
    """Execute one optimizer trial end-to-end."""
    optimizer.iteration += 1
    optimizer.last_trial_scorer_summary = {"aggregate": {}, "payloads": []}
    effective_iteration = optimizer.iteration + optimizer.completed_trials
    iteration_start_time = time.time()

    iteration_type = "warmup" if effective_iteration <= optimizer.cfg.optimizer.init_points else "optimization"
    if effective_iteration in {1, optimizer.cfg.optimizer.init_points + 1}:
        logger.info("\n%s Starting %s Phase %s>", "-" * 10, iteration_type, "-" * 10)

    logger.info("\n--- %s - Iteration: %s ---", iteration_type, effective_iteration)
    logger.info("Optimizer proposed parameters: %s", params)

    payloads, target_paths = optimizer.prompter.render_payloads(optimizer.cfg.batch_size)
    if not payloads:
        logger.error("Prompter generated no payloads.")
        raise RuntimeError("Prompter failed to generate any payloads.")

    current_scorers = {s.lower() for s in optimizer.cfg.scorer_method}
    scorer_setup_fp = optimizer.scorer_setup_fp
    cache_results = []
    overall_tier = "full_hit"

    for payload in payloads:
        image_hash = calculate_image_hash(params, payload, optimizer.generation_setup_fp)
        cached = optimizer.history_cache.get(image_hash)

        if cached and cached.get("final_score") is not None:
            cached_scorers = set(cached.get("scores", {}).keys()) - {"combined"}
            cached_fp = cached.get("scorer_setup_fp")
            if (
                cached_scorers
                and cached_scorers == current_scorers
                and isinstance(cached_fp, str)
                and cached_fp == scorer_setup_fp
            ):
                cache_results.append((image_hash, cached, payload, "full_hit"))
            elif cached.get("full_path") and Path(cached["full_path"]).exists():
                cache_results.append((image_hash, cached, payload, "partial_hit"))
                overall_tier = "partial_hit"
            else:
                overall_tier = "full_miss"
                break
        elif cached and cached.get("full_path") and Path(cached["full_path"]).exists():
            cache_results.append((image_hash, cached, payload, "partial_hit"))
            overall_tier = "partial_hit"
        else:
            overall_tier = "full_miss"
            break

    if overall_tier == "full_hit" and cache_results:
        cached_scores = [c[1]["final_score"] for c in cache_results]
        cached_weights = [c[2].get("score_weight", 1.0) for c in cache_results]
        avg_score = optimizer.scorer.average_calc(cached_scores, cached_weights, optimizer.cfg.img_average_type)
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
        optimizer.last_trial_scorer_summary = build_trial_scorer_summary(
            payload_entries,
            final_score=avg_score,
            combine_scores=lambda values, weights: optimizer.scorer.average_calc(
                values,
                weights,
                optimizer.cfg.img_average_type,
            ),
        )
        elapsed = time.time() - iteration_start_time
        logger.info("CACHE HIT: All %s images reused. Score: %.4f (%.2fs)", len(cached_scores), avg_score, elapsed)
        return avg_score

    if overall_tier == "partial_hit" and cache_results:
        logger.info(
            "PARTIAL CACHE HIT: %s images found, re-scoring with current scorers (%s)",
            len(cache_results),
            ", ".join(sorted(current_scorers)),
        )
        rescored_scores = []
        rescored_weights = []
        payload_entries: list[dict[str, Any]] = []
        for idx, (image_hash, cached, payload, tier) in enumerate(cache_results):
            payload_name = target_paths[idx] if idx < len(target_paths) else f"payload_{idx}"
            if tier == "full_hit":
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

            try:
                image = Image.open(cached["full_path"])
                prompt_for_scorer = payload.get("prompt", "")
                individual_score = await optimizer.scorer.score(image, prompt_for_scorer, name=payload_name)
                rescored_scores.append(individual_score)
                rescored_weights.append(payload.get("score_weight", 1.0))

                scorer_results = dict(optimizer.scorer.last_scorer_results)
                scorer_results["combined"] = individual_score
                cached["scores"] = scorer_results
                cached["final_score"] = individual_score
                cached["scorer_setup_fp"] = scorer_setup_fp

                try:
                    output_dir = Path(HydraConfig.get().runtime.output_dir)
                    optimizer.current_run_manifest[image_hash] = build_run_manifest_entry(
                        image_path=Path(cached["full_path"]),
                        output_dir=output_dir,
                        scorer_results=scorer_results,
                        final_score=individual_score,
                        scorer_setup_fp=scorer_setup_fp,
                    )
                except Exception:
                    optimizer.current_run_manifest[image_hash] = build_run_manifest_entry(
                        image_path=Path(cached["full_path"]),
                        output_dir=None,
                        scorer_results=scorer_results,
                        final_score=individual_score,
                        scorer_setup_fp=scorer_setup_fp,
                    )

                payload_entries.append(
                    {
                        "name": payload_name,
                        "weight": payload.get("score_weight", 1.0),
                        "scores": scorer_results,
                        "combined": individual_score,
                    }
                )
                logger.info("  Re-scored: %.4f", individual_score)
                image.close()
            except Exception as error:
                logger.warning("Failed to re-score cached image: %s", error)
                overall_tier = "full_miss"
                break

        if overall_tier == "partial_hit" and rescored_scores:
            avg_score = optimizer.scorer.average_calc(rescored_scores, rescored_weights, optimizer.cfg.img_average_type)
            optimizer.last_trial_scorer_summary = build_trial_scorer_summary(
                payload_entries,
                final_score=avg_score,
                combine_scores=lambda values, weights: optimizer.scorer.average_calc(
                    values,
                    weights,
                    optimizer.cfg.img_average_type,
                ),
            )
            elapsed = time.time() - iteration_start_time
            logger.info(
                "PARTIAL HIT complete: Score: %.4f (re-scored in %.2fs, skipped merge+gen)",
                avg_score,
                elapsed,
            )
            _save_run_manifest_if_needed(optimizer)
            return avg_score

    concurrency_limit = optimizer.cfg.get("generator_concurrency_limit", 2)
    keepalive_interval = optimizer.cfg.get("generator_keepalive_interval", 60)
    total_timeout_seconds = optimizer.cfg.get("generator_total_timeout", 3600)

    connector = aiohttp.TCPConnector(limit=concurrency_limit, keepalive_timeout=keepalive_interval)
    timeout_settings = aiohttp.ClientTimeout(total=total_timeout_seconds) if total_timeout_seconds > 0 else None

    async with aiohttp.ClientSession(connector=connector, timeout=timeout_settings) as session:
        try:
            await optimizer.generator.unload_model(session)
            logger.info("Unload model request processed.")
        except Exception as unload_error:
            logger.warning("Unload request failed (continuing): %s", unload_error)

        model_path: Path | None = None

        try:
            start_merge_time = time.time()

            effective_iteration = optimizer.iteration + optimizer.completed_trials
            if optimizer.cfg.optimization_mode == "merge":
                optimizer.merger.output_file = optimizer.merger.create_model_output_name(iteration=effective_iteration)
                model_path = optimizer.merger.merge(
                    params=params,
                    param_info=optimizer.param_info,
                    cache=optimizer.cache,
                    iteration=effective_iteration,
                )
            elif optimizer.cfg.optimization_mode == "layer_adjust":
                optimizer.merger.output_file = optimizer.merger.create_model_output_name(iteration=effective_iteration)
                model_path = optimizer.merger.layer_adjust(params, optimizer.cfg)
            elif optimizer.cfg.optimization_mode == "recipe":
                model_path = optimizer.merger.recipe_optimization(
                    params=params,
                    param_info=optimizer.param_info,
                    cache=optimizer.cache,
                    iteration=effective_iteration,
                )
            else:
                raise ValueError(f"Invalid optimization mode: {optimizer.cfg.optimization_mode}")

            merge_duration = time.time() - start_merge_time
            logger.info("Model processing took %.2f seconds.", merge_duration)

        except (ValueError, TypeError, FileNotFoundError) as config_error:
            if fail_on_error_enabled(optimizer.cfg):
                logger.error("FATAL CONFIGURATION ERROR: %s", config_error, exc_info=True)
                logger.error("Halting optimization due to unrecoverable setup error.")
                raise
            logger.error(
                "Trial failed during model processing with a configuration/runtime error: %s",
                config_error,
                exc_info=True,
            )
            logger.warning("Continuing optimization because fail_on_error is disabled.")
            return float("-inf")
        except Exception as error:
            if fail_on_error_enabled(optimizer.cfg):
                logger.error("A runtime error occurred during the trial: %s", error, exc_info=True)
                logger.error("Halting optimization because fail_on_error is enabled.")
                raise
            logger.error("Trial failed during model processing: %s", error, exc_info=True)
            logger.warning("Continuing optimization because fail_on_error is disabled.")
            return float("-inf")

        if not model_path or not model_path.exists():
            error_message = f"CRITICAL: Model processing finished but the output file was not created at '{model_path}'. Halting."
            logger.error(error_message)
            raise RuntimeError(error_message)

        logger.info("Performing immediate post-merge memory cleanup before image generation...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("PyTorch CUDA cache cleared.")

        try:
            await optimizer.generator.load_model(model_path, session)
            logger.info("Load model request processed for %s.", model_path.name)
        except Exception as load_error:
            logger.error("Failed to load model: %s. Cannot generate.", load_error)
            raise RuntimeError(f"Critical Load Failure: {load_error}") from load_error

        start_gen_score_time = time.time()
        scores = []
        norm_weights = []
        payload_entries = []
        image_queue = asyncio.Queue(maxsize=concurrency_limit)
        interrupt_event = asyncio.Event()
        total_expected_images = len(payloads)
        interrupt_triggered = False
        fake_score_value = 0.0
        producer_task = None

        try:
            producer_task = asyncio.create_task(
                sequential_producer(
                    optimizer,
                    payloads,
                    target_paths,
                    image_queue,
                    session,
                    interrupt_event,
                )
            )

            logger.info(
                "Consumer: Waiting to receive and score up to %s images sequentially...",
                total_expected_images,
            )

            for i in range(total_expected_images):
                if interrupt_event.is_set():
                    logger.warning("Consumer: Interrupt detected before waiting for image %s. Stopping consumption.", i)
                    interrupt_triggered = True
                    break

                logger.debug("Consumer: Waiting for image %s from queue...", i)
                try:
                    effective_timeout = total_timeout_seconds if total_timeout_seconds else 3600
                    queue_item = await asyncio.wait_for(image_queue.get(), timeout=effective_timeout)
                except TimeoutError:
                    logger.error("Consumer: Timeout waiting for image %s from queue. Stopping.", i)
                    interrupt_triggered = True
                    interrupt_event.set()
                    break

                if queue_item is None:
                    logger.info("Consumer: Received end signal from producer.")
                    break

                order_index, image, current_payload, current_target_base_name = queue_item

                if order_index != i:
                    logger.error("Consumer: Order mismatch! Expected index %s, got %s. Stopping.", i, order_index)
                    interrupt_triggered = True
                    interrupt_event.set()
                    image_queue.task_done()
                    break

                if image is None:
                    logger.warning(
                        "Consumer: Received failure signal for image %s ('%s'). Skipping scoring.",
                        i,
                        current_target_base_name,
                    )
                    image_queue.task_done()
                    continue

                logger.info(
                    "Consumer: Scoring image %s/%s ('%s')...",
                    i + 1,
                    total_expected_images,
                    current_target_base_name,
                )
                score_start_time = time.time()
                processed_item = False
                try:
                    prompt_for_scorer = current_payload.get("prompt", "")
                    individual_score = await optimizer.scorer.score(
                        image,
                        prompt_for_scorer,
                        name=current_target_base_name,
                    )

                    if individual_score == -1.0:
                        logger.warning("Consumer: OVERRIDE_SCORE detected during scoring of image %s.", i)
                        interrupt_event.set()
                        fake_score_value = optimizer.scorer.handle_override_prompt()
                        interrupt_triggered = True
                        break

                    score_duration = time.time() - score_start_time
                    logger.debug("Scoring index %s took %.2fs.", i, score_duration)

                    weight = current_payload.get("score_weight", 1.0)
                    scorer_results = dict(optimizer.scorer.last_scorer_results)
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

                    if optimizer.cfg.save_imgs:
                        effective_iteration = optimizer.iteration + optimizer.completed_trials
                        scorer_results["combined"] = individual_score
                        optimizer.save_img(
                            image,
                            current_target_base_name,
                            individual_score,
                            effective_iteration,
                            i,
                            current_payload,
                            params=params,
                            scorer_results=scorer_results,
                        )

                    processed_item = True

                except Exception as score_error:
                    logger.error(
                        "Consumer: Error scoring image '%s' (index %s): %s",
                        current_target_base_name,
                        i,
                        score_error,
                        exc_info=True,
                    )
                    processed_item = True
                finally:
                    if processed_item:
                        image_queue.task_done()

        except asyncio.CancelledError:
            logger.warning("Main task cancelled.")
            interrupt_event.set()
        except Exception as main_error:
            logger.error("Error during concurrent generation/scoring: %s", main_error, exc_info=True)
            interrupt_event.set()
        finally:
            if producer_task and not producer_task.done():
                logger.info("Cancelling producer task...")
                producer_task.cancel()
                with suppress(asyncio.CancelledError):
                    await producer_task
                logger.info("Producer task cancelled.")

    gen_score_duration = time.time() - start_gen_score_time
    logger.info("Generation & scoring phase took %.2f seconds.", gen_score_duration)

    if interrupt_triggered:
        logger.info("Iteration interrupted by override. Using final score: %.4f", fake_score_value)
        avg_score = fake_score_value
    elif not scores:
        logger.warning("No images were successfully scored.")
        raise RuntimeError("Generation failed: No images were produced or scored.")
    else:
        try:
            avg_score = optimizer.scorer.average_calc(scores, norm_weights, optimizer.cfg.img_average_type)
            logger.info("Calculated average score: %.4f", avg_score)
        except Exception as avg_error:
            logger.error("Error calculating average score: %s", avg_error, exc_info=True)
            raise RuntimeError(f"Score calculation error: {avg_error}") from avg_error

    optimizer.last_trial_scorer_summary = build_trial_scorer_summary(
        payload_entries,
        final_score=avg_score,
        combine_scores=lambda values, weights: optimizer.scorer.average_calc(
            values,
            weights,
            optimizer.cfg.img_average_type,
        ),
    )

    optimizer.update_best_score(params, avg_score)
    optimizer.scorer.unload_lazy_models()

    iteration_duration = time.time() - iteration_start_time
    logger.info(
        "Iteration %s finished. Final Score for Optimizer: %.4f. Duration: %.2fs",
        optimizer.iteration,
        avg_score,
        iteration_duration,
    )

    _save_run_manifest_if_needed(optimizer)
    return avg_score
