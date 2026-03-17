from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
from PIL import Image

from sd_optim.core.optimizer_cache import calculate_image_hash


def test_optimizer_runtime_module_exports_trial_helpers() -> None:
    module = importlib.import_module("sd_optim.core.optimizer_runtime")

    assert callable(module.run_trial_iteration)
    assert callable(module.sequential_producer)


def test_sequential_producer_enqueues_generated_image() -> None:
    module = importlib.import_module("sd_optim.core.optimizer_runtime")

    class DummyGenerator:
        async def generate(self, payload, cfg, session):  # noqa: ARG002
            yield Image.new("RGB", (1, 1), color=(255, 0, 0))

    optimizer = SimpleNamespace(generator=DummyGenerator(), cfg=OmegaConf.create({}))
    queue: asyncio.Queue = asyncio.Queue()

    async def run() -> tuple:
        await module.sequential_producer(
            optimizer,
            payloads=[{"prompt": "test"}],
            target_paths=["payload_a"],
            queue=queue,
            session=None,
            interrupt_event=asyncio.Event(),
        )
        return await queue.get()

    order_index, image, payload, name = asyncio.run(run())

    assert order_index == 0
    assert payload == {"prompt": "test"}
    assert name == "payload_a"
    assert image.size == (1, 1)
    image.close()


def test_run_trial_iteration_returns_cached_score_on_full_hit() -> None:
    module = importlib.import_module("sd_optim.core.optimizer_runtime")
    params = {"alpha": 0.5}
    payload = {"prompt": "cached", "seed": 1, "score_weight": 1.0}
    image_hash = calculate_image_hash(params, payload, generation_setup_fp="gen-fp")

    class DummyPrompter:
        def render_payloads(self, batch_size):  # noqa: ARG002
            return [payload], ["cached_payload"]

    class DummyScorer:
        def average_calc(self, scores, weights, average_type):  # noqa: ARG002
            return sum(score * weight for score, weight in zip(scores, weights)) / sum(weights)

    optimizer = SimpleNamespace(
        iteration=-1,
        completed_trials=0,
        last_trial_scorer_summary={},
        cfg=OmegaConf.create(
            {
                "batch_size": 1,
                "img_average_type": "arithmetic",
                "scorer_method": ["manual"],
                "optimizer": {"init_points": 1},
            }
        ),
        prompter=DummyPrompter(),
        scorer=DummyScorer(),
        history_cache={
            image_hash: {
                "final_score": 0.75,
                "scores": {"manual": 0.75, "combined": 0.75},
                "scorer_setup_fp": "score-fp",
            }
        },
        scorer_setup_fp="score-fp",
        generation_setup_fp="gen-fp",
    )

    result = asyncio.run(module.run_trial_iteration(optimizer, params))

    assert result == 0.75
    assert optimizer.last_trial_scorer_summary["aggregate"]["combined"] == 0.75


def test_run_trial_iteration_executes_generation_path_with_stubs(tmp_path: Path) -> None:
    module = importlib.import_module("sd_optim.core.optimizer_runtime")

    class DummyPrompter:
        def render_payloads(self, batch_size):  # noqa: ARG002
            return [{"prompt": "generated", "score_weight": 1.0}], ["generated_payload"]

    class DummyGenerator:
        async def unload_model(self, session):  # noqa: ARG002
            return None

        async def load_model(self, model_path, session):  # noqa: ARG002
            assert model_path.exists()
            return None

        async def generate(self, payload, cfg, session):  # noqa: ARG002
            yield Image.new("RGB", (2, 2), color=(0, 255, 0))

    class DummyScorer:
        last_scorer_results = {"manual": 0.8}

        async def score(self, image, prompt, name):  # noqa: ARG002
            assert image.size == (2, 2)
            return 0.8

        def average_calc(self, scores, weights, average_type):  # noqa: ARG002
            return sum(score * weight for score, weight in zip(scores, weights)) / sum(weights)

        def unload_lazy_models(self):
            return None

        def handle_override_prompt(self):
            return 0.0

    class DummyMerger:
        def __init__(self, output_path: Path):
            self.output_path = output_path
            self.output_file: Path | None = None
            self.best_output_file: Path | None = None

        def create_model_output_name(self, iteration):  # noqa: ARG002
            return self.output_path

        def merge(self, params, param_info, cache, iteration):  # noqa: ARG002
            self.output_path.write_text("model-bytes", encoding="utf-8")
            self.output_file = self.output_path
            return self.output_path

    update_calls: list[tuple[dict, float]] = []
    merger = DummyMerger(tmp_path / "generated_model.safetensors")
    optimizer = SimpleNamespace(
        iteration=-1,
        completed_trials=0,
        last_trial_scorer_summary={},
        cfg=OmegaConf.create(
            {
                "batch_size": 1,
                "img_average_type": "arithmetic",
                "scorer_method": ["manual"],
                "optimization_mode": "merge",
                "generator_concurrency_limit": 1,
                "generator_keepalive_interval": 60,
                "generator_total_timeout": 10,
                "save_imgs": False,
                "optimizer": {"init_points": 1},
            }
        ),
        prompter=DummyPrompter(),
        generator=DummyGenerator(),
        scorer=DummyScorer(),
        merger=merger,
        param_info={},
        cache={},
        history_cache={},
        current_run_manifest={},
        scorer_setup_fp="score-fp",
        generation_setup_fp="gen-fp",
        save_img=lambda *args, **kwargs: None,  # noqa: ARG005
        update_best_score=lambda params, avg_score: update_calls.append((params, avg_score)),
    )

    result = asyncio.run(module.run_trial_iteration(optimizer, {"alpha": 0.25}))

    assert result == 0.8
    assert update_calls == [({"alpha": 0.25}, 0.8)]
    assert optimizer.last_trial_scorer_summary["aggregate"]["combined"] == 0.8
