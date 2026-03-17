import asyncio
import importlib.util
import os
from PIL import Image
import numpy as np
import pytest
from omegaconf import OmegaConf
from sd_optim.scorer import Scorer
import logging

# Simple integration test for TextureScorer via Scorer

logging.basicConfig(level=logging.INFO)


@pytest.mark.skipif(
    importlib.util.find_spec("rembg") is None or os.environ.get("RUN_OPTIONAL_SCORER_INTEGRATION") != "1",
    reason="optional texture scorer integration requires rembg and explicit opt-in",
)
def test_texture_scorer():
    # Mock configuration
    cfg = OmegaConf.create(
        {
            "scorer_method": ["textureclean"],
            "scorer_model_dir": "./models",  # Dummy dir
            "scorer_default_device": "cpu",
            "save_imgs": False,
            "scorer_print_individual": True,
            "scorer_average_type": "arithmetic",
            "scorer_weight": {"textureclean": 1.0},
            "scorer_device": {"textureclean": "cpu"},
            "scorer_lazy_load_list": [],
        }
    )

    # Create a dummy image
    dummy_img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))

    scorer = Scorer(cfg)

    try:
        score = asyncio.run(scorer.score(dummy_img, prompt="a test image"))
        assert 0 <= score <= 10
    except Exception as e:
        import traceback

        traceback.print_exc()
        pytest.fail(f"Texture scorer integration failed: {e}")


if __name__ == "__main__":
    test_texture_scorer()
