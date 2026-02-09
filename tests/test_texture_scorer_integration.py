import asyncio
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
from sd_optim.scorer import AestheticScorer
import logging

# Simple integration test for TextureScorer via AestheticScorer

logging.basicConfig(level=logging.INFO)


async def test_texture_scorer():
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
    dummy_img = Image.fromarray(
        np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    )

    print("Initializing AestheticScorer...")
    scorer = AestheticScorer(cfg)

    print("Testing TextureScorer...")
    try:
        score = await scorer.score(dummy_img, prompt="a test image")
        print(f"Final Score: {score}")
        assert 0 <= score <= 10
        print("Test passed!")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_texture_scorer())
