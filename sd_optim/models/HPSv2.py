# models/HPSv2.py
import os
import hpsv2
import torch
from PIL import Image
from pathlib import Path

class HPSv2Scorer:
    def __init__(self, model_path: Path, device: str = "cuda"):
        self.device = device
        self.model_path = model_path
        self.hps_version = "v2.1" # Official default
        
        # Validate model weights
        if not self.model_path.exists():
            raise FileNotFoundError(f"HPSv2.1 model not found at {model_path}")
            
        # Initialize using official API
        self.model = hpsv2.HPSv2(
            ckpt=str(model_path),
            version=self.hps_version,
            device=self.device,
            torch_dtype=torch.bfloat16 if "cuda" in device else torch.float32
        )
        
    def score(self, prompt: str, image: Image.Image) -> float:
        """
        Scores an image-text pair using HPSv2.1
        
        Args:
            prompt: Text prompt associated with the image
            image: PIL Image object (RGB format)
            
        Returns:
            float: Score between 0 (low quality) and 10 (high quality)
        """
        # Official scoring method handles preprocessing automatically
        with torch.inference_mode():
            score = hpsv2.score(
                imgs=image,
                prompt=prompt,
                hps_version=self.hps_version,
                device=self.device
            )
            
        # Ensure proper output range
        return float(torch.clamp(score, min=0, max=10).cpu().item())