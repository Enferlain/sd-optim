import os
import torch
from PIL import Image
import tempfile

# It is assumed that the 'hpsv3' package is installed.
# The package can be installed from the official repository:
# pip install git+https://github.com/MizzenAI/HPSv3.git
try:
    from hpsv3 import HPSv3RewardInferencer
except ImportError:
    raise ImportError("hpsv3 is not installed. Please install it by running: pip install git+https://github.com/MizzenAI/HPSv3.git")

class HPSv3Scorer:
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initializes the HPSv3Scorer.
        
        Args:
            model_path (str): Path to the HPSv3 model file (e.g., .safetensors).
            device (str): The device to run the model on ('cuda' or 'cpu').
        """
        self.device = device
        # The underlying hpsv3 library uses the 'model_name_or_path' parameter
        # to load a local model file instead of downloading from the Hub.
        try:
            from hpsv3 import HPSv3RewardInferencer
            self.model = HPSv3RewardInferencer(model_name_or_path=model_path, device=self.device)
        except ImportError:
            raise ImportError("hpsv3 is not installed. Please install it by running: pip install git+https://github.com/MizzenAI/HPSv3.git")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize HPSv3RewardInferencer with model_path='{model_path}': {e}")
        
    def score(self, prompt: str, image: Image.Image) -> tuple[float, float]:
        """
        Scores an image-text pair using HPSv3.
        
        Args:
            prompt: Text prompt associated with the image.
            image: PIL Image object (RGB format).
            
        Returns:
            tuple[float, float]: A tuple containing the clamped mu score (0-10) and the sigma value.
        """
        if not isinstance(image, Image.Image):
            raise TypeError("Image must be a PIL.Image.Image instance")

        pil_image = image.convert("RGB")

        temp_filepath = None
        try:
            # HPSv3's reward method expects a list of image paths.
            # We save the image to a temporary file to get a path.
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                temp_filepath = temp_file.name
                pil_image.save(temp_filepath)
            
            with torch.inference_mode():
                # The reward method expects lists of image paths and prompts.
                rewards = self.model.reward([temp_filepath], [prompt])
            
            # The reward method returns a list of tuples, where each tuple is (mu, sigma).
            # - mu is the mean preference score.
            # - sigma is the uncertainty (standard deviation).
            mu_score = rewards[0][0].item()
            sigma_score = rewards[0][1].item()

        finally:
            # Clean up the temporary file
            if temp_filepath and os.path.exists(temp_filepath):
                os.remove(temp_filepath)

        # According to the official HPSv3 benchmarks, the raw 'mu' score is not
        # strictly bounded to a 0-10 range. To meet the user's requirement and
        # maintain consistency with other scorers in this project (e.g., HPSv21),
        # we clamp the score to the [0, 10] interval.
        clamped_mu = max(0.0, min(mu_score, 10.0))
        
        return clamped_mu, sigma_score
