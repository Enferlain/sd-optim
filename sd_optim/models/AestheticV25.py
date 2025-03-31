import torch
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
from PIL import Image


class AestheticV25:
    def __init__(self, device="cuda"):
        self.device = device
        try:
            self.model, self.preprocessor = convert_v2_5_from_siglip(
                low_cpu_mem_usage=False,
                trust_remote_code=True,
            )
            self.model = self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize AestheticV25: {str(e)}")

    def score(self, prompt, image):
        # Input validation
        if not isinstance(image, Image.Image):
            raise TypeError("Image must be a PIL.Image.Image instance")

        try:
            # Ensure RGB mode
            pil_image = image.convert("RGB")

            # Process image with error handling
            pixel_values = self.preprocessor(
                images=pil_image,
                return_tensors="pt"
            ).pixel_values
            pixel_values = pixel_values.to(self.device)

            # Score with proper error handling
            with torch.inference_mode():
                score = self.model(pixel_values).logits.squeeze()
                return score.item()

        except Exception as e:
            print(f"Error in AestheticV25 scoring: {str(e)}")
