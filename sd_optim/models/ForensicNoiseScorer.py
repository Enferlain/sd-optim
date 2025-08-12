import cv2
import numpy as np
from PIL import Image
import io
from rembg import new_session, remove

class ForensicNoiseScorer:
    def __init__(
        self,
        noise_threshold: float = 5.0,
        rembg_session=None,
    ):
        """
        Scorer for detecting forensic noise in image backgrounds.
        A higher score indicates less noise.

        Args:
            noise_threshold (float): The standard deviation threshold to normalize the noise score.
            rembg_session: An existing rembg session. If None, a new one is created.
        """
        self.noise_threshold = noise_threshold
        if rembg_session is None:
            self.rembg_session = new_session(providers=['CPUExecutionProvider'])
        else:
            self.rembg_session = rembg_session

    def _pil_to_cv2(self, pil_image: Image.Image) -> np.ndarray:
        """Convert PIL Image to OpenCV BGR format."""
        return cv2.cvtColor(np.array(pil_image.convert('RGB')), cv2.COLOR_RGB2BGR)

    def _isolate_background(self, cv2_image: np.ndarray) -> np.ndarray:
        """Isolates the background of an image, returning only the background pixels."""
        # Convert cv2 image to bytes for rembg
        _, buffer = cv2.imencode('.png', cv2_image)
        input_bytes = buffer.tobytes()

        # Remove foreground
        output_bytes = remove(input_bytes, session=self.rembg_session)
        
        # Read the image with alpha channel
        output_image_with_alpha = cv2.imdecode(np.frombuffer(output_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
        
        # Create a mask from the alpha channel where transparent pixels are the background
        if output_image_with_alpha.shape[2] == 4:
            alpha_channel = output_image_with_alpha[:, :, 3]
            background_mask = (alpha_channel == 0)
            
            # Get the original background pixels using the mask
            background_pixels = cv2_image[background_mask]
            return background_pixels
        else:
            # If no alpha channel, assume no background was removed
            return np.array([]) # Return empty array

    def _detect_colored_noise(self, pixels: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Extracts RGB noise from a region of pixels."""
        if len(pixels.shape) < 2: # Not a valid image region
            return pixels
        # We need to reconstruct a temporary image to apply the filter
        # This is a simplification; for a robust solution, we'd need to handle pixel clouds better.
        # For now, we assume the background is mostly flat and analyze the pixel values directly.
        return pixels # Placeholder for a more complex implementation if needed

    def _detect_structural_noise(self, pixels: np.ndarray, method: str = "gaussian_diff") -> float:
        """
        Calculates a noise score based on the standard deviation of pixels.
        This is a simplified proxy for the more complex methods.
        """
        if pixels.size == 0:
            return 0.0
        
        # For structural noise, we are interested in the luminance variation.
        # Convert to grayscale to measure intensity variation.
        if len(pixels.shape) > 1 and pixels.shape[1] == 3:
             gray_pixels = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY)
        else:
             # It might already be grayscale or a flat array
             gray_pixels = pixels

        noise_level = np.std(gray_pixels)
        return noise_level

    def score(self, image: Image.Image, detection_method: str = "structural") -> float:
        """
        Scores the image based on noise in the background.
        Higher score = less noise.
        """
        cv2_image = self._pil_to_cv2(image)
        
        background_pixels = self._isolate_background(cv2_image)

        if background_pixels.size == 0:
            # No background found or isolated, cannot score.
            # Return a neutral score or handle as an edge case.
            return 5.0 

        if detection_method == "structural":
            noise_level = self._detect_structural_noise(background_pixels)
        elif detection_method == "colored":
            # This is more complex as it requires filtering, which works on 2D images, not pixel lists.
            # As a proxy, we can measure the std dev of each color channel.
            noise_level_r = np.std(background_pixels[:, 2])
            noise_level_g = np.std(background_pixels[:, 1])
            noise_level_b = np.std(background_pixels[:, 0])
            noise_level = (noise_level_r + noise_level_g + noise_level_b) / 3
        else:
            noise_level = self._detect_structural_noise(background_pixels)

        # Invert the score: lower noise_level should result in a higher score.
        final_score = 10.0 * (1.0 - min(noise_level / self.noise_threshold, 1.0))
        
        return float(np.clip(final_score, 0.0, 10.0))
