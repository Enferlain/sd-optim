import numpy as np
from PIL import Image

class GammaNoiseScorer:
    def __init__(
        self,
        gamma_value: float = 0.3,
        noise_threshold: float = 10.0,
    ):
        """
        Scorer for detecting noise revealed by gamma correction.
        A higher score indicates less noise.

        Args:
            gamma_value (float): The gamma value to apply to the image.
            noise_threshold (float): The standard deviation threshold to normalize the noise score.
        """
        self.gamma_value = gamma_value
        self.noise_threshold = noise_threshold

    def _gamma_noise_reveal(self, image: Image.Image) -> Image.Image:
        """
        Applies gamma correction to reveal noise.
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        np_img = np.array(image, dtype=np.float32)
        
        # Apply gamma correction
        gamma_corrected = np.power(np_img / 255.0, self.gamma_value) * 255
        
        result = Image.fromarray(gamma_corrected.astype(np.uint8))
        return result

    def score(self, image: Image.Image) -> float:
        """
        Scores the image based on the amount of noise revealed by gamma correction.
        Higher score = less noise (better quality).
        Lower score = more noise (worse quality).
        """
        # Apply gamma correction to reveal noise
        gamma_image = self._gamma_noise_reveal(image)
        
        # Convert to numpy array for analysis
        gamma_array = np.array(gamma_image, dtype=np.float32)
        
        # Calculate the standard deviation of the pixel intensities
        # A higher std dev means more variation, which we interpret as noise
        noise_level = np.std(gamma_array)
        
        # Normalize the noise level to a score from 0.0 to 10.0
        # We want a higher score for lower noise, so we invert the relationship
        noise_score = 10.0 * (1.0 - min(noise_level / self.noise_threshold, 1.0))
        
        # Clip the final score to be strictly within the 0.0 to 10.0 range
        return np.clip(noise_score, 0.0, 10.0)
