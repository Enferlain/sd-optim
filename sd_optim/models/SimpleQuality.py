# Add to your imports
import cv2
import numpy as np
from PIL import Image


class SimpleQualityScorer:
    def __init__(
        self,
        # v1.1: Added configurable thresholds and weights
        sharpness_threshold: float = 100.0,
        contrast_threshold: float = 50.0,
        weights: dict = None
    ):
        """
        Simple quality scorer - no external models needed.
        v1.1: Made thresholds and weights configurable.

        Args:
            sharpness_threshold (float): Laplacian variance value to distinguish blurry from sharp.
            contrast_threshold (float): Standard deviation of grayscale pixels to normalize contrast score.
            weights (dict): Dictionary of weights for each score component.
        """
        if weights is None:
            # Default weights from your original code! They're good!
            self.weights = {
                'saturation': 0.35,
                'sharpness': 0.30,
                'contrast': 0.20,
                'range': 0.15
            }
        else:
            self.weights = weights

        self.sharpness_threshold = sharpness_threshold
        self.contrast_threshold = contrast_threshold

    def score(self, image: Image.Image) -> float:
        """
        Score image quality from 0.0 to 10.0
        Higher = better quality (colorful, sharp, good contrast)
        Lower = poor quality (grey, blurry, washed out)
        """
        # Ensure the image is in RGB format for consistency
        rgb_image = image.convert('RGB')
        rgb_array = np.array(rgb_image)

        # Convert to different color spaces for analysis
        hsv_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
        gray_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)

        # 1. Saturation check (0.0 - 10.0)
        # Measures color intensity. Low score for grayscale/washed-out images.
        saturation_mean = np.mean(hsv_array[:, :, 1]) / 255.0
        saturation_score = saturation_mean * 10.0

        # 2. Sharpness check (0.0 - 10.0)
        # Measures edge detail. Low score for blurry images.
        laplacian_var = cv2.Laplacian(gray_array, cv2.CV_64F).var()
        if laplacian_var < self.sharpness_threshold:
            # Scale score from 0 to 4.0 for blurry images
            sharpness_score = (laplacian_var / self.sharpness_threshold) * 4.0
        else:
            # Scale score from 4.0 to 10.0 for sharp images
            # We map a range of [threshold, threshold + 400] to [4.0, 10.0]
            sharpness_score = 4.0 + min((laplacian_var - self.sharpness_threshold) / 400.0 * 6.0, 6.0)

        # 3. Contrast check (0.0 - 10.0)
        # Measures the range of dark to light. Low score for flat/dull images.
        contrast = np.std(gray_array.astype(np.float32))
        contrast_score = min((contrast / self.contrast_threshold) * 10.0, 10.0)

        # 4. Color range check (0.0 - 10.0)
        # Measures how much of the R, G, B spectrum is used.
        color_ranges = [np.ptp(rgb_array[:, :, i]) for i in range(3)]
        range_score = (np.mean(color_ranges) / 255.0) * 10.0

        # Weighted combination of all scores
        final_score = (
            saturation_score * self.weights['saturation'] +
            sharpness_score * self.weights['sharpness'] +
            contrast_score * self.weights['contrast'] +
            range_score * self.weights['range']
        )

        # Clip the final score to be strictly within the 0.0 to 10.0 range
        return np.clip(final_score, 0.0, 10.0)
