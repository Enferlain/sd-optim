import cv2
import numpy as np
from PIL import Image
import rembg
from sklearn.cluster import KMeans
import joblib


class HybridNoiseScorer:
    def __init__(
        self,
        kernel_size: int = 3,
        noise_threshold: float = 20.0,
        color_tolerance: int = 30,
        rembg_session=None,
    ):
        """
        A smart hybrid scorer that isolates the background based on its dominant color
        and then scores the noise level within that isolated area.
        """
        self.kernel_size = kernel_size
        self.noise_threshold = noise_threshold
        self.color_tolerance = color_tolerance
        if rembg_session is None:
            self.rembg_session = rembg.new_session("u2net")
        else:
            self.rembg_session = rembg_session

    def _create_noise_map(self, image_bgr: np.ndarray) -> np.ndarray:
        denoised_img = cv2.medianBlur(image_bgr, self.kernel_size)
        original_int = image_bgr.astype(np.int16)
        denoised_int = denoised_img.astype(np.int16)
        noise_map = original_int - denoised_int
        noise_map_abs = np.abs(noise_map).astype(np.uint8)
        b, g, r = cv2.split(noise_map_abs)
        b_eq, g_eq, r_eq = cv2.equalizeHist(b), cv2.equalizeHist(g), cv2.equalizeHist(r)
        return cv2.merge([b_eq, g_eq, r_eq])

    def score(self, image: Image.Image) -> float:
        original_cv_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

        # Step 1: Get a rough background sample from the AI
        output_image_pil = rembg.remove(image, session=self.rembg_session)
        output_image_with_alpha = cv2.cvtColor(np.array(output_image_pil), cv2.COLOR_RGBA2BGRA)

        if output_image_with_alpha.shape[2] != 4:
            return 5.0

        initial_alpha_mask = output_image_with_alpha[:, :, 3] < 128
        if not np.any(initial_alpha_mask):
            return 10.0

        initial_bg_pixels = original_cv_bgr[initial_alpha_mask]

        # Step 2: Find the true dominant color of the background sample
        kmeans = KMeans(n_clusters=4, random_state=42, n_init="auto")
        with joblib.parallel_backend("threading", n_jobs=1):
            kmeans.fit(initial_bg_pixels)
        unique, counts = np.unique(kmeans.labels_, return_counts=True)
        dominant_cluster_index = unique[counts.argmax()]
        dominant_bg_color = kmeans.cluster_centers_[dominant_cluster_index]

        # Step 3: Create a new, precise mask based on color similarity
        color_diff = np.linalg.norm(original_cv_bgr.astype(np.float32) - dominant_bg_color, axis=-1)
        final_background_mask = color_diff < self.color_tolerance

        # Step 4: Run noise analysis on the new, perfect mask
        if np.any(final_background_mask):
            isolated_bg_bgr = np.zeros_like(original_cv_bgr)
            isolated_bg_bgr[final_background_mask] = original_cv_bgr[final_background_mask]

            noise_map_bgr = self._create_noise_map(isolated_bg_bgr)
            noise_map_gray = cv2.cvtColor(noise_map_bgr, cv2.COLOR_BGR2GRAY)
            noise_level = np.mean(noise_map_gray[final_background_mask])
        else:
            noise_level = 0.0

        noise_score = 10.0 * (1.0 - min(noise_level / self.noise_threshold, 1.0))
        final_score = np.clip(noise_score, 0.0, 10.0)

        return float(final_score)
