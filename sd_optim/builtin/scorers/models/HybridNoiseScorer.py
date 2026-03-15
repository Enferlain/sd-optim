import logging

import cv2
import joblib
import numpy as np
from PIL import Image
import rembg
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


def _coverage_confidence(
    coverage: float,
    min_coverage: float,
    target_coverage: float,
) -> float:
    if coverage <= min_coverage:
        return 0.0
    if target_coverage <= min_coverage:
        return 1.0
    return float(np.clip((coverage - min_coverage) / (target_coverage - min_coverage), 0.0, 1.0))


class HybridNoiseScorer:
    def __init__(
        self,
        kernel_size: int = 3,
        noise_threshold: float = 20.0,
        color_tolerance: int = 30,
        rembg_session: object | None = None,
        min_background_coverage: float = 0.01,
        target_background_coverage: float = 0.25,
        n_clusters: int = 4,
        use_rembg: bool = True,
    ):
        """Score background cleanliness using rembg-guided background masking."""
        self.kernel_size = kernel_size
        self.noise_threshold = noise_threshold
        self.color_tolerance = color_tolerance
        self.min_background_coverage = min_background_coverage
        self.target_background_coverage = target_background_coverage
        self.n_clusters = n_clusters
        self.use_rembg = use_rembg

        if self.use_rembg:
            if rembg_session is None:
                self.rembg_session = rembg.new_session("u2net")
            else:
                self.rembg_session = rembg_session
        else:
            self.rembg_session = None

    def _create_noise_map(self, image_bgr: np.ndarray) -> np.ndarray:
        denoised_img = cv2.medianBlur(image_bgr, self.kernel_size)
        original_int = image_bgr.astype(np.int16)
        denoised_int = denoised_img.astype(np.int16)
        noise_map = original_int - denoised_int
        noise_map_abs = np.abs(noise_map).astype(np.uint8)
        b, g, r = cv2.split(noise_map_abs)
        b_eq = cv2.equalizeHist(b)
        g_eq = cv2.equalizeHist(g)
        r_eq = cv2.equalizeHist(r)
        return cv2.cvtColor(cv2.merge([b_eq, g_eq, r_eq]), cv2.COLOR_BGR2GRAY)

    def _build_background_mask(self, image: Image.Image, image_bgr: np.ndarray) -> np.ndarray:
        if not self.use_rembg or self.rembg_session is None:
            return np.zeros(image_bgr.shape[:2], dtype=bool)

        try:
            output_image_pil = rembg.remove(image, session=self.rembg_session)
            output_array = np.array(output_image_pil)
        except Exception as exc:
            logger.warning("HybridNoiseScorer: rembg failed; returning empty mask (%s)", exc)
            return np.zeros(image_bgr.shape[:2], dtype=bool)

        if output_array.ndim != 3 or output_array.shape[2] != 4:
            logger.warning("HybridNoiseScorer: rembg output has no alpha channel.")
            return np.zeros(image_bgr.shape[:2], dtype=bool)

        initial_background_mask = output_array[:, :, 3] < 128
        if not np.any(initial_background_mask):
            logger.info("HybridNoiseScorer: no background pixels detected by rembg.")
            return np.zeros(image_bgr.shape[:2], dtype=bool)

        bg_pixels = image_bgr[initial_background_mask]
        if bg_pixels.shape[0] == 0:
            return np.zeros(image_bgr.shape[:2], dtype=bool)

        if bg_pixels.shape[0] < self.n_clusters:
            dominant_color = np.mean(bg_pixels, axis=0)
        else:
            kmeans = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init="auto",
            )
            with joblib.parallel_backend("threading", n_jobs=1):
                kmeans.fit(bg_pixels)
            unique, counts = np.unique(kmeans.labels_, return_counts=True)
            dominant_cluster_index = unique[counts.argmax()]
            dominant_color = kmeans.cluster_centers_[dominant_cluster_index]

        color_diff = np.linalg.norm(
            image_bgr.astype(np.float32) - dominant_color.astype(np.float32),
            axis=-1,
        )
        refined_background_mask = initial_background_mask & (color_diff < self.color_tolerance)
        if refined_background_mask.sum() < 64:
            return initial_background_mask
        return refined_background_mask

    def _score_from_mask(self, image_bgr: np.ndarray, mask: np.ndarray) -> float:
        if mask is None or not np.any(mask):
            return 0.0

        coverage = float(mask.mean())
        if coverage <= self.min_background_coverage:
            return 0.0

        noise_map_gray = self._create_noise_map(image_bgr)
        noise_level = float(np.mean(noise_map_gray[mask]))
        raw_score = 10.0 * (1.0 - min(noise_level / self.noise_threshold, 1.0))
        coverage_factor = _coverage_confidence(
            coverage=coverage,
            min_coverage=self.min_background_coverage,
            target_coverage=self.target_background_coverage,
        )
        return float(np.clip(raw_score * coverage_factor, 0.0, 10.0))

    def score(self, image: Image.Image) -> float:
        image_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        background_mask = self._build_background_mask(image, image_bgr)
        return self._score_from_mask(image_bgr, background_mask)


class HybridNoiseFullImageScorer(HybridNoiseScorer):
    def __init__(self, kernel_size: int = 3, noise_threshold: float = 20.0):
        super().__init__(
            kernel_size=kernel_size,
            noise_threshold=noise_threshold,
            color_tolerance=0,
            rembg_session=None,
            min_background_coverage=0.0,
            target_background_coverage=1.0,
            n_clusters=1,
            use_rembg=False,
        )

    def _build_background_mask(self, image: Image.Image, image_bgr: np.ndarray) -> np.ndarray:
        del image
        return np.ones(image_bgr.shape[:2], dtype=bool)
