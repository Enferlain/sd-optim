import numpy as np
from PIL import Image
import io
from rembg import new_session, remove
from sklearn.cluster import KMeans
import joblib
import logging

logger = logging.getLogger(__name__)

class BackgroundBlacknessScorer:
    def __init__(self, rembg_session=None, n_clusters: int = 4, sensitivity: float = 0.008):
        """
        Scorer for measuring how black the background of an image is.
        A higher score means a blacker background.

        Args:
            rembg_session: An existing rembg session. If None, a new one is created.
            n_clusters (int): Number of clusters for KMeans to find the dominant color.
            sensitivity (float): Sensitivity for the exponential scoring function.
        """
        if rembg_session is None:
            logger.info("Creating new rembg session for BackgroundBlacknessScorer.")
            self.rembg_session = new_session(providers=['CPUExecutionProvider'])
        else:
            self.rembg_session = rembg_session
        self.n_clusters = n_clusters
        self.sensitivity = sensitivity

    def score(self, image: Image.Image) -> float:
        """
        Calculates a score from 0.0 to 10.0 based on how close the dominant
        background color is to pure black (0, 0, 0).
        """
        if self.rembg_session is None:
            logger.error("Rembg session not initialized. Cannot perform blackness scoring.")
            return 0.0

        target_black = np.array([0, 0, 0])

        try:
            original_image = image.convert('RGB')
            original_pixels = np.array(original_image)
            
            buffer = io.BytesIO()
            original_image.save(buffer, format="PNG")
            input_bytes = buffer.getvalue()

            output_bytes = remove(input_bytes, session=self.rembg_session)
            
            # Use frombuffer to avoid writing to disk
            output_image_with_alpha = Image.open(io.BytesIO(output_bytes))
            output_pixels = np.array(output_image_with_alpha)

            if output_pixels.shape[2] != 4:
                logger.warning("Blackness score: Image has no alpha channel after background removal.")
                return 0.0 # Or a neutral score like 5.0

            alpha_channel = output_pixels[:, :, 3]
            background_mask = (alpha_channel == 0)

            if not np.any(background_mask):
                logger.warning("Blackness score: AI found no background pixels.")
                return 10.0 # No background means perfect background

            original_background_pixels = original_pixels[background_mask]

            if len(original_background_pixels) < self.n_clusters:
                dominant_color = np.mean(original_background_pixels, axis=0)
            else:
                kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init='auto')
                with joblib.parallel_backend('threading', n_jobs=1):
                    kmeans.fit(original_background_pixels)

                unique, counts = np.unique(kmeans.labels_, return_counts=True)
                dominant_cluster_index = unique[counts.argmax()]
                dominant_color = kmeans.cluster_centers_[dominant_cluster_index]

            distance = np.linalg.norm(dominant_color - target_black)
            score_0_to_1 = np.exp(-self.sensitivity * distance)
            final_score = score_0_to_1 * 10.0

            return float(np.clip(final_score, 0.0, 10.0))

        except Exception as e:
            logger.error(f"An error occurred during blackness scoring: {e}", exc_info=True)
            return 0.0
