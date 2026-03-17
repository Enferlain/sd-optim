import cv2
import numpy as np
from PIL import Image
import rembg
import argparse
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import joblib


class HybridNoiseScorerV3:
    def __init__(
        self,
        kernel_size: int = 3,
        noise_threshold: float = 20.0,
        color_tolerance: int = 30,  # <-- OUR NEW SUPER-KNOB!
        rembg_session=None,
    ):
        """
        V3: A smarter hybrid scorer.
        1. Uses rembg to find a SAMPLE of the background.
        2. Uses KMeans to find the DOMINANT color of that sample.
        3. Creates a new, precise mask of all pixels matching that dominant color.
        4. Scores the noise within the new, precise mask.
        """
        self.kernel_size = kernel_size
        self.noise_threshold = noise_threshold
        self.color_tolerance = color_tolerance  # How close a color must be to the background color
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

    def score(self, image: Image.Image) -> tuple[float, Image.Image, Image.Image, float]:
        original_cv_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

        # --- Step 1: Get a ROUGH background sample from the AI ---
        _, buffer = cv2.imencode(".png", original_cv_bgr)
        input_bytes = buffer.tobytes()
        output_bytes = rembg.remove(input_bytes, session=self.rembg_session)
        output_image_with_alpha = cv2.imdecode(np.frombuffer(output_bytes, np.uint8), cv2.IMREAD_UNCHANGED)

        if output_image_with_alpha.shape[2] != 4:
            return 5.0, image, image, 0.0

        initial_alpha_mask = output_image_with_alpha[:, :, 3] < 128
        if not np.any(initial_alpha_mask):
            return 10.0, image, Image.fromarray(np.zeros_like(original_cv_bgr)), 0.0

        initial_bg_pixels = original_cv_bgr[initial_alpha_mask]

        # --- Step 2: Find the TRUE dominant color of the background sample ---
        kmeans = KMeans(n_clusters=4, random_state=42, n_init="auto")
        with joblib.parallel_backend("threading", n_jobs=1):
            kmeans.fit(initial_bg_pixels)
        unique, counts = np.unique(kmeans.labels_, return_counts=True)
        dominant_cluster_index = unique[counts.argmax()]
        dominant_bg_color = kmeans.cluster_centers_[dominant_cluster_index]

        # --- Step 3: Create a NEW, precise mask based on color similarity ---
        # Calculate the distance of every pixel in the original image to the dominant bg color
        color_diff = np.linalg.norm(original_cv_bgr.astype(np.float32) - dominant_bg_color, axis=-1)
        final_background_mask = color_diff < self.color_tolerance

        # --- Step 4: Run noise analysis on the new, perfect mask ---
        isolated_bg_bgr = np.zeros_like(original_cv_bgr)
        if np.any(final_background_mask):
            isolated_bg_bgr[final_background_mask] = original_cv_bgr[final_background_mask]

            noise_map_bgr = self._create_noise_map(isolated_bg_bgr)
            noise_map_gray = cv2.cvtColor(noise_map_bgr, cv2.COLOR_BGR2GRAY)
            noise_level = np.mean(noise_map_gray[final_background_mask])
        else:  # Handle cases where no background is found with the new mask
            noise_map_bgr = np.zeros_like(original_cv_bgr)
            noise_level = 0.0

        noise_score = 10.0 * (1.0 - min(noise_level / self.noise_threshold, 1.0))
        final_score = np.clip(noise_score, 0.0, 10.0)

        isolated_bg_pil = Image.fromarray(cv2.cvtColor(isolated_bg_bgr, cv2.COLOR_BGR2RGB))
        noise_map_pil = Image.fromarray(cv2.cvtColor(noise_map_bgr, cv2.COLOR_BGR2RGB))

        return float(final_score), isolated_bg_pil, noise_map_pil, noise_level


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze background noise with a smarter, two-step method.")
    parser.add_argument("image_path", type=str, help="Path to the image file to analyze.")
    args = parser.parse_args()

    try:
        pil_image = Image.open(args.image_path)
    except Exception as e:
        print(f"Error opening image: {e}")
        exit()

    print("Initializing V3 scorer and running two-step analysis...")
    scorer = HybridNoiseScorerV3()

    score, isolated_bg, noise_map, raw_noise = scorer.score(pil_image)

    print("\n--------------------------")
    print(f"V3 Background Noise Score: {score:.4f} / 10.0")
    print(f"(Raw Background Noise Level: {raw_noise:.4f})")
    print("--------------------------\n")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Background Noise Analysis (V3 - Color Match Method)", fontsize=16)

    axes[0].imshow(pil_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    axes[1].imshow(isolated_bg)
    axes[1].set_title("Isolated Background (Final)")
    axes[1].axis("off")
    axes[2].imshow(noise_map)
    axes[2].set_title("Noise Map of Final Background")
    axes[2].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
