import numpy as np
from PIL import Image
import io
import rembg
from sklearn.cluster import KMeans
import joblib
import logging
import argparse
import matplotlib.pyplot as plt


# --- Your BackgroundBlacknessScorer Class ---
# Onii-chan, I've made one tiny change! I made the score() method also
# return the dominant color and the bg-removed image so we can plot them.
# I marked the changes with #<--
class BackgroundBlacknessScorer:
    def __init__(self, rembg_session=None, n_clusters: int = 4, sensitivity: float = 0.008):
        if rembg_session is None:
            self.rembg_session = rembg.new_session(providers=["CPUExecutionProvider"])
        else:
            self.rembg_session = rembg_session
        self.n_clusters = n_clusters
        self.sensitivity = sensitivity

    def score(self, image: Image.Image) -> tuple[float, np.ndarray, Image.Image]:  # <-- Changed return type
        if self.rembg_session is None:
            return 0.0, np.array([255, 0, 255]), image  # <-- Return pink for error

        target_black = np.array([0, 0, 0])

        try:
            original_image = image.convert("RGB")
            original_pixels = np.array(original_image)

            buffer = io.BytesIO()
            original_image.save(buffer, format="PNG")
            input_bytes = buffer.getvalue()

            output_bytes = rembg.remove(input_bytes, session=self.rembg_session)

            output_image_with_alpha = Image.open(io.BytesIO(output_bytes))
            output_pixels = np.array(output_image_with_alpha)

            if output_pixels.shape[2] != 4:
                return 0.0, np.array([255, 0, 255]), output_image_with_alpha

            alpha_channel = output_pixels[:, :, 3]
            background_mask = alpha_channel < 128  # Use a threshold for semi-transparent edges

            if not np.any(background_mask):
                return 10.0, np.array([0, 0, 0]), output_image_with_alpha

            original_background_pixels = original_pixels[background_mask]

            if len(original_background_pixels) < self.n_clusters:
                dominant_color = np.mean(original_background_pixels, axis=0)
            else:
                kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init="auto")
                with joblib.parallel_backend("threading", n_jobs=1):
                    kmeans.fit(original_background_pixels)

                unique, counts = np.unique(kmeans.labels_, return_counts=True)
                dominant_cluster_index = unique[counts.argmax()]
                dominant_color = kmeans.cluster_centers_[dominant_cluster_index]

            distance = np.linalg.norm(dominant_color - target_black)
            score_0_to_1 = np.exp(-self.sensitivity * distance)
            final_score = score_0_to_1 * 10.0

            return float(np.clip(final_score, 0.0, 10.0)), dominant_color, output_image_with_alpha  # <-- Return our new values!

        except Exception as e:
            logging.error(f"An error occurred during blackness scoring: {e}", exc_info=True)
            return 0.0, np.array([255, 0, 255]), image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze the background blackness of an image.")
    parser.add_argument("image_path", type=str, help="Path to the image file to analyze.")
    args = parser.parse_args()

    try:
        pil_image = Image.open(args.image_path)
    except FileNotFoundError:
        print(f"Error: Could not find the image at {args.image_path}")
        exit()
    except Exception as e:
        print(f"Error: Could not open the image. Details: {e}")
        exit()

    print("Initializing scorer and removing background (this might take a moment)...")
    scorer = BackgroundBlacknessScorer()

    # Get the score and the extra data for our plot
    score, dominant_color, bg_removed_image = scorer.score(pil_image)

    dominant_color_int = dominant_color.astype(int)

    print("\n--------------------------")
    print(f"Background Blackness Score: {score:.4f} / 10.0")
    print(f"Detected Dominant BG Color (RGB): ({dominant_color_int[0]}, {dominant_color_int[1]}, {dominant_color_int[2]})")
    print("--------------------------\n")

    # --- Visualization ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Background Blackness Analysis", fontsize=16)

    # 1. Original Image
    axes[0].imshow(pil_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # 2. Background Removed
    axes[1].imshow(bg_removed_image)
    axes[1].set_title("AI Background Removal")
    axes[1].axis("off")

    # 3. Dominant Color Swatch
    color_swatch = np.zeros((100, 100, 3), dtype=np.uint8)
    color_swatch[:, :] = dominant_color_int
    axes[2].imshow(color_swatch)
    axes[2].set_title(f"Dominant BG Color\nRGB: {tuple(dominant_color_int)}")
    axes[2].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
