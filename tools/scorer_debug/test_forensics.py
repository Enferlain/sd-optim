import cv2
import numpy as np
from PIL import Image
import rembg
import argparse
import matplotlib.pyplot as plt


# --- Your ForensicNoiseScorer Class ---
# Onii-chan, I made one small change here! I made the score() method also
# return the background pixels and noise level so we can make our cool plot.
# I marked the changes with #<--
class ForensicNoiseScorer:
    def __init__(self, noise_threshold: float = 5.0, rembg_session=None):
        self.noise_threshold = noise_threshold
        if rembg_session is None:
            self.rembg_session = rembg.new_session(providers=["CPUExecutionProvider"])
        else:
            self.rembg_session = rembg_session

    def _pil_to_cv2(self, pil_image: Image.Image) -> np.ndarray:
        return cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)

    def _isolate_background(self, cv2_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  # <--
        _, buffer = cv2.imencode(".png", cv2_image)
        input_bytes = buffer.tobytes()
        output_bytes = rembg.remove(input_bytes, session=self.rembg_session)
        output_image_with_alpha = cv2.imdecode(np.frombuffer(output_bytes, np.uint8), cv2.IMREAD_UNCHANGED)

        if output_image_with_alpha.shape[2] == 4:
            alpha_channel = output_image_with_alpha[:, :, 3]
            background_mask = alpha_channel < 128  # Use a threshold
            background_pixels = cv2_image[background_mask]
            return background_pixels, background_mask  # <-- Return mask too!
        else:
            return np.array([]), np.array([])

    def _detect_structural_noise(self, pixels: np.ndarray) -> float:
        if pixels.size == 0:
            return 0.0
        if len(pixels.shape) > 1 and pixels.shape[1] == 3:
            gray_pixels = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY)
        else:
            gray_pixels = pixels
        return np.std(gray_pixels)

    def score(self, image: Image.Image) -> tuple[float, np.ndarray, float]:  # <-- Changed return type
        cv2_image = self._pil_to_cv2(image)
        background_pixels, background_mask = self._isolate_background(cv2_image)  # <-- Get mask

        if background_pixels.size < 10:  # Need some pixels to score
            return 5.0, np.array([]), 0.0

        noise_level = self._detect_structural_noise(background_pixels)
        final_score = 10.0 * (1.0 - min(noise_level / self.noise_threshold, 1.0))

        # We need to return the mask and noise level for our plot!
        return float(np.clip(final_score, 0.0, 10.0)), background_mask, noise_level  # <--


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze the background noise of an image.")
    parser.add_argument("image_path", type=str, help="Path to the image file to analyze.")
    args = parser.parse_args()

    try:
        pil_image = Image.open(args.image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Could not find the image at {args.image_path}")
        exit()
    except Exception as e:
        print(f"Error: Could not open the image. Details: {e}")
        exit()

    print("Initializing scorer and removing background (this might take a moment)...")
    scorer = ForensicNoiseScorer()

    # Get the score and the extra data for our plot
    score, background_mask, noise_level = scorer.score(pil_image)

    print("\n--------------------------")
    print(f"Forensic Noise Score: {score:.4f} / 10.0")
    print(f"(Raw Noise Level (Std Dev): {noise_level:.4f})")
    print("--------------------------\n")

    # --- Visualization ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Forensic Noise Analysis", fontsize=16)

    # 1. Original Image
    axes[0].imshow(pil_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # 2. Isolated Background
    original_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    isolated_bg_image = np.zeros_like(original_cv_image)
    if background_mask.size > 0:
        isolated_bg_image[background_mask] = original_cv_image[background_mask]

    axes[1].imshow(cv2.cvtColor(isolated_bg_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Isolated Background Pixels")
    axes[1].axis("off")

    # 3. Histogram of Background Pixels
    if background_mask.size > 0:
        bg_pixels_gray = cv2.cvtColor(original_cv_image[background_mask].reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY)
        axes[2].hist(bg_pixels_gray.flatten(), bins=50, color="gray", range=[0, 255])
        axes[2].set_title("Background Brightness Distribution")
        axes[2].set_xlabel("Pixel Brightness (0=Black, 255=White)")
        axes[2].set_ylabel("Pixel Count")
        axes[2].set_xlim(0, 255)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
