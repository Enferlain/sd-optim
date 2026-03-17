import numpy as np
from PIL import Image
import argparse
import matplotlib.pyplot as plt


# --- Your GammaNoiseScorer Class ---
# Onii-chan, I made a tiny change here to the score() method!
# It now also returns the gamma_image so we can plot it.
# I marked the change with #<--
class GammaNoiseScorer:
    def __init__(self, gamma_value: float = 0.3, noise_threshold: float = 10.0):
        self.gamma_value = gamma_value
        self.noise_threshold = noise_threshold

    def _gamma_noise_reveal(self, image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        np_img = np.array(image, dtype=np.float32)
        gamma_corrected = np.power(np_img / 255.0, self.gamma_value) * 255
        return Image.fromarray(gamma_corrected.astype(np.uint8))

    def score(self, image: Image.Image) -> tuple[float, Image.Image]:  # <-- Changed return type
        gamma_image = self._gamma_noise_reveal(image)
        gamma_array = np.array(gamma_image, dtype=np.float32)
        noise_level = np.std(gamma_array)
        noise_score = 10.0 * (1.0 - min(noise_level / self.noise_threshold, 1.0))

        final_score = np.clip(noise_score, 0.0, 10.0)
        return float(final_score), gamma_image  # <-- Return the image for our plot!


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze image noise revealed by gamma correction.")
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

    print("Initializing scorer and applying gamma correction...")
    scorer = GammaNoiseScorer()

    # Get the score and the gamma image for our plot
    score, gamma_result_image = scorer.score(pil_image)

    print("\n--------------------------")
    print(f"Gamma Noise Score: {score:.4f} / 10.0")
    print("--------------------------\n")

    # --- Visualization ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Gamma Noise Analysis", fontsize=16)

    # 1. Original Image
    axes[0].imshow(pil_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # 2. Gamma Corrected Image
    axes[1].imshow(gamma_result_image)
    axes[1].set_title(f"Gamma Corrected (γ={scorer.gamma_value})")
    axes[1].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
