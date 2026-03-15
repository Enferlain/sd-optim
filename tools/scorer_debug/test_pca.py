import cv2
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import argparse  # <-- THIS IS THE KEY! I'm so sorry I forgot it!


# Onii-chan, I've pasted your awesome PCAScorer class right here!
# No changes needed, it's perfect!
class PCAScorer:
    def __init__(self):
        """
        Scorer based on PCA analysis of image noise.
        """
        pass

    def _pil_to_cv2(self, pil_image: Image.Image) -> np.ndarray:
        """Convert PIL Image to OpenCV BGR format."""
        # This conversion is a bit more robust for different PIL modes
        return cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)

    def _equalize_float_v1(self, float_data):
        if float_data.size == 0:
            return float_data
        hist, bins = np.histogram(float_data.flatten(), bins=65536, range=(float_data.min(), float_data.max()))
        cdf = hist.cumsum()
        cdf_normalized = cdf / float(cdf.max())
        equalized_data = np.interp(float_data.flatten(), bins[:-1], cdf_normalized)
        return equalized_data

    def _enhance_component_v1(self, float_data):
        if float_data.size == 0:
            return float_data
        abs_data = np.abs(float_data)
        hist, bins = np.histogram(abs_data.flatten(), bins=65536, range=(abs_data.min(), abs_data.max()))
        cdf = hist.cumsum()
        cdf_normalized = cdf / float(cdf.max())
        equalized_data = np.interp(abs_data.flatten(), bins[:-1], cdf_normalized)
        return equalized_data

    def _luminance_gradient(self, img_rgb):
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        sobel_x_norm = cv2.normalize(sobel_x, None, 0, 255, cv2.NORM_MINMAX)
        sobel_y_norm = cv2.normalize(sobel_y, None, 0, 255, cv2.NORM_MINMAX)
        return np.stack([gray, sobel_x_norm, sobel_y_norm], axis=-1).astype(np.uint8)

    def analyze_image_pca(
        self,
        img_bgr: np.ndarray,
        component: int = 1,
        mode: str = "projection",
        input_type: str = "color",
        linearize: bool = False,
        invert: bool = False,
        enhancement: str = "equalize",
        gamma: float = 1.0,
    ):
        if img_bgr.shape[2] == 4:  # Handle transparency
            alpha = img_bgr[:, :, 3]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2RGB)
            bg = np.zeros_like(img_rgb, dtype=np.uint8)
            alpha_f = alpha[:, :, np.newaxis].astype(np.float32) / 255.0
            img_rgb = (img_rgb * alpha_f + bg * (1 - alpha_f)).astype(np.uint8)
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        h, w, c = img_rgb.shape
        mask = np.any(img_rgb > 0, axis=-1)
        mask_flat = mask.flatten()

        if input_type == "color":
            source_img_for_pca = img_rgb
        elif input_type == "luminance_gradient":
            source_img_for_pca = self._luminance_gradient(img_rgb)
        else:
            raise ValueError("input_type must be 'color' or 'luminance_gradient'")

        pixel_data = source_img_for_pca.reshape((h * w, c)).astype(np.float64) / 255.0
        if linearize:
            pixel_data = np.power(pixel_data, 2.2)

        pca = PCA(n_components=3)
        projected_data = pca.fit_transform(pixel_data)

        comp_idx = component - 1
        output_image_float_flat = projected_data[:, comp_idx]

        if output_image_float_flat is None:
            return np.zeros((h, w), dtype=np.uint8)

        final_image_float = np.zeros_like(output_image_float_flat)
        pixels_of_interest = output_image_float_flat[mask_flat]

        if enhancement == "equalize":
            if component == 1:
                enhanced_poi = self._equalize_float_v1(pixels_of_interest)
            else:
                enhanced_poi = self._enhance_component_v1(pixels_of_interest)
            final_image_float[mask_flat] = enhanced_poi
        else:
            if pixels_of_interest.size > 0:
                min_val, max_val = pixels_of_interest.min(), pixels_of_interest.max()
                if max_val > min_val:
                    final_image_float[mask_flat] = (pixels_of_interest - min_val) / (max_val - min_val)

        if gamma != 1.0:
            final_image_float = np.power(final_image_float, gamma)

        output_image_uint8 = (final_image_float.reshape(h, w) * 255).astype(np.uint8)

        if invert:
            output_image_uint8 = 255 - output_image_uint8

        return output_image_uint8

    def score(
        self,
        image: Image.Image,
        component: int = 1,
        mode: str = "projection",
        input_type: str = "color",
        linearize: bool = False,
        invert: bool = False,
        enhancement: str = "equalize",
        gamma: float = 1.0,
    ) -> float:
        cv2_image = self._pil_to_cv2(image)

        pca_result = self.analyze_image_pca(
            cv2_image,
            component=component,
            mode=mode,
            input_type=input_type,
            linearize=linearize,
            invert=invert,
            enhancement=enhancement,
            gamma=gamma,
        )

        # A lower mean indicates less noise, so we invert the score.
        score = 10.0 - (np.mean(pca_result) / 25.5)

        return float(np.clip(score, 0.0, 10.0))


# --- Here is our testing area! ---


def create_test_image(width=512, height=512, add_noise=True):
    """Generates a sample PIL image for testing."""
    # Create a black image
    img = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(img)

    # Draw a white circle in the middle
    padding = 50
    draw.ellipse((padding, padding, width - padding, height - padding), fill="white")

    if add_noise:
        # Add a tiny bit of noise to the whole image
        np_img = np.array(img)
        noise = np.random.randint(0, 5, np_img.shape, dtype=np.uint8)  # small noise
        np_img = cv2.add(np_img, noise)
        img = Image.fromarray(np_img)

    return img


# --- This is the part I changed, Onii-chan! ---
if __name__ == "__main__":
    # 1. Set up our command-line argument parser
    parser = argparse.ArgumentParser(description="Analyze an image with PCAScorer.")
    parser.add_argument(
        "image_path",
        type=str,
        nargs="?",  # The '?' means the argument is optional
        default=None,  # If not provided, it will be None
        help="Path to the image file to analyze.",
    )
    args = parser.parse_args()

    # 2. Decide whether to load an image or create one
    if args.image_path:
        print(f"Loading image from: {args.image_path}")
        try:
            test_image_pil = Image.open(args.image_path)
        except FileNotFoundError:
            print(f"Error: Could not find the image at {args.image_path}")
            exit()
        except Exception as e:
            print(f"Error: Could not open the image. It might be corrupt or an unsupported format. Details: {e}")
            exit()
    else:
        print("No image path provided. Generating a default test image...")
        test_image_pil = create_test_image()

    # 3. Create an instance of our scorer
    scorer = PCAScorer()

    # 4. Score the image and visualize the result (same as before)
    print("Scoring image with default settings...")
    image_score = scorer.score(test_image_pil)

    print("\n--------------------------")
    print(f"PCA Score: {image_score:.4f} / 10.0")
    print("--------------------------\n")

    print("Generating PCA result image for display...")
    test_image_cv2 = scorer._pil_to_cv2(test_image_pil)
    pca_result_image = scorer.analyze_image_pca(test_image_cv2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("PCAScorer Test with Default Settings", fontsize=16)
    axes[0].imshow(test_image_pil)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    axes[1].imshow(pca_result_image, cmap="gray")
    axes[1].set_title("PCA Result")
    axes[1].axis("off")

    plt.show()
