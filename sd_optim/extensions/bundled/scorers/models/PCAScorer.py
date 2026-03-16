import cv2
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image


class PCAScorer:
    def __init__(self):
        """
        Scorer based on PCA analysis of image noise.
        """
        pass

    def _pil_to_cv2(self, pil_image: Image.Image) -> np.ndarray:
        """Convert PIL Image to OpenCV RGB format."""
        # Convert PIL image to numpy array
        numpy_image = np.array(pil_image)
        if pil_image.mode == "RGBA":
            # Convert RGBA to BGR
            return cv2.cvtColor(numpy_image, cv2.COLOR_RGBA2BGR)
        else:
            # Convert RGB to BGR
            return cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

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

        # Create destination arrays for normalization
        sobel_x_norm = np.empty_like(sobel_x)
        sobel_y_norm = np.empty_like(sobel_y)

        cv2.normalize(sobel_x, dst=sobel_x_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(sobel_y, dst=sobel_y_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
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
        output_image_float_flat = None

        if mode == "projection":
            output_image_float_flat = projected_data[:, comp_idx]
        else:
            reconstruction_data = np.zeros_like(projected_data)
            reconstruction_data[:, comp_idx] = projected_data[:, comp_idx]
            reconstructed_pixels = pca.inverse_transform(reconstruction_data)

            if mode == "component":
                reconstructed_img = reconstructed_pixels.reshape((h, w, c))
                reconstructed_img = np.clip(reconstructed_img, 0, 1)
                if linearize:
                    reconstructed_img = np.power(reconstructed_img, 1 / 2.2)
                reconstructed_uint8 = (reconstructed_img * 255).astype(np.uint8)
                output_image_float_flat = (cv2.cvtColor(reconstructed_uint8, cv2.COLOR_RGB2GRAY) / 255.0).flatten()
            else:
                difference = pixel_data - reconstructed_pixels
                if linearize:
                    difference = np.power(np.abs(difference), 1 / 2.2) * np.sign(difference)

                if mode == "difference":
                    output_image_float_flat = np.mean(difference, axis=1)
                elif mode == "distance":
                    output_image_float_flat = np.linalg.norm(difference, axis=1)

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

        # Analyze the image using PCA with the provided settings
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

        # Calculate a score based on the PCA result.
        # A lower mean indicates less noise, so we invert the score.
        # (10.0 - normalized_mean)
        score = 10.0 - (np.mean(pca_result) / 25.5)

        return float(np.clip(score, 0.0, 10.0))
