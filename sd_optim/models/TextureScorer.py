import cv2
import numpy as np
from PIL import Image
from skimage.restoration import denoise_tv_chambolle
from typing import Optional, Dict, Tuple

# Texture Clean Scorer ported from test_texture.py

TEXTURE_SCORE_METRICS = [
    "fg_flat_noise_tv",
    "fg_noise_tv",
    "fg_starved_ratio",
    "fg_edge_spread",
    "fg_mf_median",
]

TEXTURE_SCORE_DEFAULT_GOOD_BAD: Dict[str, Tuple[float, float]] = {
    "fg_flat_noise_tv": (0.0045, 0.0075),
    "fg_noise_tv": (0.0045, 0.0075),
    "fg_starved_ratio": (0.020, 0.050),
    "fg_edge_spread": (0.100, 0.130),
    "fg_mf_median": (0.018, 0.024),
}

TEXTURE_SCORE_DEFAULT_WEIGHTS: Dict[str, float] = {
    "fg_flat_noise_tv": 1.0,
    "fg_noise_tv": 1.0,
    "fg_starved_ratio": 2.0,
    "fg_edge_spread": 0.5,
    "fg_mf_median": 0.5,
}


def score_lower_better(value: float, good: float, bad: float) -> float:
    if bad <= good:
        return 0.0
    t = (value - good) / (bad - good)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return float(1.0 - t)


def texture_clean_score_absolute(
    metrics: Dict[str, float],
    good_bad: Optional[Dict[str, Tuple[float, float]]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    good_bad = good_bad or TEXTURE_SCORE_DEFAULT_GOOD_BAD
    weights = weights or TEXTURE_SCORE_DEFAULT_WEIGHTS

    total = 0.0
    wsum = 0.0
    for key in TEXTURE_SCORE_METRICS:
        v = float(metrics[key])
        g, b = good_bad[key]
        s = score_lower_better(v, g, b)
        w = float(weights.get(key, 1.0))
        total += s * w
        wsum += w
    avg = (total / wsum) if wsum > 1e-8 else 0.0
    return float(avg * 10.0)


def build_rembg_mask(
    img: Image.Image,
    model: str,
    post_process_mask: bool,
    session: Optional[object] = None,
) -> np.ndarray:
    from rembg import new_session
    from rembg.bg import post_process

    if session is None:
        session = new_session(model)

    # In certain rembg versions, session.predict might behave differently.
    # test_texture.py implementation:
    masks = session.predict(img)
    if not masks:
        return np.zeros((img.height, img.width), dtype=bool)

    merged = None
    for mask in masks:
        arr = np.array(mask)
        if post_process_mask:
            arr = post_process(arr)
        if merged is None:
            merged = arr.astype(np.uint8)
        else:
            merged = np.maximum(merged, arr.astype(np.uint8))
    return merged >= 128


class TextureScorer:
    def __init__(
        self,
        rembg_model: str = "u2net",
        post_process_mask: bool = True,
        rembg_session: Optional[object] = None,
    ):
        self.rembg_model = rembg_model
        self.post_process_mask = post_process_mask
        self._session = rembg_session

    def _ensure_session(self) -> object:
        if self._session is None:
            from rembg import new_session

            self._session = new_session(self.rembg_model)
        return self._session

    def score(self, image: Image.Image) -> float:
        img = image.convert("RGB")
        img_rgb = np.array(img)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        mask = build_rembg_mask(
            img,
            model=self.rembg_model,
            post_process_mask=self.post_process_mask,
            session=self._ensure_session(),
        )

        if mask is None or not np.any(mask):
            metrics = {
                "fg_flat_noise_tv": 0.0,
                "fg_noise_tv": 0.0,
                "fg_starved_ratio": 0.0,
                "fg_edge_spread": 0.0,
                "fg_mf_median": 0.0,
            }
            return float(texture_clean_score_absolute(metrics))

        denoised = denoise_tv_chambolle(gray, weight=0.10)
        residual = np.abs(gray - denoised)

        mean = cv2.blur(gray, (5, 5))
        mean_sq = cv2.blur(gray * gray, (5, 5))
        lvar = np.maximum(0.0, mean_sq - mean * mean)
        s_lvar = lvar[mask]

        # --- Noise-robust edge/noise metrics ---
        gx = cv2.Sobel(denoised, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(denoised, cv2.CV_32F, 0, 1, ksize=3)
        gmag = np.sqrt(gx * gx + gy * gy)
        gvals = gmag[mask]

        if gvals.size >= 64:
            thresh = float(np.percentile(gvals, 90))
            edge_mask = (gmag >= thresh) & mask
            if edge_mask.sum() < 16:
                edge_mask = (gmag >= np.percentile(gvals, 80)) & mask
        else:
            edge_mask = np.zeros_like(mask)

        kernel = np.ones((3, 3), np.uint8)
        dil = cv2.dilate(edge_mask.astype(np.uint8), kernel, iterations=2) > 0
        ring = dil & (~edge_mask) & mask
        if ring.sum() < 16:
            ring = mask & (~edge_mask)

        edge_mean = float(np.mean(gmag[edge_mask])) if edge_mask.sum() else 0.0
        ring_mean = float(np.mean(gmag[ring])) if ring.sum() else 0.0
        # edge_sharpness = edge_mean / (ring_mean + 1e-6) # Not used in final score calculation but in metrics dict
        edge_spread = ring_mean / (edge_mean + 1e-6)

        v_thresh = float(np.percentile(s_lvar, 25)) if s_lvar.size else 0.0
        flat_mask = (lvar <= v_thresh) & mask
        if flat_mask.sum() < 64:
            noise_tv = float(np.std(residual[mask]))
        else:
            noise_tv = float(np.std(residual[flat_mask]))

        # lap = cv2.Laplacian(denoised, cv2.CV_32F)
        # hf_clean = float(np.mean(np.abs(lap[mask]))) # Not used in absolute score

        # --- Texture metrics ---
        v_baseline = float(np.percentile(s_lvar, 20)) if s_lvar.size else 0.0
        starved_mask = (lvar < (v_baseline * 0.4)) & mask if s_lvar.size else mask
        starved_ratio = float(np.mean(starved_mask)) if mask.any() else 0.0

        flat_mask_tex = (lvar < np.percentile(s_lvar, 25)) & mask
        if flat_mask_tex.sum() < 64:
            flat_noise_tv = float(np.std(residual[mask]))
        else:
            flat_noise_tv = float(np.std(residual[flat_mask_tex]))

        gauss1 = cv2.GaussianBlur(gray, (0, 0), 1.0)
        gauss5 = cv2.GaussianBlur(gray, (0, 0), 5.0)
        mf = np.abs(gauss1 - gauss5)
        mf_median = float(np.median(mf[mask])) if mask.any() else 0.0

        metrics = {
            "fg_flat_noise_tv": float(flat_noise_tv),
            "fg_noise_tv": float(noise_tv),
            "fg_starved_ratio": float(starved_ratio),
            "fg_edge_spread": float(edge_spread),
            "fg_mf_median": float(mf_median),
        }

        return float(texture_clean_score_absolute(metrics))
