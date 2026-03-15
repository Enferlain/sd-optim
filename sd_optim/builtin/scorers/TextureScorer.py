import logging

import cv2
import numpy as np
from PIL import Image
from skimage.restoration import denoise_tv_chambolle

logger = logging.getLogger(__name__)

# Texture Clean Scorer ported from test_texture.py

TEXTURE_SCORE_METRICS = [
    "fg_flat_noise_tv",
    "fg_noise_tv",
    "fg_starved_ratio",
    "fg_edge_spread",
    "fg_mf_median",
]

TEXTURE_SCORE_DEFAULT_GOOD_BAD: dict[str, tuple[float, float]] = {
    "fg_flat_noise_tv": (0.0045, 0.0075),
    "fg_noise_tv": (0.0045, 0.0075),
    "fg_starved_ratio": (0.020, 0.050),
    "fg_edge_spread": (0.100, 0.130),
    "fg_mf_median": (0.018, 0.024),
}

TEXTURE_SCORE_DEFAULT_WEIGHTS: dict[str, float] = {
    "fg_flat_noise_tv": 1.0,
    "fg_noise_tv": 1.0,
    "fg_starved_ratio": 2.0,
    "fg_edge_spread": 0.5,
    "fg_mf_median": 0.5,
}


TEXTURE_GRAIN_REFERENCE = 0.0045
TEXTURE_MF_REFERENCE = 0.018


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
    metrics: dict[str, float],
    good_bad: dict[str, tuple[float, float]] | None = None,
    weights: dict[str, float] | None = None,
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


def _compute_texture_metrics(gray: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    denoised = denoise_tv_chambolle(gray, weight=0.10)
    residual = np.abs(gray - denoised)

    mean = cv2.blur(gray, (5, 5))
    mean_sq = cv2.blur(gray * gray, (5, 5))
    lvar = np.maximum(0.0, mean_sq - mean * mean)
    s_lvar = lvar[mask]

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
    edge_spread = ring_mean / (edge_mean + 1e-6)

    v_thresh = float(np.percentile(s_lvar, 25)) if s_lvar.size else 0.0
    flat_mask = (lvar <= v_thresh) & mask
    if flat_mask.sum() < 64:
        noise_tv = float(np.std(residual[mask]))
    else:
        noise_tv = float(np.std(residual[flat_mask]))

    v_baseline = float(np.percentile(s_lvar, 20)) if s_lvar.size else 0.0
    starved_mask = (lvar < (v_baseline * 0.4)) & mask if s_lvar.size else mask
    starved_ratio = float(np.mean(starved_mask[mask])) if mask.any() else 0.0

    flat_mask_tex = (lvar < np.percentile(s_lvar, 25)) & mask if s_lvar.size else mask
    if flat_mask_tex.sum() < 64:
        flat_noise_tv = float(np.std(residual[mask]))
    else:
        flat_noise_tv = float(np.std(residual[flat_mask_tex]))

    gauss1 = cv2.GaussianBlur(gray, (0, 0), 1.0)
    gauss5 = cv2.GaussianBlur(gray, (0, 0), 5.0)
    mf = np.abs(gauss1 - gauss5)
    mf_median = float(np.median(mf[mask])) if mask.any() else 0.0

    return {
        "fg_flat_noise_tv": float(flat_noise_tv),
        "fg_noise_tv": float(noise_tv),
        "fg_starved_ratio": float(starved_ratio),
        "fg_edge_spread": float(edge_spread),
        "fg_mf_median": float(mf_median),
    }


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


def _oversmoothing_penalty(
    metrics: dict[str, float],
    grain_floor_reference: float,
    mf_floor_reference: float,
    grain_floor_weight: float,
    mf_floor_weight: float,
) -> float:
    if grain_floor_reference <= 0.0 or mf_floor_reference <= 0.0:
        return 1.0

    grain_loss = np.clip(
        (grain_floor_reference - metrics["fg_flat_noise_tv"]) / grain_floor_reference,
        0.0,
        1.0,
    )
    mf_loss = np.clip(
        (mf_floor_reference - metrics["fg_mf_median"]) / mf_floor_reference,
        0.0,
        1.0,
    )

    weight_sum = max(grain_floor_weight + mf_floor_weight, 1e-8)
    weighted_loss = (grain_floor_weight * grain_loss + mf_floor_weight * mf_loss) / weight_sum
    return float(np.clip(1.0 - weighted_loss, 0.0, 1.0))


def build_rembg_mask(
    img: Image.Image,
    model: str,
    post_process_mask: bool,
    session: object | None = None,
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
        rembg_session: object | None = None,
        min_mask_coverage: float = 0.02,
        target_mask_coverage: float = 0.15,
        grain_floor_reference: float = TEXTURE_GRAIN_REFERENCE,
        mf_floor_reference: float = TEXTURE_MF_REFERENCE,
        grain_floor_weight: float = 0.4,
        mf_floor_weight: float = 0.6,
    ):
        self.rembg_model = rembg_model
        self.post_process_mask = post_process_mask
        self._session = rembg_session
        self.min_mask_coverage = min_mask_coverage
        self.target_mask_coverage = target_mask_coverage
        self.grain_floor_reference = grain_floor_reference
        self.mf_floor_reference = mf_floor_reference
        self.grain_floor_weight = grain_floor_weight
        self.mf_floor_weight = mf_floor_weight

    def _ensure_session(self) -> object:
        if self._session is None:
            from rembg import new_session

            self._session = new_session(self.rembg_model)
        return self._session

    def _build_mask(self, image: Image.Image, gray: np.ndarray) -> np.ndarray:
        del gray
        return build_rembg_mask(
            image,
            model=self.rembg_model,
            post_process_mask=self.post_process_mask,
            session=self._ensure_session(),
        )

    def score(self, image: Image.Image) -> float:
        img = image.convert("RGB")
        img_rgb = np.array(img)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        mask = self._build_mask(img, gray)

        if mask is None or not np.any(mask):
            logger.warning("TextureScorer: foreground mask is empty; returning 0.0.")
            return 0.0

        mask_coverage = float(np.mean(mask))
        if mask_coverage <= self.min_mask_coverage:
            logger.info(
                "TextureScorer: mask coverage %.4f is below minimum %.4f; returning 0.0.",
                mask_coverage,
                self.min_mask_coverage,
            )
            return 0.0

        metrics = _compute_texture_metrics(gray, mask)
        base_score = texture_clean_score_absolute(metrics)

        anti_oversmoothing = _oversmoothing_penalty(
            metrics=metrics,
            grain_floor_reference=self.grain_floor_reference,
            mf_floor_reference=self.mf_floor_reference,
            grain_floor_weight=self.grain_floor_weight,
            mf_floor_weight=self.mf_floor_weight,
        )
        coverage_factor = _coverage_confidence(
            coverage=mask_coverage,
            min_coverage=self.min_mask_coverage,
            target_coverage=self.target_mask_coverage,
        )

        adjusted_score = base_score * anti_oversmoothing * coverage_factor
        return float(np.clip(adjusted_score, 0.0, 10.0))


class TextureScorerFullImage(TextureScorer):
    def __init__(
        self,
        grain_floor_reference: float = TEXTURE_GRAIN_REFERENCE,
        mf_floor_reference: float = TEXTURE_MF_REFERENCE,
        grain_floor_weight: float = 0.4,
        mf_floor_weight: float = 0.6,
    ):
        super().__init__(
            rembg_model="u2net",
            post_process_mask=True,
            rembg_session=None,
            min_mask_coverage=0.0,
            target_mask_coverage=1.0,
            grain_floor_reference=grain_floor_reference,
            mf_floor_reference=mf_floor_reference,
            grain_floor_weight=grain_floor_weight,
            mf_floor_weight=mf_floor_weight,
        )

    def _build_mask(self, image: Image.Image, gray: np.ndarray) -> np.ndarray:
        del image
        return np.ones(gray.shape, dtype=bool)
