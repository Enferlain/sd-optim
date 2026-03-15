import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.restoration import denoise_tv_chambolle


# Self-contained single-image "Texture Clean" scorer + visualizer.
# This file intentionally does not import any other repo scripts.


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
) -> tuple[float, dict[str, float]]:
    good_bad = good_bad or TEXTURE_SCORE_DEFAULT_GOOD_BAD
    weights = weights or TEXTURE_SCORE_DEFAULT_WEIGHTS

    comps: dict[str, float] = {}
    total = 0.0
    wsum = 0.0
    for key in TEXTURE_SCORE_METRICS:
        v = float(metrics[key])
        g, b = good_bad[key]
        s = score_lower_better(v, g, b)
        comps[key] = s
        w = float(weights.get(key, 1.0))
        total += s * w
        wsum += w
    avg = (total / wsum) if wsum > 1e-8 else 0.0
    return float(avg * 10.0), comps


def normalize_map(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask is not None and np.any(mask):
        ref = values[mask]
    else:
        ref = values.reshape(-1)
    if ref.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    lo, hi = np.percentile(ref, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(ref))
        hi = float(np.max(ref)) if float(np.max(ref)) > lo else (lo + 1e-6)
    out = (values - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0).astype(np.float32)
    if mask is not None:
        out = out.copy()
        out[~mask] = 0.0
    return out


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


@dataclass
class TextureCleanResult:
    score_0_10: float
    score_raw: float
    metrics: dict[str, float]
    parts: dict[str, float]
    maps: dict[str, np.ndarray]


class TextureCleanScorer:
    def __init__(self, rembg_model: str = "u2net", post_process_mask: bool = True):
        self.rembg_model = rembg_model
        self.post_process_mask = post_process_mask
        self._session: object | None = None

    def _ensure_session(self) -> object:
        if self._session is None:
            from rembg import new_session

            self._session = new_session(self.rembg_model)
        return self._session

    def run(self, pil_image: Image.Image) -> TextureCleanResult:
        # Match image_quality_metrics.py behavior: rembg sees an RGB-converted PIL image.
        img = pil_image.convert("RGB")
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
                "fg_edge_sharpness": 0.0,
                "fg_hf_clean": 0.0,
            }
            score_0_10, parts = texture_clean_score_absolute(metrics)
            return TextureCleanResult(
                score_0_10=score_0_10,
                score_raw=score_0_10 / 10.0,
                metrics=metrics,
                parts=parts,
                maps={"img_rgb": img_rgb, "mask": mask},
            )

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
        edge_sharpness = edge_mean / (ring_mean + 1e-6)
        edge_spread = ring_mean / (edge_mean + 1e-6)

        v_thresh = float(np.percentile(s_lvar, 25)) if s_lvar.size else 0.0
        flat_mask = (lvar <= v_thresh) & mask
        if flat_mask.sum() < 64:
            noise_tv = float(np.std(residual[mask]))
        else:
            noise_tv = float(np.std(residual[flat_mask]))

        lap = cv2.Laplacian(denoised, cv2.CV_32F)
        hf_clean = float(np.mean(np.abs(lap[mask])))

        # --- Texture metrics ---
        v_baseline = float(np.percentile(s_lvar, 20)) if s_lvar.size else 0.0
        starved_mask = (lvar < (v_baseline * 0.4)) & mask if s_lvar.size else mask
        # NOTE: This is intentionally the fraction over the whole image (not conditional on the mask),
        # to match the historical CSV metric definition used elsewhere in this repo.
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
            "fg_edge_sharpness": float(edge_sharpness),
            "fg_hf_clean": float(hf_clean),
        }

        score_0_10, parts = texture_clean_score_absolute(metrics)
        score_raw = score_0_10 / 10.0

        cutout = img_rgb.copy()
        cutout[~mask] = 0

        edge_viz = np.zeros_like(img_rgb)
        edge_viz[edge_mask] = (255, 50, 50)
        edge_viz[ring] = (50, 220, 50)

        maps = {
            "img_rgb": img_rgb,
            "cutout": cutout,
            "mask": mask,
            "starved_mask": starved_mask,
            "flat_mask": flat_mask,
            "residual": residual,
            "edge_viz": edge_viz,
            "mf": mf,
        }
        return TextureCleanResult(
            score_0_10=score_0_10,
            score_raw=score_raw,
            metrics=metrics,
            parts=parts,
            maps=maps,
        )


def plot_result(result: TextureCleanResult) -> None:
    import matplotlib.pyplot as plt

    img = result.maps["img_rgb"]
    cutout = result.maps.get("cutout", img)
    mask = result.maps["mask"]
    starved_mask = result.maps.get("starved_mask", np.zeros(mask.shape, dtype=bool))
    flat_mask = result.maps.get("flat_mask", np.zeros(mask.shape, dtype=bool))
    residual = result.maps.get("residual", np.zeros(mask.shape, dtype=np.float32))
    edge_viz = result.maps.get("edge_viz", np.zeros_like(img))
    mf = result.maps.get("mf", np.zeros(mask.shape, dtype=np.float32))

    flat_resid = residual.copy()
    if flat_mask is not None:
        flat_resid[~flat_mask] = 0.0

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Texture Clean Scorer", fontsize=16)

    ax = axes[0, 0]
    ax.imshow(img)
    ax.set_title("Original")
    ax.axis("off")

    ax = axes[0, 1]
    ax.imshow(cutout)
    ax.set_title("Subject Cutout")
    ax.axis("off")

    ax = axes[0, 2]
    ax.imshow(starved_mask.astype(np.uint8) * 255, cmap="gray", vmin=0, vmax=255)
    ax.set_title("Starved Mask (Binary)")
    ax.axis("off")

    ax = axes[1, 0]
    ax.imshow(normalize_map(flat_resid, mask), cmap="plasma", vmin=0.0, vmax=1.0)
    ax.set_title("Flat-Region Noise (TV)")
    ax.axis("off")

    ax = axes[1, 1]
    ax.imshow(edge_viz)
    ax.set_title("Edge vs Ring (Spread)")
    ax.axis("off")

    ax = axes[1, 2]
    ax.imshow(normalize_map(mf, mask), cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_title("Mid-Frequency Energy")
    ax.axis("off")

    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-file texture-clean scorer with visualization.")
    parser.add_argument("image_path", type=str, help="Path to an image file.")
    parser.add_argument("--rembg-model", type=str, default="u2net")
    parser.add_argument("--no-rembg-post-process", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    img_path = Path(args.image_path)
    img = Image.open(img_path)
    scorer = TextureCleanScorer(
        rembg_model=args.rembg_model,
        post_process_mask=not args.no_rembg_post_process,
    )

    result = scorer.run(img)

    print("\n--- Texture Clean Score ---")
    print(f"Score: {result.score_0_10:.3f} / 10.0  (raw={result.score_raw:.4f})")
    print(f"fg_flat_noise_tv: {result.metrics['fg_flat_noise_tv']:.6f}")
    print(f"fg_noise_tv:      {result.metrics['fg_noise_tv']:.6f}")
    print(f"fg_starved_ratio: {result.metrics['fg_starved_ratio']:.6f}")
    print(f"fg_edge_spread:   {result.metrics['fg_edge_spread']:.6f}")
    print(f"fg_mf_median:     {result.metrics['fg_mf_median']:.6f}")
    print(f"fg_edge_sharpness:{result.metrics['fg_edge_sharpness']:.6f}")
    print(f"fg_hf_clean:      {result.metrics['fg_hf_clean']:.6f}")

    print("\nScore parts (0..1, lower is worse):")
    for key in TEXTURE_SCORE_METRICS:
        print(f"{key} {result.parts.get(key, 0.0):.4f}")

    if not args.no_plot:
        plot_result(result)


if __name__ == "__main__":
    main()
