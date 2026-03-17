"""
test_artifact.py  –  Artifact-focused scorer for SD-generated images.

Detects:
  1. JPEG 8×8 block boundary strength  (fg_blockiness)
  2. 8-pixel periodicity in TV residual  (fg_dct_period)
  3. Ringing around edges  (fg_ringing)
  4. Loss of expected SD grain in flat regions  (fg_grain_loss)  ← inverted from original
  5. Mid-frequency energy loss  (fg_mf_loss)               ← inverted from original

Score: 0..10, higher = cleaner (fewer artifacts).
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.restoration import denoise_tv_chambolle
import matplotlib.pyplot as plt

# ── Calibration ─────────────────────────────────────────────────────────────
# (good, bad): metric value at which partial score = 1.0 vs 0.0

ARTIFACT_METRICS = [
    "fg_blockiness",
    "fg_dct_period",
    "fg_ringing",
    "fg_grain_loss",
    "fg_mf_loss",
]

ARTIFACT_GOOD_BAD: dict[str, tuple[float, float]] = {
    "fg_blockiness": (1.05, 1.40),   # ratio of block-boundary grad vs interior
    "fg_dct_period": (0.05, 0.20),   # autocorr peak at lag=8 relative to lag=0
    "fg_ringing":    (0.08, 0.18),   # ring_mean / edge_mean
    "fg_grain_loss": (0.00, 1.00),   # fraction of expected SD grain that's missing
    "fg_mf_loss":    (0.00, 1.00),   # fraction of expected MF energy that's missing
}

ARTIFACT_WEIGHTS: dict[str, float] = {
    "fg_blockiness": 2.0,   # strongest JPEG signal
    "fg_dct_period": 2.0,   # strongest JPEG signal
    "fg_ringing":    1.0,
    "fg_grain_loss": 1.5,   # important inversion of original scorer flaw
    "fg_mf_loss":    1.0,
}

# SD-clean reference floors (calibrated from PNG baseline terminal output)
SD_GRAIN_REFERENCE = 0.0045   # expected min flat-region TV residual for SD images
SD_MF_REFERENCE    = 0.018    # expected min median mid-frequency energy for SD images


# ── Scoring helpers ──────────────────────────────────────────────────────────

def score_lower_better(value: float, good: float, bad: float) -> float:
    if bad <= good:
        return 0.0
    t = float(np.clip((value - good) / (bad - good), 0.0, 1.0))
    return 1.0 - t


def artifact_score(
    metrics: dict[str, float],
    good_bad: dict | None = None,
    weights: dict | None = None,
) -> tuple[float, dict[str, float]]:
    good_bad = good_bad or ARTIFACT_GOOD_BAD
    weights  = weights  or ARTIFACT_WEIGHTS
    parts: dict[str, float] = {}
    total = wsum = 0.0
    for key in ARTIFACT_METRICS:
        v = float(metrics[key])
        g, b = good_bad[key]
        s = score_lower_better(v, g, b)
        parts[key] = s
        w = float(weights.get(key, 1.0))
        total += s * w
        wsum  += w
    avg = (total / wsum) if wsum > 1e-8 else 0.0
    return float(avg * 10.0), parts


# ── Mask ─────────────────────────────────────────────────────────────────────

def build_rembg_mask(
    img: Image.Image,
    model: str = "u2net",
    post_process_mask: bool = True,
    session=None,
) -> np.ndarray:
    from rembg import new_session
    from rembg.bg import post_process
    if session is None:
        session = new_session(model)
    masks = session.predict(img)
    if not masks:
        return np.zeros((img.height, img.width), dtype=bool)
    merged = None
    for m in masks:
        arr = np.array(m)
        if post_process_mask:
            arr = post_process(arr)
        merged = arr.astype(np.uint8) if merged is None else np.maximum(merged, arr.astype(np.uint8))
    return merged >= 128


# ── Per-metric computation ───────────────────────────────────────────────────

def compute_blockiness(gray: np.ndarray, mask: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Ratio of gradient magnitude at 8-pixel-aligned boundaries vs interior.
    Standard JPEG blocks are always aligned to (0,0), so this is reliable.
    ratio > 1.0  →  block boundaries are stronger than the surroundings  →  JPEG artifact.
    """
    dx = np.abs(np.diff(gray, axis=1, append=gray[:, -1:]))
    dy = np.abs(np.diff(gray, axis=0, append=gray[-1:, :]))
    grad = (dx + dy) * 0.5

    H, W = gray.shape
    block_map = np.zeros((H, W), dtype=bool)
    block_map[:, 7::8] = True   # vertical block seams
    block_map[7::8, :] = True   # horizontal block seams

    fg_block    = block_map & mask
    fg_nonblock = (~block_map) & mask

    if fg_block.sum() < 16 or fg_nonblock.sum() < 16:
        return 1.0, grad

    ratio = float(np.mean(grad[fg_block])) / (float(np.mean(grad[fg_nonblock])) + 1e-6)

    # Viz: block seams in FG, normalized
    viz = np.zeros_like(gray)
    viz[fg_block] = grad[fg_block]
    return ratio, viz


def compute_dct_periodicity(residual: np.ndarray, mask: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Autocorrelation of the TV residual along rows and columns.
    Clean SD images have white-noise residuals → flat autocorrelation.
    JPEG images have 8px block structure → peaks at lag=8, 16, 24.
    """
    r = residual.copy()
    r[~mask] = 0.0

    def _period_strength(signal_1d: np.ndarray) -> float:
        ac = np.correlate(signal_1d, signal_1d, mode="full")
        center = len(ac) // 2
        ac0 = ac[center]
        if ac0 < 1e-12:
            return 0.0
        peaks = [abs(ac[center + lag]) for lag in (8, 16, 24) if center + lag < len(ac)]
        return float(max(peaks) / (ac0 + 1e-6)) if peaks else 0.0

    # Only use rows/cols with enough FG pixels to avoid noise
    row_strengths = [
        _period_strength(r[row])
        for row in range(r.shape[0])
        if mask[row].sum() > r.shape[1] * 0.3
    ]
    col_strengths = [
        _period_strength(r[:, col])
        for col in range(r.shape[1])
        if mask[:, col].sum() > r.shape[0] * 0.3
    ]

    all_strengths = row_strengths + col_strengths
    period_strength = float(np.median(all_strengths)) if all_strengths else 0.0
    return period_strength, r


def compute_ringing(denoised: np.ndarray, mask: np.ndarray) -> tuple[float, np.ndarray]:
    """ring_mean / edge_mean — same logic as fg_edge_spread in the original scorer."""
    gx = cv2.Sobel(denoised, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(denoised, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.sqrt(gx * gx + gy * gy)
    gvals = gmag[mask]
    if gvals.size < 64:
        return 0.0, np.zeros((*gmag.shape, 3), dtype=np.uint8)

    thresh = float(np.percentile(gvals, 90))
    edge_mask = (gmag >= thresh) & mask
    if edge_mask.sum() < 16:
        edge_mask = (gmag >= np.percentile(gvals, 80)) & mask

    kernel = np.ones((3, 3), np.uint8)
    dil = cv2.dilate(edge_mask.astype(np.uint8), kernel, iterations=2) > 0
    ring = dil & (~edge_mask) & mask
    if ring.sum() < 16:
        ring = mask & (~edge_mask)

    edge_mean = float(np.mean(gmag[edge_mask])) if edge_mask.sum() else 0.0
    ring_mean = float(np.mean(gmag[ring]))       if ring.sum()      else 0.0
    edge_spread = ring_mean / (edge_mean + 1e-6)

    viz = np.zeros((*gmag.shape, 3), dtype=np.uint8)
    viz[edge_mask] = (255, 50, 50)
    viz[ring]      = (50, 220, 50)
    return edge_spread, viz


def compute_grain_loss(
    residual: np.ndarray, lvar: np.ndarray, mask: np.ndarray
) -> tuple[float, np.ndarray]:
    """
    KEY FIX vs original scorer:
    Instead of rewarding low flat-region noise (which rewards JPEG compression),
    we penalize when flat-region residual drops BELOW the expected SD grain floor.
    grain_loss = 0.0  →  normal SD grain present
    grain_loss = 1.0  →  grain completely absent (heavily compressed / over-smoothed)
    """
    s_lvar = lvar[mask]
    v_thresh = float(np.percentile(s_lvar, 25)) if s_lvar.size else 0.0
    flat_mask = (lvar <= v_thresh) & mask
    if flat_mask.sum() < 64:
        actual = float(np.std(residual[mask]))
    else:
        actual = float(np.std(residual[flat_mask]))

    loss = float(np.clip((SD_GRAIN_REFERENCE - actual) / SD_GRAIN_REFERENCE, 0.0, 1.0))

    viz = residual.copy()
    viz[~flat_mask] = 0.0
    return loss, viz


def compute_mf_loss(gray: np.ndarray, mask: np.ndarray) -> tuple[float, np.ndarray]:
    """Penalize loss of mid-frequency energy below the SD reference floor."""
    gauss1 = cv2.GaussianBlur(gray, (0, 0), 1.0)
    gauss5 = cv2.GaussianBlur(gray, (0, 0), 5.0)
    mf = np.abs(gauss1 - gauss5)
    mf_median = float(np.median(mf[mask])) if mask.any() else 0.0
    loss = float(np.clip((SD_MF_REFERENCE - mf_median) / SD_MF_REFERENCE, 0.0, 1.0))
    return loss, mf


# ── Visualization helper ─────────────────────────────────────────────────────

def normalize_map(values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    ref = values[mask] if (mask is not None and np.any(mask)) else values.ravel()
    if ref.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    lo, hi = np.percentile(ref, [2, 98])
    if hi <= lo:
        lo, hi = float(ref.min()), float(ref.max()) + 1e-6
    out = np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    if mask is not None:
        out = out.copy()
        out[~mask] = 0.0
    return out


# ── Scorer class ─────────────────────────────────────────────────────────────

@dataclass
class ArtifactResult:
    score_0_10: float
    score_raw:  float
    metrics:    dict[str, float]
    parts:      dict[str, float]
    maps:       dict = field(default_factory=dict)


class ArtifactScorer:
    def __init__(
        self,
        rembg_model: str = "u2net",
        post_process_mask: bool = True,
        rembg_session=None,
    ):
        self.rembg_model       = rembg_model
        self.post_process_mask = post_process_mask
        self._session          = rembg_session

    def _ensure_session(self):
        if self._session is None:
            from rembg import new_session
            self._session = new_session(self.rembg_model)
        return self._session

    def run(self, pil_image: Image.Image) -> ArtifactResult:
        img     = pil_image.convert("RGB")
        img_rgb = np.array(img)
        gray    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        mask = build_rembg_mask(
            img, self.rembg_model, self.post_process_mask, self._ensure_session()
        )
        if mask is None or not np.any(mask):
            metrics = {k: 0.0 for k in ARTIFACT_METRICS}
            score_0_10, parts = artifact_score(metrics)
            return ArtifactResult(score_0_10, score_0_10 / 10.0, metrics, parts)

        denoised = denoise_tv_chambolle(gray, weight=0.10)
        residual = np.abs(gray - denoised)
        mean_    = cv2.blur(gray, (5, 5))
        mean_sq  = cv2.blur(gray * gray, (5, 5))
        lvar     = np.maximum(0.0, mean_sq - mean_ * mean_)

        blockiness, viz_block = compute_blockiness(gray, mask)
        dct_period, viz_dct   = compute_dct_periodicity(residual, mask)
        ringing,    viz_ring  = compute_ringing(denoised, mask)
        grain_loss, viz_grain = compute_grain_loss(residual, lvar, mask)
        mf_loss,    viz_mf    = compute_mf_loss(gray, mask)

        metrics = {
            "fg_blockiness": blockiness,
            "fg_dct_period": dct_period,
            "fg_ringing":    ringing,
            "fg_grain_loss": grain_loss,
            "fg_mf_loss":    mf_loss,
        }
        score_0_10, parts = artifact_score(metrics)
        cutout = img_rgb.copy()
        cutout[~mask] = 0

        return ArtifactResult(
            score_0_10=score_0_10,
            score_raw=score_0_10 / 10.0,
            metrics=metrics,
            parts=parts,
            maps={
                "img_rgb":   img_rgb,
                "cutout":    cutout,
                "mask":      mask,
                "viz_block": viz_block,
                "viz_dct":   viz_dct,
                "viz_ring":  viz_ring,
                "viz_grain": viz_grain,
                "viz_mf":    viz_mf,
            },
        )


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_result(result: ArtifactResult) -> None:
    m    = result.maps
    mask = m["mask"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"Artifact Scorer   {result.score_0_10:.2f}/10"
        f"  |  blockiness={result.metrics['fg_blockiness']:.3f}"
        f"  dct_period={result.metrics['fg_dct_period']:.3f}"
        f"  ringing={result.metrics['fg_ringing']:.3f}"
        f"  grain_loss={result.metrics['fg_grain_loss']:.3f}"
        f"  mf_loss={result.metrics['fg_mf_loss']:.3f}",
        fontsize=9,
    )

    axes[0, 0].imshow(m["img_rgb"])
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(m["cutout"])
    axes[0, 1].set_title("Subject Cutout")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(normalize_map(m["viz_block"], mask), cmap="hot")
    axes[0, 2].set_title(f"Block Boundaries (8px grid)  [{result.parts['fg_blockiness']:.3f}]")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(normalize_map(m["viz_grain"], mask), cmap="plasma")
    axes[1, 0].set_title(f"Grain Loss (flat-region TV residual)  [{result.parts['fg_grain_loss']:.3f}]")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(m["viz_ring"])
    axes[1, 1].set_title(f"Ringing (Edge=red, Ring=green)  [{result.parts['fg_ringing']:.3f}]")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(normalize_map(m["viz_mf"], mask), cmap="magma")
    axes[1, 2].set_title(f"Mid-Freq Energy Loss  [{result.parts['fg_mf_loss']:.3f}]")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.show()


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Artifact-focused scorer for SD images.")
    parser.add_argument("image_path", type=str)
    parser.add_argument("--rembg-model", type=str, default="u2net")
    parser.add_argument("--no-rembg-post-process", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    img = Image.open(Path(args.image_path))
    scorer = ArtifactScorer(
        rembg_model=args.rembg_model,
        post_process_mask=not args.no_rembg_post_process,
    )
    result = scorer.run(img)

    print("\n--- Artifact Score ---")
    print(f"Score:          {result.score_0_10:.3f} / 10.0  (raw={result.score_raw:.4f})")
    print(f"fg_blockiness   {result.metrics['fg_blockiness']:.6f}  (part={result.parts['fg_blockiness']:.4f})")
    print(f"fg_dct_period   {result.metrics['fg_dct_period']:.6f}  (part={result.parts['fg_dct_period']:.4f})")
    print(f"fg_ringing      {result.metrics['fg_ringing']:.6f}  (part={result.parts['fg_ringing']:.4f})")
    print(f"fg_grain_loss   {result.metrics['fg_grain_loss']:.6f}  (part={result.parts['fg_grain_loss']:.4f})")
    print(f"fg_mf_loss      {result.metrics['fg_mf_loss']:.6f}  (part={result.parts['fg_mf_loss']:.4f})")

    if not args.no_plot:
        plot_result(result)

if __name__ == "__main__":
    main()
