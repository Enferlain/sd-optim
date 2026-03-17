import functools
import enum
import operator
import logging
import torch
import math
import torch.nn.functional as F
import fnmatch
import ptwt

from torch import Tensor
from sd_mecha import Parameter, Return, merge_method  # Import Parameter and Return

from sd_optim.svd import torch_svd_lowrank  # you need to make your own or use the one from mecha
from sd_mecha.extensions.builtin.merge_methods.svd import svd_lowrank, stiefel_interpolate

try:
    import cupy as cp
    from cupy.cuda import cusolver

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

EPSILON = 1e-10
logger = logging.getLogger(__name__)


@merge_method
def delta_widen(
        *deltas: Parameter(Tensor, "delta"),  # subtract
        magnitude_ratio: Parameter(float) = 2.0,  # weight of magnitude divergence
        direction_ratio: Parameter(float) = 2.0,  # weight of directional divergence
        temperature: Parameter(float) = 1.0,  # softmax sharpness
        critical_quantile: Parameter(float) = 0.80,  # pooled per-parameter threshold across models
        topk: Parameter(int) = 0,  # 0=off; 1..M enables per-column top-k gating
        rank_blend: Parameter(float) = 0.0,  # 0=off; 0.3–0.6 blends rank with raw divergences
        baseline_index: Parameter(int) = 1,  # -1 = zero baseline; >=0 = anchor model
        baseline_bias: Parameter(float) = 0.0,  # logit boost for baseline model when anchoring
        keep_baseline: Parameter(float) = 0.0,  # convex blend with baseline delta
        early_exit: Parameter(bool) = False,  # fast path when both ratios are 0
) -> Return(Tensor, "delta"):  # add diff
    """
    Stable WIDEN-style columnwise merge.

    Core knobs (highest effect first):
    - magnitude_ratio: Weight of magnitude divergence in logits; higher prioritizes columns with large norm changes (typical 1–3).
    - direction_ratio: Weight of directional divergence (1 - cosine); higher emphasizes changed directions (typical 1–3).
    - temperature: Softmax sharpness across models per column; lower picks winners, higher blends more (0.5–2.0).
    - critical_quantile: Cross-model per-parameter threshold for "critical" status; higher marks fewer columns as critical (0.70–0.90).
    - topk: Keep only top-k models per column before softmax; 0 disables, 2 is a good default for crisper gating.
    - rank_blend: Blend factor for rank-normalized vs raw divergences; 0.3–0.6 improves robustness across heterogeneous layers.

    Anchoring and safety:
    - baseline_index: -1 uses a zero baseline; >=0 anchors to that delta for patching-style merges.
    - baseline_bias: Additive logit boost for the baseline when anchoring; positive values favor the anchor per column (0.5–2.0).
    - keep_baseline: Convex blend with the baseline delta after merging; ensures a minimum of baseline signal (0.1–0.3).
    - early_exit: If both ratios are 0, return the baseline (or zeros) immediately to avoid unnecessary work.

    Notes:
    - Columnwise disentanglement for linear/conv weights; magnitude-only for 1D params (bias/norm).
    - Uses float32 for softmax stability and eps-guarded cosine similarity.
    """

    if len(deltas) == 0:
        raise ValueError("At least one model delta is required.")

    ref = deltas[baseline_index] if 0 <= baseline_index < len(deltas) else torch.zeros_like(deltas[0])
    dtype, device = deltas[0].dtype, deltas[0].device

    # Early exit if no divergence terms are used
    if early_exit and magnitude_ratio == 0.0 and direction_ratio == 0.0:
        return ref

    # Helpers: map param tensor to [d, k] columns (features) and back
    def to_dk(t: torch.Tensor):
        if t.dim() == 0:  # scalar
            return t.reshape(1, 1), (t.shape, "scalar")
        if t.dim() == 1:  # vector (bias/norm) -> magnitude-only
            return t.reshape(1, -1), (t.shape, "vec")
        if t.dim() == 2:  # linear [out, in] -> [d=out, k=in]
            return t, (t.shape, "linear")
        # conv [out, in, kh, kw] -> [d=out, k=in*kh*kw]
        d, c, kh, kw = t.shape
        return t.reshape(d, c * kh * kw), (t.shape, "conv")

    def components_dk(Wdk: torch.Tensor):
        # columnwise 2-norms and unit directions
        m = torch.linalg.vector_norm(Wdk, ord=2, dim=0)  # [k]
        D = Wdk / m.clamp_min(1e-12)  # [d, k]
        return m, D

    def minmax01(x: torch.Tensor):
        # normalize to [0,1] per model row
        x_min = x.min(dim=1, keepdim=True).values
        x_max = x.max(dim=1, keepdim=True).values
        return (x - x_min) / (x_max - x_min).clamp_min(1e-12)

    # Build columnwise components for each model and reference
    M = len(deltas)
    mags, dirs, metas = [], [], []
    for m in deltas:
        Wdk, meta = to_dk(m)
        metas.append(meta)
        if Wdk.numel() == 0:
            mags.append(torch.zeros(1, device=device, dtype=dtype))
            dirs.append(torch.zeros_like(Wdk))
            continue
        if meta[1] in ("scalar", "vec"):
            mags.append(Wdk.reshape(-1))  # KEEP SIGN for 1D
            dirs.append(None)
        else:
            mcol, D = components_dk(Wdk)
            mags.append(mcol)
            dirs.append(D)

    ref_dk, ref_meta = to_dk(ref)
    if ref_meta[1] in ("scalar", "vec"):
        ref_mag = ref_dk.reshape(-1)  # KEEP SIGN for 1D
        ref_dir = None
        is_zero_baseline = False
    else:
        ref_mag, ref_dir = components_dk(ref_dk)
        # Check if reference is effectively zero
        is_zero_baseline = torch.all(ref_mag < 1e-12).item()

    # Compute divergences per column j
    mag_divs, dir_divs = [], []
    for i in range(M):
        md = (mags[i] - ref_mag).abs()  # Correct for 1D because mags/ref_mag kept their signs
        mag_divs.append(md)
        if dirs[i] is None or ref_dir is None or is_zero_baseline:
            dd = torch.zeros_like(md)
        else:
            # per-column cosine similarity (dim=0 across rows)
            cos = F.cosine_similarity(dirs[i], ref_dir, dim=0, eps=1e-12)
            dd = 1.0 - cos  # in [0,2]
        dir_divs.append(dd)

    mag_divs = torch.stack(mag_divs, dim=0)  # [M, k]
    dir_divs = torch.stack(dir_divs, dim=0)  # [M, k]

    # Optional rank–raw hybridization for robustness
    if rank_blend > 0.0:
        mag_raw01 = minmax01(mag_divs)
        dir_raw01 = minmax01(dir_divs)
        # ranks within each model row (columns axis)
        mag_rank = torch.argsort(torch.argsort(mag_divs, dim=1), dim=1).to(mag_divs.dtype)
        dir_rank = torch.argsort(torch.argsort(dir_divs, dim=1), dim=1).to(dir_divs.dtype)
        den = (mag_divs.shape[1] - 1) if mag_divs.shape[1] > 1 else 1
        mag_rank01 = mag_rank / den
        dir_rank01 = dir_rank / den
        mag_divs = rank_blend * mag_rank01 + (1.0 - rank_blend) * mag_raw01
        dir_divs = rank_blend * dir_rank01 + (1.0 - rank_blend) * dir_raw01

    # Per-parameter pooled criticality via quantiles across models
    q = torch.tensor(critical_quantile, device=device, dtype=mag_divs.dtype)
    mag_thr = torch.quantile(mag_divs, q, dim=0)
    dir_thr = torch.quantile(dir_divs, q, dim=0)
    crit_mask = (mag_divs > mag_thr) | (dir_divs > dir_thr)  # [M, k]

    # Build logits with additive logit offset for critical positions
    logits = magnitude_ratio * mag_divs + direction_ratio * dir_divs
    logits = logits + 0.5 * crit_mask.to(logits.dtype)  # fixed δ=0.5 offset (simple, effective)

    # Optional baseline logit bias (anchoring)
    if 0 <= baseline_index < M and baseline_bias != 0.0:
        logits[baseline_index] = logits[baseline_index] + baseline_bias

    # Optional per-column top-k gating before softmax
    if topk > 0 and topk < M:
        _, idx = torch.topk(logits, topk, dim=0)  # [topk, k]
        mask = torch.full_like(logits, float("-inf"))
        logits = mask.scatter(0, idx, logits.gather(0, idx))

    # Softmax over models in float32 for stability
    logits32 = logits.to(torch.float32) / float(temperature)
    weights = torch.softmax(logits32, dim=0).to(dtype)  # [M, k]

    # Merge back with correct broadcasting per param type
    merged = torch.zeros_like(deltas[0])
    for i in range(M):
        meta = metas[i]
        if meta[1] in ("scalar", "vec"):
            w = weights[i].reshape(-1)
            merged = merged + deltas[i] * (w if w.numel() == deltas[i].numel() else 1.0)
        elif meta[1] == "linear":  # [out, in] -> broadcast over rows
            wcol = weights[i].reshape(1, -1)
            merged = merged + deltas[i] * wcol
        else:  # conv [out, in, kh, kw] -> FIXED: per position weights
            d, c, kh, kw = deltas[i].shape
            # FIX: Reshape to full [1, c, kh, kw] to match columnwise disentanglement
            wcol = weights[i].reshape(1, c, kh, kw)
            merged = merged + deltas[i] * wcol

    if keep_baseline != 0.0:
        merged = keep_baseline * ref + (1.0 - keep_baseline) * merged

    return merged
