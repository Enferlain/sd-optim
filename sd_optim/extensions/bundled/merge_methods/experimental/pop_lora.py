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
def pop_lora(
        a: Parameter(Tensor, "weight"),
        b: Parameter(Tensor, "weight"),
        *,
        alpha: Parameter(Tensor) = 0.5,
        rank_ratio: Parameter(float) = 0.25,
        early_exit: Parameter(bool) = False,
        **kwargs,
) -> Return(Tensor, "weight"):
    """
    Merge two weight tensors using Pivoted Orthogonal Projection (POP) LoRA.

    Projects the difference between tensors onto the column space of 'a', then
    applies low-rank approximation via column-pivoted QR decomposition.

    Args:
        a (Tensor): Source weight tensor.
        b (Tensor): Target weight tensor.
        alpha (Tensor or float, optional): Blend ratio (0.0=a, 1.0=b). Default: 0.5.
        rank_ratio (float, optional): Low-rank approximation ratio. Default: 0.25.
        early_exit (bool, optional): Enable fast paths for edge cases. Default: False.

    Returns:
        Tensor: Merged weight tensor with same shape as inputs.

    Notes:
        Falls back to linear interpolation for 1D tensors and token embeddings.
    """
    original_shape = a.shape
    key = kwargs["key"]
    cache = kwargs.get("cache")

    shape_a = a.shape
    shape_b = b.shape

    # The most important check: do the shapes match?
    if shape_a != shape_b:
        logger.error("=" * 80)
        logger.error("MISMATCH DETECTED FOR KEY: %s", key)
        logger.error("Shape of Tensor 'a': %s", shape_a)
        logger.error("Shape of Tensor 'b': %s", shape_b)
        logger.error("Dtype of 'a': %s, Device: %s", a.dtype, a.device)
        logger.error("Dtype of 'b': %s, Device: %s", b.dtype, b.device)
        logger.error("=" * 80)

    layer_cache = None
    if cache is not None:
        if key not in cache:
            cache[key] = {}
        layer_cache = cache[key]

    # Early exit handling
    if early_exit and alpha == 0.0:
        return a
    if early_exit and alpha == 1.0:  # Add missing early exit
        return b

    if len(original_shape) <= 1:
        return (1 - alpha) * a + alpha * b

    if "token_embedding" in key or len(original_shape) <= 1:
        return (1 - alpha) * a + alpha * b

    # if key:
    #     alpha_val = float(alpha) if isinstance(alpha, torch.Tensor) else alpha
    #     print(f"[pop_lora] Key: {key} -- Using alpha: {alpha_val:.4f}")

    # Reshaping logic
    if "token_embedding" in key:
        a_2d = a
        b_2d = b
    elif len(original_shape) == 4:
        a_2d = a.reshape(original_shape[0], -1)
        b_2d = b.reshape(original_shape[0], -1)
    elif len(original_shape) == 2:
        a_2d = a
        b_2d = b
    else:
        a_2d = a.reshape(original_shape[0], -1)
        b_2d = b.reshape(original_shape[0], -1)

    def _cpqr(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Column-pivoted QR (Businger–Golub)"""
        m, n = A.shape
        device, dtype = A.device, A.dtype
        R = A.clone()
        piv = torch.arange(n, device=device)
        Q = torch.eye(m, device=device, dtype=dtype)

        col_norms = (R.to(torch.float32).pow(2).sum(dim=0)).to(torch.float64)
        eps64 = torch.finfo(torch.float64).eps

        for k in range(min(m, n)):
            j_rel = torch.argmax(col_norms[k:])
            j = k + int(j_rel.item())
            if j != k:
                R[:, [k, j]] = R[:, [j, k]]
                piv[[k, j]] = piv[[j, k]]
                col_norms[[k, j]] = col_norms[[j, k]]

            x = R[k:, k]
            norm_x = torch.linalg.norm(x)
            if norm_x <= eps64:
                continue

            # Scalar sign from first element
            sgn = 1.0 if float(x[0]) >= 0.0 else -1.0
            v = x.clone()
            v[0] += sgn * norm_x
            v_norm = torch.linalg.norm(v)
            if v_norm > 0:
                v = v / v_norm
            tau = torch.tensor(2.0, device=device, dtype=dtype)

            R_sub = R[k:, k:]
            w = v @ R_sub
            R[k:, k:] = R_sub - (v.unsqueeze(1) @ (tau * w).unsqueeze(0))

            Q_sub = Q[:, k:]
            wq = Q_sub @ v
            Q[:, k:] = Q_sub - (wq.unsqueeze(1) @ (tau * v).unsqueeze(0))

            if k + 1 < m:
                R[k + 1 :, k] = 0

            if k + 1 < n:
                col_norms[k + 1 :] = torch.clamp(col_norms[k + 1 :] - R[k, k + 1 :].to(torch.float64).pow(2), min=0.0)
                if (k % 8) == 7 or k == 0:
                    col_norms[k + 1 :] = (R[k:, k + 1 :].to(torch.float32).pow(2).sum(dim=0)).to(torch.float64)

        return Q, R, piv

    def _unpivot_R(R: torch.Tensor, piv: torch.Tensor, n_cols: int) -> torch.Tensor:
        # FIX: Use proper torch.zeros dimensions
        R_unp = torch.zeros(R.size(0), n_cols, device=R.device, dtype=R.dtype)
        R_unp[:, piv] = R
        return R_unp

    # 1) Basis from a (QR) - NO CACHE, recompute each time
    Qa, _Ra = torch.linalg.qr(a_2d, mode="reduced")

    # 2) Project difference - NO CACHE, recompute each time
    diff = b_2d - a_2d
    projected_diff = Qa @ (Qa.T @ diff) if Qa.shape[1] > 0 else torch.zeros_like(diff)

    # 3) CPQR on projected difference - CACHE ONLY THIS
    if projected_diff.numel() == 0:
        low_rank_diff = torch.zeros_like(projected_diff)
    else:
        if layer_cache is not None and "Qd" in layer_cache and "R_unp" in layer_cache:
            Qd = layer_cache["Qd"].to(device=a.device, dtype=a.dtype)
            R_unp = layer_cache["R_unp"].to(device=a.device, dtype=a.dtype)
        else:
            Qd, Rd, piv = _cpqr(projected_diff)
            R_unp = _unpivot_R(Rd, piv, projected_diff.shape[1])
            if layer_cache is not None:
                layer_cache["Qd"] = Qd.detach().cpu()
                layer_cache["R_unp"] = R_unp.detach().cpu()

        max_rank = min(Qd.shape[1], R_unp.shape[0], R_unp.shape[1])
        r = max(1, int(max_rank * float(rank_ratio)))
        r = min(r, max_rank)
        low_rank_diff = Qd[:, :r] @ R_unp[:r, :]

    alpha_f = float(alpha) if isinstance(alpha, torch.Tensor) else alpha
    merged = a_2d + alpha_f * low_rank_diff
    return merged.reshape(original_shape)