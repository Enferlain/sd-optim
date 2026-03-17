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

from sd_optim.extensions.bundled.merge_methods import torch_svd_lowrank
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
def butterfly_projection(
        a: Parameter(Tensor, "weight"),
        b: Parameter(Tensor, "weight"),
        *,
        alpha: Parameter(float) = 0.5,
        rank_ratio: Parameter(float) = 0.25,
        lora_dim: Parameter(int) = 64,
        constraint: Parameter(float) = 0.05,
        boft_iters: Parameter(int) = 3,
        boft_step_scale: Parameter(float) = 0.1,
        projector_eps: Parameter(float) = 1e-6,
        seed: Parameter(int) = None,
        early_exit: Parameter(bool) = False,
        **kwargs,
) -> Return(Tensor, "weight"):
    """
    Merges tensors 'a' and 'b' using data-aligned butterfly orthogonalization.

    The method creates a data-aligned subspace using butterfly factorization,
    projects the difference into this subspace, and applies a data-aligned
    low-rank approximation within the subspace (not optimal SVD, but structured).

    Args:
        a: First input tensor.
        b: Second input tensor.
        alpha: Interpolation factor.
        rank_ratio: The ratio of dimensions to keep for low-rank approximation.
        lora_dim: Controls the subspace dimension for projection.
        constraint: Controls the magnitude of butterfly rotations (0-1).
        boft_iters: Number of alignment iterations for butterfly orthogonalization.
        boft_step_scale: Scaling factor for the alignment step size.
        projector_eps: Epsilon for numerical stability in the Cholesky projector.
        seed: Optional seed for deterministic behavior.
        **kwargs: Keyword arguments, including 'key' for layer identification.

    Returns:
        Merged tensor, reshaped to original shape of 'a'.
    """
    original_shape = a.shape
    key = kwargs.get("key", "")
    cache = kwargs.get("cache")

    # Early exit if alpha is 0.0 and flag is above 0.0
    if early_exit and alpha == 0.0:
        return a

    if early_exit and alpha == 1.0:
        return b

    if key.endswith(("in_proj_weight", "in_proj_bias")):
        # **FIX: Propagate all parameters in recursive calls**
        vs = []
        for i, k in enumerate(("to_q", "to_k", "to_v")):
            k_kwargs = kwargs.copy()
            k_kwargs["key"] = key.replace("in_proj_", f"{k}.")
            dim = a.shape[0] // 3
            t_start = dim * i
            t_end = dim * (i + 1)
            k_a = a[t_start:t_end]
            k_b = b[t_start:t_end]
            vs.append(
                butterfly_projection.__wrapped__(
                    k_a,
                    k_b,
                    alpha=alpha,
                    rank_ratio=rank_ratio,
                    lora_dim=lora_dim,
                    constraint=constraint,
                    boft_iters=boft_iters,
                    boft_step_scale=boft_step_scale,
                    projector_eps=projector_eps,
                    seed=seed,
                    early_exit=early_exit,
                    **k_kwargs,
                )
            )
        return torch.cat(vs)

    if "token_embedding" in key or len(original_shape) <= 1:
        return (1 - alpha) * a + alpha * b

    # Reshape based on layer type
    if len(original_shape) == 4:  # Convolutional layers
        if original_shape[2] == 1 and original_shape[3] == 1:  # 1x1 conv
            a_2d = a.reshape(original_shape[0], -1)
            b_2d = b.reshape(original_shape[0], -1)
        else:  # Assume 3x3 (or other) conv
            a_2d = a.reshape(original_shape[0], -1)
            b_2d = b.reshape(original_shape[0], -1)
    elif len(original_shape) == 2:  # Linear layers
        a_2d = a
        b_2d = b
    else:
        # Fallback for unexpected shapes
        a_2d = a.reshape(original_shape[0], -1)
        b_2d = b.reshape(original_shape[0], -1)

    # Auto-select lora_dim if not specified
    if lora_dim <= 0:
        dimension = a_2d.shape[0]
        if dimension < 768:
            lora_dim = 4
        elif dimension < 1536:
            lora_dim = 8
        elif dimension < 4096:
            lora_dim = 16
        else:
            lora_dim = 32

    # **NEW: Calculate consistent subspace dimension using the same clamping logic**
    out_dim = a_2d.shape[0]
    padded_dim = _next_pow2(out_dim)
    lora_dim_pow2 = _clamp_lora_dim_pow2(lora_dim, padded_dim)

    # Use the clamped dimension for consistency with butterfly_orthogonalize
    subspace_dim = min(lora_dim_pow2, min(a_2d.shape))

    # More stable seed generation
    seed_a = seed if seed is not None else stable_seed_from_tensor(a_2d, key)

    # **UPDATED: Cache key now includes lora_dim_pow2 for consistency**
    if cache is not None:
        cache_key = f"{key}_lora{lora_dim_pow2}"  # Include clamped lora_dim in cache key
        if cache_key not in cache:
            cache[cache_key] = {}
        layer_cache = cache[cache_key]
    else:
        layer_cache = None

    if layer_cache is not None and "Q_a_full" in layer_cache:
        Q_a_full = layer_cache["Q_a_full"].to(device=a.device, dtype=a.dtype)
    else:
        # Apply data-aligned butterfly orthogonalization to create subspace basis
        # **UPDATED: Pass lora_dim_pow2 for consistency**
        Q_a_full = butterfly_orthogonalize(
            a_2d, key, lora_dim_pow2, constraint, a.device, seed=seed_a, guide=a_2d, iters=boft_iters, step=constraint * boft_step_scale
        )
        if layer_cache is not None:
            layer_cache["Q_a_full"] = Q_a_full.cpu()

    # Create projection matrix for subspace restriction
    P = Q_a_full[:, :subspace_dim]  # [out_dim, subspace_dim]
    diff = b_2d - a_2d

    # Modern QR-based projector with Cholesky fallback - CONVERT TO FP64
    k = P.shape[1]
    P_fp64 = P.to(torch.float64)
    diff_fp64 = diff.to(torch.float64)

    # Primary approach: QR-based projector (more numerically stable)
    try:
        Q, _ = torch.linalg.qr(P_fp64, mode="reduced")
        # Compute projection coefficients: E = Q^T diff
        E = (Q.T @ diff_fp64).to(P.dtype)

    except RuntimeError as qr_error:
        # Fallback to Cholesky if QR fails (very rare)
        logger.warning("QR failed, using Cholesky fallback: %s", qr_error)
        try:
            I_k = torch.eye(k, device=P.device, dtype=torch.float64)
            G = (P_fp64.T @ P_fp64) + projector_eps * I_k
            L = torch.linalg.cholesky(G)

            # Compute E using modern triangular solves
            rhs = P_fp64.T @ diff_fp64
            y = torch.linalg.solve_triangular(L, rhs, upper=False)
            E = torch.linalg.solve_triangular(L.T, y, upper=True).to(P.dtype)

        except RuntimeError as chol_error:
            logger.warning("Both QR and Cholesky failed, using pseudo-inverse: %s", chol_error)
            # Last resort: pseudo-inverse
            P_pinv = torch.linalg.pinv(P_fp64)
            E = (P_pinv @ diff_fp64).to(P.dtype)

    # Cap rank by subspace dimension
    r = min(max(1, int(min(E.shape) * rank_ratio)), subspace_dim)

    diff_seed = stable_seed_from_tensor(E, f"{key}_diff")

    # Optimal low-rank approximation in subspace coordinates using RSVD
    if E.shape[1] > 3 * r and E.shape[0] >= r:
        # Use randomized SVD for wide matrices (more efficient than exact SVD)
        power_iters = 1 if min(E.shape) > 100 else 0  # Power iterations for large matrices
        svd_driver = "gesvd" if E.is_cuda else "gesvda"
        U, S, Vt = svd_lowrank(E, rank=r, iters=power_iters, seed=diff_seed, driver=svd_driver)
        subspace_update = U * S.unsqueeze(-2) @ Vt  # Optimal rank-r approximation
    else:
        # Use exact SVD for small matrices (more accurate)
        svd_driver = "gesvd" if E.is_cuda else "gesvda"
        U, S, Vt = torch.linalg.svd(E, full_matrices=False, driver=svd_driver)
        if S.shape[0] > r:
            U, S, Vt = U[:, :r], S[:r], Vt[:r, :]
        subspace_update = U * S.unsqueeze(-2) @ Vt

    # Project back to ambient space - update is guaranteed to be in span(P)
    optimal_update = P @ subspace_update  # [out_dim, in_dim]

    # Merge with data-aligned butterfly-based update
    merged = a_2d + alpha * optimal_update

    return merged.reshape(original_shape)

def stable_seed_from_tensor(tensor, key=""):
    """
    Generate a stable seed from tensor content and key to avoid collisions.
    **IMPROVED: More stable across minor numeric variations**
    """
    import hashlib

    # **FIX: Use more stable tensor characteristics**
    # Quantize stats to avoid floating point noise affecting seeds
    def quantize(x, scale=1000):
        return int(float(x) * scale) / scale

    tensor_stats = [
        tensor.shape[0],
        tensor.shape[1] if len(tensor.shape) > 1 else 1,
        quantize(torch.median(tensor).item()),  # More stable than mean
        quantize(torch.norm(tensor).item()),
        str(tensor.dtype),
        str(tensor.device),
    ]

    # Combine with key string
    hash_input = f"{key}_{tensor_stats}"
    hash_obj = hashlib.md5(hash_input.encode())
    return int(hash_obj.hexdigest()[:8], 16) % 2147483647

def butterfly_orthogonalize(x, key, lora_dim=4, constraint=0.01, device=None, seed=None, guide=None, iters=1, step=None):
    """
    Creates a data-aligned orthogonal basis using butterfly factorization.

    Args:
        x (Tensor): Input tensor to base the orthogonalization on.
        key (str): Unique identifier for caching computed bases.
        lora_dim (int): Target low-rank dimension; internally adjusted to power-of-two.
        constraint (float): Maximum rotation amplitude constraint (0-1).
        device (torch.device, optional): Compute device.
        seed (int, optional): Random seed for reproducibility.
        guide (Tensor, optional): Guidance tensor for alignment.
        iters (int): Number of alignment iterations.
        step (float, optional): Step size for alignment iterations.

    Returns:
        Tensor: Orthogonal basis matrix aligned to input data.
    """
    if device is None:
        device = x.device

    # **FIX: Store original dtype for consistency**
    original_dtype = x.dtype

    # Coerce to 2D for all downstream linear algebra
    x2d = x.reshape(x.shape[0], -1)
    guide2d = (guide if guide is not None else x).reshape(x.shape[0], -1)

    if step is None:
        step = constraint

    out_dim = x2d.shape[0]

    # Use local generator instead of global seeding
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    try:
        # **NEW: Use helpers for power-of-two factorization**
        padded_dim = _next_pow2(out_dim)
        lora_dim_pow2 = _clamp_lora_dim_pow2(lora_dim, padded_dim)
        block_size, block_num = butterfly_factor(padded_dim, lora_dim_pow2)
        boft_m = int(math.log2(block_num))  # Now guaranteed to be valid

        # Initialize butterfly blocks with input-informed scaling
        oft_blocks = initialize_butterfly_blocks(x, boft_m, block_num, block_size, device, gen)

        # Apply data alignment sweep - returns orthogonal blocks directly
        if iters > 0 and step > 0:
            r_blocks = align_butterfly_blocks(oft_blocks, guide2d, boft_m, block_num, block_size, iters, step, device, key)
        else:
            # Convert initial skew blocks to orthogonal for consistency
            I = torch.eye(block_size, device=device, dtype=original_dtype)  # **ADD DTYPE**
            q = oft_blocks - oft_blocks.transpose(-1, -2)
            try:
                r_blocks = (I + q) @ torch.linalg.solve(I - q + 1e-10 * I, I)
            except Exception:
                r_blocks = torch.eye(block_size, device=device, dtype=original_dtype).expand_as(q)  # **ADD DTYPE**

        # **FIX: Ensure consistent dtype before apply_butterfly_transform**
        r_blocks = r_blocks.to(dtype=original_dtype)
        result = apply_butterfly_transform(x, r_blocks, boft_m, block_size, block_num, device, key)

        # **FIX: Ensure final result matches input dtype**
        return result.to(dtype=original_dtype)

    except Exception as e:
        logger.warning(
            "Butterfly factorization failed for dimension %s; falling back to QR decomposition. Error: %s",
            out_dim,
            e,
        )

        # Proper fallback that creates basis from guide
        Q, _ = torch.linalg.qr(guide2d, mode="reduced")

        if constraint > 0:
            # Apply constraint as magnitude scaling
            Q_norm = torch.norm(Q, dim=0, keepdim=True)
            constraint_value = constraint * torch.sqrt(
                torch.tensor(guide2d.shape[0], dtype=original_dtype, device=device)
            )  # **ADD DTYPE**
            scale = torch.clamp(constraint_value / (Q_norm + 1e-8), max=1.0)
            Q = Q * scale

        # **FIX: Ensure fallback result matches input dtype**
        return Q.to(dtype=original_dtype)

def initialize_butterfly_blocks(x, boft_m, block_num, block_size, device, generator=None):
    """
    Initializes butterfly blocks with anti-symmetric matrices scaled to input properties.

    Args:
        x (Tensor): Reference tensor for dtype and scaling statistics.
        boft_m (int): Number of butterfly layers (log2 of block count).
        block_num (int): Number of blocks in each butterfly layer.
        block_size (int): Size of each square block matrix.
        device (torch.device): Compute device.
        generator (torch.Generator, optional): Random generator for reproducibility.

    Returns:
        Tensor: Initialized butterfly blocks tensor of shape (boft_m, block_num, block_size, block_size).
    """
    dtype = x.dtype

    blocks = torch.zeros(boft_m, block_num, block_size, block_size, device=device, dtype=dtype)  # **ADD DTYPE**

    with torch.no_grad():
        # Scale initialization based on input tensor characteristics
        x_mean = torch.mean(torch.abs(x))

        # Adaptive scaling based on input statistics
        base_scale = min(0.01, x_mean.item() * 0.01)

        for i in range(boft_m):
            for j in range(block_num):
                # Create anti-symmetric matrices with input-informed scaling
                random_vals = (
                        torch.randn(block_size, block_size, device=device, dtype=dtype, generator=generator) * base_scale
                )  # **ADD DTYPE**

                # Decay scale with butterfly layer depth for stability
                layer_scale = 1.0 / (i + 1)

                # Make anti-symmetric (Q = -Q^T) and scale
                blocks[i, j] = (random_vals - random_vals.T) * layer_scale / 2.0

    return blocks

def align_butterfly_blocks(oft_blocks, guide2d, boft_m, block_num, block_size, iters, step, device, key):
    """
    Align butterfly blocks to a reference data subspace.

    Uses fast Cayley transform steps combined with Newton–Schulz iterations
    to compute orthogonal rotations that align the blocks to the target data.
    Supports fallback to Stiefel manifold interpolation for non-square blocks.

    Args:
        oft_blocks (Tensor): Initial butterfly blocks, skew-symmetric matrices.
        guide2d (Tensor): Data guiding the alignment.
        boft_m (int): Number of butterfly factorization stages.
        block_num (int): Number of butterfly blocks.
        block_size (int): Size of each block.
        iters (int): Number of alignment iterations.
        step (float): Initial alignment step size.
        device (torch.device): Computation device.
        key (str): Identifier for caching and logging.

    Returns:
        Tensor: Series of aligned orthogonal butterfly blocks.
    """
    original_dtype = oft_blocks.dtype

    def _polar_newton(A: torch.Tensor, iters: int = 2) -> torch.Tensor:
        """
        Approximate nearest orthogonal matrix via Newton–Schulz iteration.
        Assumes A is already near-orthogonal for best convergence.
        """
        Q = A
        I = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
        for _ in range(iters):
            QTQ = Q.mH @ Q
            Q = 0.5 * Q @ (3 * I - QTQ)
        return Q

    with torch.no_grad():
        # Convert blocks to orthogonal matrices via Cayley transform
        I = torch.eye(block_size, device=device, dtype=original_dtype)
        q = oft_blocks - oft_blocks.transpose(-1, -2)  # Make anti-symmetric

        # Convert to orthogonal via Cayley: R = (I+Q)(I-Q)^-1
        try:
            r = (I + q) @ torch.linalg.solve(I - q + 1e-10 * I, I)
        except Exception:
            r = torch.eye(block_size, device=device, dtype=original_dtype).expand_as(q)

        # Apply alignment iterations - work directly with orthogonal matrices
        for iter_idx in range(iters):
            alignment_strength = step / (iter_idx + 1)  # Decay over iterations

            for i in range(boft_m):
                stage_step = alignment_strength / (i + 1)  # Further decay by stage

                for j in range(block_num):
                    start_idx = j * block_size
                    end_idx = min((j + 1) * block_size, guide2d.shape[0])

                    if end_idx <= start_idx:
                        continue

                    actual_block_size = end_idx - start_idx

                    # Skip degenerate cases
                    if actual_block_size < 2:
                        continue

                    # Get current orthogonal matrix
                    current_Q = r[i, j, :actual_block_size, :actual_block_size]

                    # Create target alignment from guide data
                    guide_block = guide2d[start_idx:end_idx]

                    if guide_block.shape[1] >= actual_block_size:
                        # Use QR to get orthogonal target from guide
                        target_Q, _ = torch.linalg.qr(guide_block[:, :actual_block_size], mode="reduced")

                        # Enforce SO(n) target (avoid reflections)
                        sign = torch.linalg.slogdet(target_Q)
                        if sign.sign < 0:
                            target_Q = target_Q.clone()
                            target_Q[:, -1] = -target_Q[:, -1]

                        try:
                            if current_Q.shape[0] == current_Q.shape[1]:
                                # **NEW: Fast Cayley step instead of fractional matrix power**
                                R_rel = target_Q @ current_Q.mH

                                # Short backtracking with Cayley update
                                max_step = 0.1 / (i + 1)
                                tau = min(stage_step, max_step)

                                for _ in range(4):  # Reduced from 8 to 4 attempts
                                    # Cayley step: solve (I - τS) X = (I + τS)
                                    S = 0.5 * (R_rel - R_rel.mH)  # skew-symmetric part
                                    I_loc = torch.eye(actual_block_size, device=current_Q.device, dtype=current_Q.dtype)

                                    try:
                                        update = torch.linalg.solve(I_loc - tau * S, I_loc + tau * S)
                                        trial = update @ current_Q

                                        # 2-step Newton–Schulz polar retraction (replaces SVD)
                                        trial = _polar_newton(trial, iters=2)

                                        step_norm = torch.linalg.matrix_norm(trial - current_Q)

                                        if step_norm <= max_step or tau < 1e-6:
                                            aligned_Q = trial
                                            break
                                        tau *= 0.5
                                    except Exception:
                                        # If solve fails, try smaller step
                                        tau *= 0.5
                                        if tau < 1e-6:
                                            aligned_Q = current_Q
                                            break
                                else:
                                    aligned_Q = trial
                            else:
                                # Tall block: Stiefel geodesic (rectangular case)
                                # Keep existing stiefel_interpolate for non-square blocks
                                aligned_Q = stiefel_interpolate(
                                    current_Q.to(torch.float64), target_Q.to(torch.float64), stage_step, eps=1e-8, max_iters=50
                                ).to(current_Q.dtype)

                            # **FIX: Ensure aligned_Q has correct dtype before assignment**
                            r[i, j, :actual_block_size, :actual_block_size] = aligned_Q.to(dtype=original_dtype)

                        except (RuntimeError, AssertionError):
                            # Fallback: keep current block
                            r[i, j, :actual_block_size, :actual_block_size] = current_Q

        # **FIX: Ensure final result has correct dtype**
        return r.to(dtype=original_dtype)

def butterfly_factor(dimension: int, factor: int = -1) -> tuple[int, int]:
    """
    Compute butterfly factorization parameters.

    Given a dimension and optional factor, computes block size and block number
    that are compatible with butterfly matrix structure.
    Ensures `block_num` is a power of two, suitable for FFT-like operations.

    Args:
        dimension (int): Input dimension to factorize.
        factor (int): Desired factor; if <= 0, a default factor based on log scale is used.

    Returns:
        tuple[int, int]: (block_size, block_num) representing the factorization.
    """
    if dimension <= 1:
        return max(1, dimension), 1

    # Default factor if not provided
    if factor <= 0:
        factor = 2 ** max(1, int(math.log2(dimension) // 4))

    # Clamp factor to a power-of-two divisor of dimension
    f = 1
    while (f << 1) <= factor and (dimension % (f << 1) == 0):
        f <<= 1

    block_size = max(2, dimension // f)
    block_num = max(1, dimension // block_size)

    # Force block_num to power-of-two (defensive)
    while block_num & (block_num - 1):  # Check if not power of 2
        block_size <<= 1
        block_num = dimension // block_size
        if block_size > dimension:
            block_size = dimension
            block_num = 1
            break

    return block_size, block_num

def apply_butterfly_transform(x, r_blocks, boft_m, block_size, block_num, device, key):
    """
    Apply butterfly orthogonal transformation using precomputed orthogonal blocks.

    Transforms input tensor using a butterfly factorization approach that preserves
    orthogonality through staged block-wise operations on a power-of-two padded grid.

    Args:
        x (Tensor): Input tensor to determine output dimensions and dtype.
        r_blocks (Tensor): Precomputed orthogonal blocks with shape
                          (boft_m, block_num, block_size, block_size).
        boft_m (int): Number of butterfly stages (log2 of block_num).
        block_size (int): Size of individual transformation blocks.
        block_num (int): Number of blocks in the butterfly factorization.
        device (torch.device): Device for tensor operations.
        key (str): Layer identifier for debugging/logging.

    Returns:
        Tensor: Orthogonal transformation matrix of shape (x.shape[0], x.shape[0]).
    """
    out_dim = x.shape[0]
    min_butterfly_dim = block_size * block_num

    # True padding: work on a power-of-two grid >= out_dim
    padded_dim = 1
    while padded_dim < out_dim:
        padded_dim <<= 1

    # Ensure we have at least the butterfly coverage; extra dims are identity
    working_dim = max(min_butterfly_dim, padded_dim)

    # Initialize with identity on the working dimension
    result = torch.eye(working_dim, device=device, dtype=x.dtype)

    # Conditions guaranteed true with padded factorization
    for stage in range(boft_m):
        stage_blocks = r_blocks[stage]

        # Calculate stride and step for this stage (powers of 2)
        stride = 1 << stage
        step_size = max(1, block_size >> stage)

        # Elegant reshape/transpose path (no condition checks needed)
        temp = result.unflatten(-1, (-1, 2, stride * step_size))
        temp = temp.transpose(-2, -1).flatten(-3)
        temp = temp.unflatten(-1, (-1, block_size))

        # Apply block transformations using orthogonal matrices
        valid_blocks = min(temp.shape[-2], stage_blocks.shape[0])
        for j in range(valid_blocks):
            if temp.shape[-1] >= stage_blocks.shape[-1]:
                # **FIXED: Right-multiply with conjugate transpose**
                temp[..., j, :] = temp[..., j, :] @ stage_blocks[j].mH

        # Reshape back
        temp = temp.flatten(-2)
        temp = temp.unflatten(-1, (-1, stride * step_size, 2))
        temp = temp.transpose(-2, -1).flatten(-3)
        result = temp

    # Extract the portion corresponding to original dimensions
    final_result = torch.eye(out_dim, device=device, dtype=x.dtype)
    copy_dim = min(out_dim, working_dim)
    final_result[:copy_dim, :copy_dim] = result[:copy_dim, :copy_dim]

    return final_result

def _next_pow2(n: int) -> int:
    """Return the next power of 2 greater than or equal to n"""
    p = 1
    while p < n:
        p <<= 1
    return p

def _clamp_lora_dim_pow2(lora_dim: int, padded_dim: int) -> int:
    """Choose largest power-of-two ≤ lora_dim that divides padded_dim"""
    d = 1
    while (d << 1) <= lora_dim and (padded_dim % (d << 1) == 0):
        d <<= 1
    return d


# OLD

# @merge_method
# def butterfly_merge(
#         a: Parameter(Tensor, "weight"),
#         b: Parameter(Tensor, "weight"),
#         *,
#         alpha: Parameter(float) = 0.5,
#         rank_ratio: Parameter(float) = 0.25,
#         lora_dim: Parameter(int) = 64,
#         constraint: Parameter(float) = 0.05,
#         early_exit: Parameter(bool) = True,
#         **kwargs
# ) -> Return(Tensor, "weight"):
#     """
#     Merges tensors 'a' and 'b' using butterfly orthogonalization.
#
#     Args:
#         a: First input tensor.
#         b: Second in put tensor.
#         alpha: Interpolation factor.
#         rank_ratio: The ratio of the dimensions to keep.
#         lora_dim: Controls complexity of butterfly factorization. If -1, auto-selected.
#         constraint: Controls orthogonality constraint (0-1).
#         epsilon: Small value for numerical stability.
#         **kwargs: Keyword arguments, including 'key' for layer identification.
#
#     Returns:
#         Merged tensor, reshaped to original shape of 'a'.
#     """
#     original_shape = a.shape
#     key = kwargs.get("key", "")
#     print(
#         f"DEBUG butterfly_merge called for key: {key} | alpha type: {type(alpha)}, value: {alpha} | rank_ratio type: {type(rank_ratio)}, value: {rank_ratio}")
#
#     # Early exit if alpha is 0.0 and flag is above 0.0
#     if early_exit and alpha == 0.0:
#         return a
#
#     if early_exit and alpha == 1.0:
#         return b
#
#     if key.endswith(("in_proj_weight", "in_proj_bias")):
#         # workaround for concatenated attention projection layers
#         vs = []
#         for i, k in enumerate(("to_q", "to_k", "to_v")):
#             k_kwargs = kwargs.copy()
#             k_kwargs["key"] = key.replace("in_proj_", f"{k}.")
#             dim = a.shape[0] // 3
#             t_start = dim * i
#             t_end = dim * (i + 1)
#             k_a = a[t_start:t_end]
#             k_b = b[t_start:t_end]
#             vs.append(butterfly_merge.__wrapped__(k_a, k_b, **k_kwargs))
#         return torch.cat(vs)
#
#     if "token_embedding" in key or len(original_shape) <= 1:
#         return (1 - alpha) * a + alpha * b
#
#     # Reshape based on layer type
#     if len(original_shape) == 4:  # Convolutional layers
#         if original_shape[2] == 1 and original_shape[3] == 1:  # 1x1 conv
#             a_2d = a.reshape(original_shape[0], -1)
#             b_2d = b.reshape(original_shape[0], -1)
#         else:  # Assume 3x3 (or other) conv
#             a_2d = a.reshape(original_shape[0], -1)
#             b_2d = b.reshape(original_shape[0], -1)
#     elif len(original_shape) == 2:  # Linear layers
#         a_2d = a
#         b_2d = b
#     else:
#         # Fallback for unexpected shapes
#         a_2d = a.reshape(original_shape[0], -1)
#         b_2d = b.reshape(original_shape[0], -1)
#
#     # Auto-select lora_dim if not specified
#     if lora_dim <= 0:
#         dimension = a_2d.shape[0]
#         if dimension < 768:
#             lora_dim = 4
#         elif dimension < 1536:
#             lora_dim = 8
#         elif dimension < 4096:
#             lora_dim = 16
#         else:
#             lora_dim = 32
#
#     # Apply butterfly orthogonalization
#     Q_a = butterfly_orthogonalize(a_2d, lora_dim, constraint, a.device)
#
#     # Project the difference using the butterfly orthogonal basis
#     diff = b_2d - a_2d
#     projected_diff = Q_a @ (Q_a.T @ diff)
#
#     # Apply butterfly orthogonalization to the projected difference
#     Q_diff = butterfly_orthogonalize(projected_diff, lora_dim, constraint, a.device)
#
#     # Low-rank approximation
#     rank = max(1, int(min(projected_diff.shape) * rank_ratio))
#     Q_diff_trunc = Q_diff[:, :rank]
#     R_diff_trunc = Q_diff_trunc.T @ projected_diff
#
#     # Merge
#     merged = a_2d + alpha * (Q_diff_trunc @ R_diff_trunc)
#
#     return merged.reshape(original_shape)
#
# @staticmethod
# def butterfly_orthogonalize(x, lora_dim=4, constraint=0.01, device=None):
#     """
#     Creates an orthogonal basis for matrix x using butterfly factorization.
#
#     Args:
#         x: Input tensor to orthogonalize
#         lora_dim: Controls the complexity of the butterfly factorization
#         constraint: Controls the orthogonality constraint (0.0 = no constraint)
#         device: Device to use for computation
#
#     Returns:
#         Q: Orthogonal matrix that forms a basis for x
#     """
#     if device is None:
#         device = x.device
#
#     out_dim = x.shape[0]
#
#     # --- This is the corrected structure ---
#     try:
#         # Attempt the butterfly factorization path completely within the try block.
#
#         block_size, block_num = butterfly_factor(out_dim, lora_dim)
#         boft_m = sum(int(i) for i in f"{block_num - 1:b}") + 1
#
#         # If all parameters are calculated successfully, proceed.
#         oft_blocks = initialize_butterfly_blocks(x, boft_m, block_num, block_size, device)
#
#         # The return for the happy path is INSIDE the try block.
#         return apply_butterfly_transform(x, oft_blocks, boft_m, block_size, block_num, constraint,
#                                                       device)
#
#     except Exception as e:
#         # If ANYTHING in the try block fails, we land here.
#         print(
#             f"Butterfly factorization failed for dimension {out_dim}, falling back to QR decomposition. Error: {e}")
#
#         # The entire fallback logic is now INSIDE the except block.
#         # This guarantees it only runs on failure and doesn't use unassigned variables.
#         Q, _ = torch.linalg.qr(x, mode='reduced')
#
#         if constraint > 0:
#             Q_norm = torch.norm(Q)
#             constraint_value = constraint * x.shape[0]
#             if Q_norm > constraint_value:
#                 Q = Q * constraint_value / Q_norm
#
#         return Q
#
# @staticmethod
# def butterfly_factor(dimension: int, factor: int = -1) -> tuple[int, int]:
#     """
#     Factorize dimension into butterfly-compatible factors.
#     Returns (block_size, block_num) where block_num is a power of 2.
#     """
#     # If factor is negative, use a reasonable default
#     if factor <= 0:
#         factor = 2 ** max(1, int(math.log2(dimension) / 4))
#
#     # Find a factorization where both factors are powers of 2 if possible
#     # This is different from the original algorithm and better handles odd dimensions
#
#     # Find largest power of 2 less than or equal to dimension
#     n = 1
#     while n * 2 <= dimension:
#         n *= 2
#
#     # If dimension is a power of 2, split it evenly
#     if n == dimension:
#         block_size = max(2, n // factor) if factor > 0 else int(math.sqrt(n))
#         block_num = dimension // block_size
#         return block_size, block_num
#
#     # Otherwise, find valid factorization
#     if dimension % 2 == 0:
#         # For even dimensions, find a power-of-2 block_num that divides dimension
#         block_num = 1
#         while block_num * 2 <= factor and dimension % (block_num * 2) == 0:
#             block_num *= 2
#
#         block_size = dimension // block_num
#         return block_size, block_num
#     else:
#         # For odd dimensions, block_size must be the dimension itself
#         # This is a special case that doesn't use butterfly structure
#         # but allows the algorithm to work with any dimension
#         return dimension, 1
#
# @staticmethod
# def initialize_butterfly_blocks(x, boft_m, block_num, block_size, device):
#     """
#     Initialize butterfly blocks efficiently.
#     """
#     # Initialize anti-symmetric matrices (Q = -Q^T) with small random values
#     blocks = torch.zeros(boft_m, block_num, block_size, block_size, device=device)
#
#     with torch.no_grad():
#         # Scale initialization based on input tensor norm
#         norm = torch.norm(x) / (x.shape[0] * math.sqrt(boft_m * block_num))
#         scale = min(0.01, norm * 0.1)  # Limit scale to avoid numerical issues
#
#         for i in range(boft_m):
#             for j in range(block_num):
#                 # Create small random values scaled appropriately
#                 random_vals = torch.randn(block_size, block_size, device=device) * scale
#
#                 # Make it anti-symmetric (Q = -Q^T)
#                 blocks[i, j] = (random_vals - random_vals.T) / 2.0
#
#     return blocks
#
# @staticmethod
# def apply_butterfly_transform(x, oft_blocks, boft_m, block_size, block_num, constraint, device):
#     """
#     Apply butterfly orthogonal transformation to input x, preserving the
#     elegant approach from the original BOFT implementation.
#     """
#     # Identity matrix for Cayley transform
#     I = torch.eye(block_size, device=device)
#
#     # Make blocks anti-symmetric first
#     q = oft_blocks - oft_blocks.transpose(-1, -2)
#
#     # Apply constraint per block rather than globally
#     if constraint > 0:
#         constraint_value = constraint * x.shape[0] / (boft_m * block_num)  # Scale by number of blocks
#         for i in range(boft_m):
#             for j in range(block_num):
#                 block_norm = torch.norm(q[i, j]) + 1e-8
#                 if block_norm > constraint_value:
#                     q[i, j] = q[i, j] * constraint_value / block_norm
#
#     # Convert to orthogonal matrices via Cayley transform: R = (I+Q)(I-Q)^-1
#     try:
#         r = (I + q) @ torch.linalg.solve(I - q + 1e-10 * I, I)
#     except torch.linalg.LinAlgError:
#         # Fallback if matrix is singular
#         r = torch.eye(block_size, device=device).expand_as(q)
#
#     # Create identity matrix to handle arbitrary dimensions
#     out_dim = x.shape[0]
#     padded_dim = block_size * block_num
#
#     if out_dim > padded_dim:
#         # Matrix is bigger than butterfly factorization can handle
#         # Create full-size identity, but only transform the first padded_dim dimensions
#         result = torch.eye(out_dim, device=device)
#         transform_size = padded_dim
#         print(f"DEBUG: Large matrix {out_dim} > {padded_dim}, transforming first {transform_size} dims")
#     else:
#         # Matrix fits within butterfly factorization
#         # Create padded identity, transform all actual dimensions
#         result = torch.eye(padded_dim, device=device)
#         transform_size = out_dim
#
#     # Safety check
#     transform_size = max(1, min(transform_size, padded_dim, out_dim))
#
#     # Apply butterfly transformation using original approach
#     for i in range(boft_m):
#         bi = r[i]  # [block_num, block_size, block_size]
#         g = 2
#         k = 2 ** i * (block_size // 2)
#
#         # Only transform the active part of the matrix
#         active = result[:transform_size, :transform_size]
#
#         # Reshape for butterfly application - with proper padding handling
#         try:
#             # Try the elegant reshape approach
#             reshaped = active.unflatten(-1, (-1, g, k))
#             reshaped = reshaped.transpose(-2, -1).flatten(-3)
#             reshaped = reshaped.unflatten(-1, (-1, block_size))
#
#             # Apply the butterfly block - only to blocks that fit
#             valid_blocks = min(reshaped.shape[-2], bi.shape[0])
#             transformed = torch.zeros_like(reshaped)
#             transformed[..., :valid_blocks, :] = torch.einsum(
#                 "b i j, b j ... -> b i ...",
#                 bi[:valid_blocks],
#                 reshaped[..., :valid_blocks, :]
#             )
#
#             # Reshape back
#             transformed = transformed.flatten(-2)
#             transformed = transformed.unflatten(-1, (-1, k, g))
#             transformed = transformed.transpose(-2, -1).flatten(-3)
#
#             # Update the result
#             result[:transform_size, :transform_size] = transformed
#
#         except RuntimeError:
#             # If reshape fails due to dimension issues, fall back to block-by-block
#             # This preserves the correct transformation even when dimensions don't align perfectly
#             for j in range(min(block_num, transform_size // block_size)):
#                 start = j * block_size
#                 end = min((j + 1) * block_size, transform_size)
#                 if end <= start:
#                     continue
#
#                 # Apply transformation to this block
#                 block = result[start:end, start:end]
#                 result[start:end, start:end] = bi[j % bi.shape[0], :end - start, :end - start] @ block
#
#     # Return only the part corresponding to original dimensions
#     return result[:out_dim, :out_dim].to(x.dtype)
