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
def merge_layers(
    a: Parameter(Tensor, "weight"),
    b: Parameter(Tensor, "weight"),
    *,
    alpha: Parameter(float) = 0.5,
    corr_threshold: Parameter(float) = 0.5,
    early_exit: Parameter(bool) = True,
    **kwargs,
) -> Return(Tensor, "weight"):
    cache = kwargs["cache"]
    key = kwargs["key"]

    if key:
        logger.debug("[merge_layers] Key: %s -- Using alpha: %.4f", key, alpha)

    # --- ADDED NaN/Inf CHECK ---
    a_is_finite = torch.isfinite(a).all()
    b_is_finite = torch.isfinite(b).all()

    if not a_is_finite or not b_is_finite:
        warning_msg = f"({key}): Non-finite values detected in input tensors! "
        if not a_is_finite:
            warning_msg += "Input 'a' has NaNs/Infs. "
        if not b_is_finite:
            warning_msg += "Input 'b' has NaNs/Infs. "
        warning_msg += "Returning input 'a' as fallback."
        logger.warning(warning_msg)
        return a  # Return tensor 'a'
    # --- END OF NaN/Inf CHECK ---

    # Early exit if alpha is 0.0 and flag is above 0.0
    if early_exit and alpha == 0.0:  # Changed from: early_exit > 0.0
        return a

    if cache is not None:
        if key not in cache:
            cache[key] = {}
        layer_cache = cache[key]
    else:
        layer_cache = None

    layer_type = get_layer_type(a.shape, kwargs)

    if layer_type == LayerType.SCALAR:
        return geometric_sum_full.__wrapped__(a, b, alpha=alpha)
    elif layer_type == LayerType.OFFSET:
        return torch.lerp(a, b, alpha)
    elif layer_type == LayerType.EMBEDD:
        return clip_embedding_merge_v3(a, b, alpha=alpha)
    elif layer_type == LayerType.CROSS_ATTENTION_QKV:
        return merge_cross_attention_qkv(a, b, alpha=alpha, key=key, cache=layer_cache)
    elif layer_type == LayerType.ATTENTION_QKV:
        return merge_self_attention_qkv(a, b, alpha, key=key, cache=layer_cache)
    elif layer_type == LayerType.ATTENTION_PROJ:
        return merge_attention_output(a, b, alpha, key=key, cache=layer_cache)
    elif layer_type == LayerType.FFN_PROJ:
        return merge_ffn_proj(a, b, alpha=alpha, key=key)
    elif layer_type == LayerType.FFN_OUT:
        return merge_ffn_out(a, b, alpha=alpha, corr_threshold=corr_threshold, cache=layer_cache)
    elif layer_type == LayerType.MATMUL:
        return polar_decomposition(a, b, alpha=alpha, cache=layer_cache)
    elif layer_type == LayerType.CONV2D:
        return merge_wavelets(a, b, alpha=alpha)
    else:
        return torch.lerp(a, b, alpha)

def polar_decomposition(
    a: Tensor, b: Tensor, alpha: float, regularization_eps: float = 1e-6, cache: dict | None = None, key_prefix: str = "polar"
) -> Tensor:
    """
    Interpolate between tensors using polar decomposition.
    Decomposes each tensor into orthogonal and positive semidefinite parts,
    then interpolates each part separately.
    """
    device, dtype, original_shape = a.device, a.dtype, a.shape

    if not original_shape:
        shape_2d = (1, 1)
    elif len(a.shape) == 4:
        shape_2d = (a.shape[0], functools.reduce(operator.mul, a.shape[1:]))
    else:
        shape_2d = (a.shape[0] if len(a.shape) > 1 else 1, a.shape[-1])
    a_2d, b_2d = a.reshape(*shape_2d), b.reshape(*shape_2d)

    def get_cached_svd(matrix: torch.Tensor, name_suffix: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        svd_cache_key_prefix = f"{key_prefix}_{name_suffix}"
        u_svd, s_svd, vt_svd = _get_standard_cached_svd(matrix, cache, svd_cache_key_prefix, device, dtype)
        u_polar = u_svd @ vt_svd  # Orthogonal factor (closest orthogonal matrix)
        return u_polar, s_svd, vt_svd

    u_a_polar, s_a, vt_a = get_cached_svd(a_2d, "a")
    u_b_polar, s_b, vt_b = get_cached_svd(b_2d, "b")

    # Align orthogonal factors using Procrustes
    transform_cache_key = f"{key_prefix}_transform"
    if cache is not None and transform_cache_key in cache:
        transform = cache[transform_cache_key].to(device, dtype)
    else:
        transform = orthogonal_procrustes_ml(u_a_polar, u_b_polar)
        if cache is not None:
            cache[transform_cache_key] = transform.to("cpu")

    u_b_polar_aligned = u_b_polar @ transform

    # Construct positive semidefinite factors with regularization
    p_a = vt_a.T @ torch.diag(s_a + regularization_eps) @ vt_a
    p_b = vt_b.T @ torch.diag(s_b + regularization_eps) @ vt_b

    # SLERP on orthogonal factors (routing based on efficiency for matrix dimensions)
    M_polar, N_polar = u_a_polar.shape
    slerp_sub_cache_key = f"{key_prefix}_slerp_cache"
    slerp_internal_cache = cache.get(slerp_sub_cache_key, {}) if cache is not None else {}

    # Note: Routing choice is for computational efficiency
    if N_polar > M_polar:  # Wide matrices
        merged_u = slerp_grassmann(
            u_a_polar, u_b_polar_aligned, alpha, cache=slerp_internal_cache, key_prefix=f"{key_prefix}_grassmann"
        )
    else:  # Tall or square matrices
        merged_u = slerp_stiefel(
            u_a_polar, u_b_polar_aligned, alpha, cache=slerp_internal_cache, key_prefix=f"{key_prefix}_stiefel"
        )

    if cache is not None and slerp_internal_cache:
        cache[slerp_sub_cache_key] = slerp_internal_cache

    # LERP on positive factors
    merged_p = torch.lerp(p_a, p_b, alpha)

    # Reconstruct result
    result = (merged_u @ merged_p).reshape(original_shape)

    return result

def slerp_grassmann(  # Version 1.0 from user, minimally modified
    u_a: Tensor, u_b: Tensor, alpha: float, cache: dict | None = None, key_prefix: str = "grassmann"
) -> Tensor:
    # Based on Edelman, Arias, Smith (1998) "The Geometry of Algorithms with Orthogonality Constraints", Eq (2.4)
    # Adapted for U_A, U_B being M x N with Orthonormal Rows (ONR)
    # Original formula is for N x P with Orthonormal Columns (ONC)

    if alpha == 0.0:
        return u_a
    if alpha == 1.0:
        return u_b

    if torch.allclose(u_a, u_b, atol=1e-6):
        return u_a

    device, dtype, M, N = u_a.device, u_a.dtype, u_a.shape[0], u_a.shape[1]

    if M == N:  # Square matrix - delegate to specialized function
        return slerp_square_unitary(u_a, u_b, alpha, cache=cache, key_prefix=f"{key_prefix}_as_sq_unitary")

    C_matrix = u_a @ u_b.T  # M x M

    svd_C_key_v, svd_C_key_s, svd_C_key_w_t = f"{key_prefix}_svd_C_v", f"{key_prefix}_svd_C_s", f"{key_prefix}_svd_C_w_t"
    if cache is not None and svd_C_key_v in cache:
        v_c, s_c_diag, w_c_t = (
            cache[svd_C_key_v].to(device, dtype),
            cache[svd_C_key_s].to(device, dtype),
            cache[svd_C_key_w_t].to(device, dtype),
        )
    else:
        svd_driver = "gesvda" if u_a.is_cuda else None
        v_c, s_c_diag, w_c_t = torch.linalg.svd(C_matrix, driver=svd_driver)
        if cache is not None:
            cache[svd_C_key_v], cache[svd_C_key_s], cache[svd_C_key_w_t] = v_c.cpu(), s_c_diag.cpu(), w_c_t.cpu()

    s_c_diag_clamped = torch.clamp(s_c_diag, -1.0 + EPSILON, 1.0 - EPSILON)
    theta_s = torch.acos(s_c_diag_clamped)

    q_onc_key = f"{key_prefix}_q_onc"
    if cache is not None and q_onc_key in cache:
        q_onc = cache[q_onc_key].to(device, dtype)
    else:
        identity_N = torch.eye(N, device=device, dtype=dtype)
        q_factor_cols = (identity_N - u_a.T @ u_a) @ u_b.T  # N x M
        q_factor_norm = torch.norm(q_factor_cols)
        if q_factor_norm < EPSILON * 100:  # Numerical stability: avoid QR issues
            q_onc = torch.zeros_like(q_factor_cols)
        else:
            q_onc, _ = torch.linalg.qr(q_factor_cols)  # mode='reduced' is default
        if cache is not None:
            cache[q_onc_key] = q_onc.cpu()

    cos_interp_theta, sin_interp_theta = torch.cos(alpha * theta_s), torch.sin(alpha * theta_s)
    term1_cols = u_a.T @ w_c_t.T @ torch.diag(cos_interp_theta) @ v_c.T
    term2_cols = q_onc @ torch.diag(sin_interp_theta) @ v_c.T
    x_interp_cols = term1_cols + term2_cols

    u_interp = x_interp_cols.T

    return u_interp

def slerp_stiefel(a: Tensor, b: Tensor, alpha: float, cache: dict | None = None, key_prefix: str = "stiefel") -> Tensor:
    """Complete Stiefel manifold interpolation"""
    if alpha == 0.0:
        return a
    if alpha == 1.0:
        return b

    m, n = a.shape
    if m == n:
        return slerp_square_unitary(a, b, alpha, cache=cache, key_prefix=f"{key_prefix}_as_sq_unitary")

    if torch.allclose(a, b, atol=1e-6):
        return a

    try:  # Primary path with proper caching
        tangent_vector = log_stiefel(a, b, cache=cache, key_prefix=f"{key_prefix}_log")
        scaled_tangent = alpha * tangent_vector
        result = exp_stiefel(a, scaled_tangent)
        return result
    except Exception as e_logexp:  # Fallback path
        logger.warning(
            "slerp_stiefel fallback triggered for %s. Reason: %s. Using direct SVD method.",
            key_prefix,
            type(e_logexp).__name__,
        )

        svd_driver = "gesvda" if a.is_cuda else None
        u, s, vt = torch.linalg.svd(a.T @ b, driver=svd_driver, full_matrices=False)
        s_clamped = torch.clamp(s, -1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(s_clamped)

        # More efficient calculation but keeping the corrected formula
        theta_interp = alpha * theta  # Corrected: alpha*theta for interp
        cos_interp = torch.cos(theta_interp)
        sin_interp = torch.sin(theta_interp)

        y = b - a @ (a.T @ b)
        q, _ = torch.linalg.qr(y)
        result = a @ (u @ torch.diag(cos_interp) @ vt) + q @ (u @ torch.diag(sin_interp) @ vt)

        return result

def log_stiefel(a, b, tau=None, max_iter=30, cache: dict | None = None, key_prefix: str = "log_stiefel"):
    assert max_iter >= 1

    log_stiefel_key = f"{key_prefix}_result_log_stiefel_cpu"
    if cache is not None and log_stiefel_key in cache:
        return cache[log_stiefel_key].to(a.device, a.dtype)

    n, p = a.shape
    device, dtype = a.device, a.dtype
    tau = tau or 100 * torch.finfo(dtype).eps * p

    m_mat = a.T @ b
    b_minus_am = b - a @ m_mat
    q_orth, n_mat = qr_pos(b_minus_am)

    v_intermediate_cat = torch.cat((m_mat, n_mat), dim=0)
    v = orthogonal_complete(v_intermediate_cat)

    r_svd, sigma_svd, r_hat_t_svd = torch.linalg.svd(v[p:, p:], driver="gesvda" if v.is_cuda else None)
    q_orth @= r_svd
    v[p:, :p] = r_svd.T @ n_mat
    v[:p, p:] @= r_hat_t_svd.T
    p_arange = torch.arange(p, 2 * p, device=device)
    v[p:, p:].zero_()
    v[p_arange, p_arange] = sigma_svd
    del r_svd, sigma_svd, r_hat_t_svd, p_arange

    for i in range(max_iter):
        l_iter = logm(v)
        c_syl, lh_block = l_iter[p:, p:], l_iter[p:, :p]
        c_norm = torch.linalg.matrix_norm(c_syl)
        if c_norm <= tau:
            break
        s_syl = (lh_block @ lh_block.mH) / 12.0 - torch.eye(p, device=device, dtype=dtype) / 2.0
        g_syl = solve_symmetric_sylvester(s_syl, c_syl)
        v_update_exp = torch.linalg.matrix_exp(g_syl)
        v[:, p:] @= v_update_exp

    # Robust final computation - always recompute after iterations
    l_final = logm(v)
    delta = a @ l_final[:p, :p] + q_orth @ l_final[p:, :p]
    if cache is not None:
        cache[log_stiefel_key] = delta.to("cpu")

    return delta

def logm(m: Tensor) -> Tensor:  # Removed unused key_prefix
    """Matrix logarithm using eigendecomposition with numerically stable reconstruction"""
    original_dtype = m.dtype

    # Promote to complex for eig if real, as eigenvalues/vectors can be complex
    compute_dtype = m.dtype if m.is_complex() else (torch.complex64 if m.dtype == torch.float32 else torch.complex128)
    m_c = m.to(compute_dtype)

    eigenvalues, eigenvectors_V = torch.linalg.eig(m_c)
    log_eigenvalues = torch.log(eigenvalues)

    # vs * v_log broadcasts to vs @ diag(v_log), then solve gives diag(v_log) @ vs^-1
    res_c = torch.linalg.solve(eigenvectors_V, eigenvectors_V * log_eigenvalues, left=False)

    if not m.is_complex():  # Original was real
        if torch.is_complex(res_c) and res_c.imag.abs().max() > EPSILON * 1000:
            # Significant imaginary part - this shouldn't happen for real orthogonal matrices
            pass  # Could add warning here
        res = res_c.real.to(original_dtype)
    else:  # Original was complex, keep complex result
        res = res_c.to(original_dtype)

    return res

def orthogonal_complete(q: Tensor) -> Tensor:
    """Complete matrix q to full orthogonal basis (orthonormalizes q first if needed)"""
    n, k = q.shape
    if n <= k:
        q_ortho, _ = torch.linalg.qr(q)  # Ensure orthonormal
        return q_ortho

    # Orthonormalize input first
    q_ortho, _ = torch.linalg.qr(q)

    # Project identity matrix columns onto orthogonal complement
    identity_cols = torch.eye(n, device=q.device, dtype=q.dtype)[:, k:]
    projected = identity_cols - q_ortho @ (q_ortho.T @ identity_cols)

    q2 = torch.linalg.householder_product(*torch.linalg.qr(projected, mode="raw"))

    return torch.cat([q_ortho, q2[:, : n - k]], dim=1)

def solve_symmetric_sylvester(s, c):
    """Solve symmetric Sylvester equation AX + XA = C where A=s, C=c"""
    v, vs = torch.linalg.eigh(s)  # s is symmetric/Hermitian

    # Transform to diagonal coordinates: vs.mH @ c @ vs
    c_t = vs.mH @ c @ vs

    # Denominator matrix: λ_i + λ_j for all pairs
    d = v.unsqueeze(0) + v.unsqueeze(1)

    # Check for singularity (λ_i + λ_j ≈ 0)
    if torch.any(torch.abs(d) < 1e-12):
        logger.warning("Singular Sylvester operator: some λ_i+λ_j ≈ 0")
        # Could regularize with: d[torch.abs(d) < 1e-12] = 1e-12 * torch.sign(d[torch.abs(d) < 1e-12])

    # Solve in diagonal coordinates
    g_t = c_t / d

    # Transform back: vs @ g_t @ vs.mH
    g = vs @ g_t @ vs.mH  # Fixed: consistent use of .mH

    return g

def qr_pos(a: Tensor) -> tuple[Tensor, Tensor]:
    """QR decomposition with positive diagonal in R"""
    q, r = torch.linalg.qr(a)

    d = torch.diagonal(r, dim1=-2, dim2=-1)
    ph = d.sign()

    # Handle zero diagonal elements (critical for numerical stability)
    ph[ph == 0] = 1.0

    # Scale Q columns and R rows to make R diagonal positive
    q *= ph.unsqueeze(-2)  # ph broadcasts across rows (M dimension)
    r *= ph.unsqueeze(-1)  # ph broadcasts across columns (N dimension)

    return q, r

def exp_stiefel(a: Tensor, delta: Tensor) -> torch.Tensor:
    """Exponential map on Stiefel manifold: exp_a(delta)"""
    n, p = a.shape

    # Construct augmented matrix [a, delta]
    augmented = torch.cat([a, delta], dim=1)  # Shape: (n, 2p)
    q, r = torch.linalg.qr(augmented)  # q: (n, min(n,2p)), r: (min(n,2p), 2p)

    # Extract blocks safely
    q1 = q[:, :p]  # First p columns

    # Handle q2 extraction for edge cases
    if q.shape[1] >= 2 * p:
        q2 = q[:, p : 2 * p]
    else:
        # When n < 2p, pad q2 with zeros
        q2 = torch.zeros(n, p, device=a.device, dtype=a.dtype)
        available_cols = q.shape[1] - p
        if available_cols > 0:
            q2[:, :available_cols] = q[:, p : p + available_cols]

    # Extract R blocks safely
    r12 = r[:p, p : 2 * p]

    # Handle r22 extraction for edge cases
    min_dim = min(r.shape[0], 2 * p)
    if min_dim >= 2 * p:
        r22 = r[p : 2 * p, p : 2 * p]
    else:
        # Degenerate case: pad with identity to avoid singularity
        r22 = torch.eye(p, device=a.device, dtype=a.dtype)
        if min_dim > p:
            actual_size = min_dim - p
            r22[:actual_size, :actual_size] = r[p:min_dim, p : 2 * p][:, :actual_size]

    # Solve for k with fallback
    try:
        k = torch.linalg.solve(r22, r12.T)
    except Exception:
        k = torch.linalg.pinv(r22) @ r12.T

    # Construct skew-symmetric matrix
    m = torch.zeros(2 * p, 2 * p, device=a.device, dtype=a.dtype)
    m[:p, p:] = k
    m[p:, :p] = -k.T

    # Matrix exponential and result
    exp_m = torch.linalg.matrix_exp(m)
    result = q1 @ exp_m[:p, :p] + q2 @ exp_m[p:, :p]

    return result

def _matrix_logarithm_eig(matrix: torch.Tensor, cache: dict | None = None, key_prefix: str = "logm_eig_default") -> torch.Tensor:
    log_eig_cache_key = f"{key_prefix}_result_cpu"

    if cache is not None and log_eig_cache_key in cache:
        cached_log_A = cache[log_eig_cache_key].to(device=matrix.device, dtype=matrix.dtype)
        return cached_log_A

    if not (matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]):
        raise ValueError(f"Matrix logarithm expects a square matrix. Got shape: {matrix.shape}")

    original_dtype = matrix.dtype
    compute_dtype = matrix.dtype if matrix.is_complex() else (torch.complex64 if matrix.dtype == torch.float32 else torch.complex128)
    matrix_c = matrix.to(compute_dtype)

    eigenvalues, eigenvectors_V = torch.linalg.eig(matrix_c)
    log_eigenvalues = torch.log(eigenvalues)

    # FIXED: Use numerically stable approach instead of explicit inverse
    log_A_complex = torch.linalg.solve(eigenvectors_V, eigenvectors_V * log_eigenvalues, left=False)

    # Rest of dtype handling stays the same...
    if not matrix.is_complex():
        if torch.is_complex(log_A_complex) and log_A_complex.imag.abs().max() > EPSILON * 1000:
            pass
        log_A_final = log_A_complex.real.to(original_dtype)
    else:
        log_A_final = log_A_complex.to(original_dtype)

    if cache is not None:
        cache[log_eig_cache_key] = log_A_final.cpu()

    return log_A_final

def slerp_square_unitary(
    A: torch.Tensor, B: torch.Tensor, alpha: float, cache: dict | None = None, key_prefix: str = "sq_unitary_default"
) -> torch.Tensor:
    """
    SLERP for square unitary/orthogonal matrices using matrix logarithm.
    Caches the expensive, alpha-independent matrix logarithm computation.
    """
    if alpha == 0.0:
        return A
    if alpha == 1.0:
        return B
    if torch.allclose(A, B, atol=1e-6):
        return A

    device, original_dtype = A.device, A.dtype
    compute_c_dtype = torch.complex64 if original_dtype in [torch.float32, torch.complex64] else torch.complex128
    was_real_input = not A.is_complex()
    A_c, B_c = A.to(compute_c_dtype), B.to(compute_c_dtype)

    try:
        relative_rotation = B_c @ A_c.mH

        # Cache the expensive matrix logarithm (alpha-independent)
        log_R_key_prefix_for_helper = f"{key_prefix}_log_rel_rot"
        log_R = _matrix_logarithm_eig(relative_rotation, cache=cache, key_prefix=log_R_key_prefix_for_helper)

        # Project to skew-symmetric/skew-Hermitian
        if was_real_input:
            log_R_proj = (log_R - log_R.T) / 2.0
        else:
            log_R_proj = (log_R - log_R.mH) / 2.0

        # Alpha-dependent computation (not cached)
        interpolated_log = alpha * log_R_proj
        delta_rotation = torch.linalg.matrix_exp(interpolated_log)
        interpolated_unitary_c = delta_rotation @ A_c

        # Convert back to original dtype
        if was_real_input:
            final_result = interpolated_unitary_c.real.to(original_dtype)
        else:
            final_result = interpolated_unitary_c.to(original_dtype)

        if not torch.isfinite(final_result).all():
            raise RuntimeError(f"slerp_square_unitary produced non-finite result for key_prefix {key_prefix}")

        return final_result

    except Exception as e:  # Fallback for any numerical issues
        logger.warning(
            "slerp_square_unitary fallback triggered for %s. Reason: %s. Using LERP+SVD.",
            key_prefix,
            type(e).__name__,
        )
        lerped_val = torch.lerp(A, B, alpha)
        try:
            u_lerp, _, vh_lerp = torch.linalg.svd(lerped_val, full_matrices=False)
            fallback_result = (u_lerp @ vh_lerp).to(original_dtype)
            return fallback_result
        except Exception as e2:
            logger.warning(
                "SVD fallback also failed for %s. Reason: %s. Using raw LERP.",
                key_prefix,
                type(e2).__name__,
            )
            return lerped_val

def clip_embedding_merge_v3(a: Tensor, b: Tensor, alpha: float = 0.5) -> Tensor:
    """
    CLIP embedding merge focused on preserving directional relationships using orthogonal Procrustes.
    """
    # 1. Normalize embeddings
    a_norm = F.normalize(a, p=2, dim=1)
    b_norm = F.normalize(b, p=2, dim=1)

    # 2. Compute rotation using orthogonal Procrustes
    rotation = orthogonal_procrustes_ml(a_norm, b_norm)  # Replace SVD-based rotation

    # 3. Apply rotation to b to align directional space
    b_aligned = torch.mm(b, rotation.T)

    # 4. Simple interpolation in aligned space
    merged = (1 - alpha) * a + alpha * b_aligned

    # 5. Preserve original norms
    a_norms = torch.norm(a, dim=1, keepdim=True)
    b_norms = torch.norm(b, dim=1, keepdim=True)
    target_norms = (1 - alpha) * a_norms + alpha * b_norms

    current_norms = torch.norm(merged, dim=1, keepdim=True)
    merged = merged * (target_norms / (current_norms + 1e-8))

    return merged

def merge_cross_attention_qkv(a: Tensor, b: Tensor, alpha: float, key: str, cache: dict | None = None) -> Tensor:
    """
    Enhanced merge for cross-attention QKV layers with optimized caching for SVD.
    Handles various architectures and projection types.
    """
    device = a.device
    dtype = a.dtype

    # Handle CLIP-G style concatenated QKV
    if "in_proj" in key:
        head_dim = a.shape[0] // 3
        merged_parts = []

        for i in range(3):
            start = head_dim * i
            end = head_dim * (i + 1)
            part_a = a[start:end]
            part_b = b[start:end]

            # Use polar decomposition for each part with separate cache entries
            part_key = f"{key}_part_{i}"
            part_cache = cache.get(part_key, {}) if cache is not None else None
            merged = polar_decomposition(part_a, part_b, alpha, cache=part_cache)
            if cache is not None:
                cache[part_key] = part_cache

            merged_parts.append(merged)

        return torch.cat(merged_parts, dim=0)

    # Handle regular CLIP text encoder layers
    elif any(x in key for x in ["k_proj", "v_proj", "q_proj"]):
        return merge_self_attention_qkv(a, b, alpha, key)

    # Handle UNet cross-attention
    else:
        # For query projections, calculate `adjusted_alpha` without caching
        if ".to_q." in key:
            with torch.no_grad():
                # Generate some sample data for cosine similarity computation
                x = torch.randn(min(100, a.shape[-1]), a.shape[-1], device=device, dtype=dtype)
                q_a = x @ a.T
                q_b = x @ b.T
                sim = F.cosine_similarity(q_a.flatten(), q_b.flatten(), dim=0)
                adjusted_alpha = alpha * torch.sigmoid(sim * 0.5)

            # Use polar decomposition with adjusted weight
            return polar_decomposition(a, b, alpha=adjusted_alpha.item(), cache=cache)

        # Get cached SVD components for matrices `a` and `b` using the centralized helper
        u_a, s_a, vh_a = _get_standard_cached_svd(a, cache, f"{key}_a", device, dtype)
        u_b, s_b, vh_b = _get_standard_cached_svd(b, cache, f"{key}_b", device, dtype)

        # Interpolate singular values
        s_merged = torch.lerp(s_a, s_b, alpha)

        # Align spaces using the smaller dimension
        k = min(vh_a.shape[0], vh_b.shape[0])

        # Get or compute alignment transform
        transform_key = f"{key}_transform"
        if cache is not None and transform_key in cache:
            R = cache[transform_key].to(device, dtype)
        else:
            R = orthogonal_procrustes_ml(vh_a[:k], vh_b[:k])
            if cache is not None:
                cache[transform_key] = R.to("cpu")

        vh_merged = torch.lerp(vh_a[:k], vh_b[:k] @ R.T, alpha)

        # Reconstruct while preserving cross-modal relationships
        merged = (u_a[:, :k] * s_merged[:k]) @ vh_merged

        # Scale to preserve magnitude
        scale_a = torch.norm(a)
        scale_b = torch.norm(b)
        target_scale = (1 - alpha) * scale_a + alpha * scale_b
        current_scale = torch.norm(merged)

        return merged * (target_scale / (current_scale + 1e-6))

def merge_self_attention_qkv(a: Tensor, b: Tensor, alpha: float, key: str, cache: dict | None = None) -> Tensor:
    """
    Merge self-attention QKV layers with caching for polar decomposition.
    Handles separate Q/K/V and concatenated formats for CLIP-G style models.
    """
    # Handle CLIP-G style concatenated QKV
    if "in_proj" in key:
        head_dim = a.shape[0] // 3
        merged_parts = []

        # Pre-fetch all cache entries to minimize repeated calls to cache.get
        part_caches = [cache.get(f"{key}_part_{i}", {}) if cache else None for i in range(3)]

        for i in range(3):
            start = head_dim * i
            end = head_dim * (i + 1)
            part_a = a[start:end]
            part_b = b[start:end]

            # Use polar decomposition with separate cache namespace for each part
            merged = polar_decomposition(part_a, part_b, alpha, cache=part_caches[i])

            # Update the main cache after polar decomposition call, if caching is enabled
            if cache is not None:
                cache[f"{key}_part_{i}"] = part_caches[i]

            merged_parts.append(merged)

        return torch.cat(merged_parts, dim=0)

    # Handle separate Q/K/V projections
    else:
        # Calculate attention similarity and adjusted alpha (not cached)
        with torch.no_grad():
            x = torch.randn(min(100, a.shape[-1]), a.shape[-1], device=a.device, dtype=a.dtype)
            attn_a = torch.softmax(x @ a.mT / math.sqrt(a.shape[-1]), dim=-1)  # Fix: Use .mT
            attn_b = torch.softmax(x @ b.mT / math.sqrt(b.shape[-1]), dim=-1)  # Fix: Use .mT

            kl_div = F.kl_div(attn_a.log(), attn_b, reduction="batchmean")
            adjusted_alpha = alpha * torch.sigmoid(1.0 - kl_div)

        # Call polar_decomposition without caching, due to dynamic adjusted_alpha
        return polar_decomposition(a, b, alpha=adjusted_alpha.item(), cache=cache)

def merge_attention_output(a: Tensor, b: Tensor, alpha: float, key: str, cache: dict | None = None) -> Tensor:
    """
    Merge attention output projections while preserving output distribution,
    without caching for dynamically adjusted alpha values.
    """
    with torch.no_grad():
        # Generate sample inputs
        x = torch.randn(min(512, a.shape[-1]), a.shape[-1], device=a.device, dtype=a.dtype)

        # Get output representations
        out_a = x @ a.T
        out_b = x @ b.T

        # Compute output statistics
        stats_a = torch.stack(
            [
                out_a.std(dim=0).mean(),  # Feature variation
                out_a.abs().mean(),  # Activation magnitude
                (out_a > 0).float().mean(),  # Activation sparsity
            ]
        )
        stats_b = torch.stack([out_b.std(dim=0).mean(), out_b.abs().mean(), (out_b > 0).float().mean()])

        # Adjust merge weight based on output similarity
        stats_diff = torch.norm(stats_a - stats_b)
        adjusted_alpha = alpha * torch.sigmoid(1.0 - stats_diff)

    # Call polar_decomposition without caching, due to dynamic adjusted_alpha
    merged = polar_decomposition(a, b, alpha=adjusted_alpha.item(), cache=cache)

    # Scale to preserve activation magnitude
    scale_a = torch.norm(out_a) / torch.norm(x)
    scale_b = torch.norm(out_b) / torch.norm(x)
    target_scale = (1 - alpha) * scale_a + alpha * scale_b

    with torch.no_grad():
        current_scale = torch.norm(x @ merged.T) / torch.norm(x)

    return merged * (target_scale / (current_scale + 1e-6))

def merge_ffn_proj(a: Tensor, b: Tensor, alpha: float, key: str) -> torch.Tensor:
    """
    Enhanced FFN projection handling that adapts to matrix size.
    """
    input_dim = a.shape[-1]  # For proj.weight, this would be 640 or 1280
    output_dim = a.shape[0]  # For proj.weight, this would be 5120 or 10240
    expansion_factor = output_dim / input_dim

    if matrix_is_large(a, threshold=2048):  # Adjust threshold as needed
        return merge_ffn_proj_conservative(a, b, alpha, expansion_factor)
    else:
        return merge_ffn_proj_standard(a, b, alpha, expansion_factor)

def merge_ffn_proj_conservative(a: Tensor, b: Tensor, alpha: float, expansion_factor: float) -> Tensor:
    """
    Conservative merging for larger FFN projections
    """
    # Split the large projection into groups
    group_size = a.shape[-1]  # Input dimension
    num_groups = int(expansion_factor)

    # Reshape to handle groups separately
    a_groups = a.reshape(num_groups, -1, a.shape[-1])
    b_groups = b.reshape(num_groups, -1, b.shape[-1])

    merged_groups = []
    for i in range(num_groups):
        # Process each group with attention to activation patterns
        a_group = a_groups[i]
        b_group = b_groups[i]

        # Check activation similarity within group
        with torch.no_grad():
            test_input = torch.randn(min(100, a_group.shape[-1]), a_group.shape[-1], device=a.device).to(
                a.dtype
            )  # Ensure correct data type
            a_act = torch.relu(test_input @ a_group.T)
            b_act = torch.relu(test_input @ b_group.T).to(a.dtype)

            # Compare activation patterns
            similarity = F.cosine_similarity(a_act.flatten(), b_act.flatten(), dim=0)

        if similarity > 0.5:
            # Similar activations - interpolate smoothly
            merged_group = torch.lerp(a_group, b_group, alpha)
        else:
            # Different activations - preserve stronger features
            merged_group = torch.where(torch.abs(a_group) > torch.abs(b_group), a_group, b_group)

        merged_groups.append(merged_group)

    # Recombine groups
    return torch.cat(merged_groups, dim=0)

def merge_ffn_proj_standard(a: Tensor, b: Tensor, alpha: float, expansion_factor: float) -> Tensor:
    """
    Standard merging for smaller FFN projections
    """
    # Normalize matrices
    a_norm = F.normalize(a, dim=-1)
    b_norm = F.normalize(b, dim=-1)

    # Compute activation statistics
    with torch.no_grad():
        test_input = torch.randn(min(100, a.shape[-1]), a.shape[-1], device=a.device).to(a.dtype)  # Cast test_input to a.dtype
        a_act = torch.relu(test_input @ a.T)
        b_act = torch.relu(test_input @ b.T).to(a.dtype)

        # Calculate activation statistics
        a_stats = torch.stack(
            [
                (a_act > 0).float().mean(),  # sparsity
                a_act[a_act > 0].std(),  # activation spread
            ]
        )
        b_stats = torch.stack([(b_act > 0).float().mean(), b_act[b_act > 0].std()])

    # Calculate merge weight based on activation properties
    stats_diff = torch.norm(a_stats - b_stats)
    merge_weight = torch.sigmoid(1.0 - stats_diff) * alpha

    # Interpolate with adjusted weight
    merged = torch.lerp(a_norm, b_norm, merge_weight)

    # Rescale to preserve activation magnitude
    scale_a = torch.norm(a_act) / torch.norm(test_input)
    scale_b = torch.norm(b_act) / torch.norm(test_input)
    target_scale = (1 - alpha) * scale_a + alpha * scale_b
    current_scale = torch.norm(torch.relu(test_input @ merged.T)) / torch.norm(test_input)

    return merged * (target_scale / (current_scale + 1e-6))

def merge_ffn_out(
    a: Tensor, b: Tensor, alpha: float, corr_threshold: float, cache: dict[str, dict[str, Tensor]] | None = None
) -> Tensor:
    """
    Enhanced FFN output merge that preserves feature relationships and activation patterns,
    optimized with caching for SVD and orthogonal Procrustes alignment.
    """
    output_dim, input_dim = a.shape
    device = a.device
    dtype = a.dtype

    # Generate sample activations
    num_samples = min(512, input_dim)
    with torch.no_grad():
        x = torch.randn(num_samples, input_dim, device=device, dtype=dtype)
        x = torch.nn.functional.gelu(x)

        # Get output space representations
        out_a = x @ a.T
        out_b = x @ b.T

        # Compute correlation matrices in output space
        corr_a = torch.corrcoef(out_a.T)
        corr_b = torch.corrcoef(out_b.T)

        # Identify strongly correlated feature groups
        groups_a = []
        groups_b = []
        used_indices = set()

        # Find feature groups in both matrices
        for i in range(output_dim):
            if i in used_indices:
                continue

            # Find correlated features
            group_a = torch.where(torch.abs(corr_a[i]) > corr_threshold)[0]
            group_b = torch.where(torch.abs(corr_b[i]) > corr_threshold)[0]

            if len(group_a) > 1 or len(group_b) > 1:
                # Ensure we don't exceed the actual group size when storing
                actual_size = min(len(group_a), len(group_b))
                groups_a.append(group_a[:actual_size])
                groups_b.append(group_b[:actual_size])
                used_indices.update(group_a[:actual_size].tolist())

        # Initialize merged tensor
        merged = torch.zeros_like(a)

        # Process each feature group
        for group_a, group_b in zip(groups_a, groups_b):
            # Extract relevant slices
            slice_a = a[group_a]
            slice_b = b[group_b]

            # Normalize the slices
            norm_a = torch.norm(slice_a, dim=1, keepdim=True)
            norm_b = torch.norm(slice_b, dim=1, keepdim=True)
            slice_a_norm = slice_a / (norm_a + 1e-8)
            slice_b_norm = slice_b / (norm_b + 1e-8)

            # Get SVD components WITHOUT caching
            # u_a, s_a, v_a = _get_standard_cached_svd(slice_a_norm, cache, f"{group_a}_a", device, dtype)
            # u_b, s_b, v_b = _get_standard_cached_svd(slice_b_norm, cache, f"{group_b}_b", device, dtype)

            # Direct SVD computation without caching
            svd_driver = "gesvdj" if slice_a_norm.is_cuda else "none"
            u_a, s_a, v_a = torch.linalg.svd(slice_a_norm, full_matrices=False, driver=svd_driver)
            u_b, s_b, v_b = torch.linalg.svd(slice_b_norm, full_matrices=False, driver=svd_driver)

            # Use minimum number of components for alignment
            k = min(v_a.shape[1], v_b.shape[1])

            # Use orthogonal Procrustes for alignment WITHOUT caching
            if k > 0:
                # procrustes_key = f"procrustes_{len(group_a)}_{len(group_b)}"
                # if cache is not None and procrustes_key in cache:
                #     r = cache[procrustes_key].to(device, dtype)
                # else:
                #     r = orthogonal_procrustes_ml(v_a[:, :k], v_b[:, :k])
                #     if cache is not None:
                #         cache[procrustes_key] = r.cpu()

                # Direct Procrustes computation without caching
                r = orthogonal_procrustes_ml(v_a[:, :k], v_b[:, :k])
                v_b_aligned = v_b[:, :k] @ r.T
            else:
                v_b_aligned = v_b[:, :k]

            # Align and interpolate
            v_merged = torch.lerp(v_a[:, :k], v_b_aligned, alpha)
            s_merged = torch.exp((1 - alpha) * torch.log(s_a[:k] + 1e-8) + alpha * torch.log(s_b[:k] + 1e-8))

            # Interpolate norms
            norm_merged = (1 - alpha) * norm_a + alpha * norm_b

            # Reconstruct and check shape before assignment
            group_result = (u_a[:, :k] * s_merged.unsqueeze(0)) @ v_merged * norm_merged

            # Ensure the reconstructed group_result has the correct shape for assignment
            expected_shape = merged[group_a].shape
            if group_result.shape != expected_shape:
                # Apply padding or trimming to match expected shape
                if group_result.shape[0] < expected_shape[0]:
                    # Pad group_result to match the expected shape
                    padding = (0, 0, 0, expected_shape[0] - group_result.shape[0])
                    group_result = torch.nn.functional.pad(group_result, padding)
                elif group_result.shape[0] > expected_shape[0]:
                    # Trim group_result to match the expected shape
                    group_result = group_result[: expected_shape[0]]

            merged[group_a] = group_result

        # Handle uncorrelated features
        uncorrelated = list(set(range(output_dim)) - used_indices)
        if uncorrelated:
            merged[uncorrelated] = torch.lerp(a[uncorrelated], b[uncorrelated], alpha)

        # Scale adjustment
        with torch.no_grad():
            out_merged = x @ merged.T
            scale_a = torch.norm(out_a) / torch.norm(x)
            scale_b = torch.norm(out_b) / torch.norm(x)
            target_scale = (1 - alpha) * scale_a + alpha * scale_b
            current_scale = torch.norm(out_merged) / torch.norm(x)
            merged = merged * (target_scale / (current_scale + 1e-8))

    return merged

@merge_method
def geometric_sum_full(
    a: Parameter(Tensor, "weight"), b: Parameter(Tensor, "weight"), alpha: Parameter(Tensor) = 0.5, **kwargs
) -> Return(Tensor, "weight"):
    key = kwargs["key"]
    if key:
        logger.debug("[geosum] Key: %s -- Using alpha: %.4f", key, alpha)
    a = torch.complex(a, torch.zeros_like(a))
    b = torch.complex(b, torch.zeros_like(b))
    res = a ** (1 - alpha) * b**alpha
    return res.real

def merge_conv_wavelets(
    a: Tensor,
    b: Tensor,
    alpha: float,
    wave: str = "db4",
    level: int | None = None,
    mode: str = "zero",
    compute_dtype: torch.dtype | None = torch.float32,
) -> Tensor:
    """
    Merges two convolutional layers using a multi-level wavelet transform
    while attempting to preserve original sizes. Kernels are reshaped to 2D
    before the transform, and explicit padding is removed.

    Args:
    - a, b: Input tensors (convolutional kernels)
    - alpha: Blending factor (0 to 1)
    - wave: Wavelet to use (default: 'db4')
    - level: Number of decomposition levels
    - mode:
    - compute_dtype:
    """
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    if a.ndim != 4:
        raise ValueError(f"Expected conv weight tensor [O,I,kH,kW], got {a.shape}")

    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1]")

    O, I, kH, kW = a.shape
    if (kH, kW) not in [(1, 1), (3, 3)]:
        raise ValueError(f"Only 1x1 and 3x3 supported here, got {kH}x{kW}")

    # Choose a level that always makes sense for these tiny kernels.
    # 1x1: no meaningful decomposition -> just linear blend.
    if (kH, kW) == (1, 1):
        return alpha * a + (1.0 - alpha) * b

    # 3x3: one level is the only practical choice.
    if level is None:
        level = 1
    level = int(level)
    if level < 1:
        # treat as no decomposition
        return alpha * a + (1.0 - alpha) * b

    # Compute in fp32 by default, cast back at end
    orig_dtype = a.dtype
    work_dtype = compute_dtype if compute_dtype is not None else orig_dtype

    A = a.to(dtype=work_dtype)
    B = b.to(dtype=work_dtype)

    # Batch all (O*I) kernels as images: [N, 1, 3, 3]
    A_img = A.reshape(O * I, 1, kH, kW)
    B_img = B.reshape(O * I, 1, kH, kW)

    # wavedec2 returns [cA_n, (cH_n,cV_n,cD_n), ...] in pywt order [page:2]
    ca = ptwt.wavedec2(A_img, wave, level=level, mode=mode)
    cb = ptwt.wavedec2(B_img, wave, level=level, mode=mode)  # mode supports "reflect/zero/constant/periodic" [page:2]

    merged = [alpha * ca[0] + (1.0 - alpha) * cb[0]]
    for da, db in zip(ca[1:], cb[1:]):
        merged.append(tuple(alpha * xa + (1.0 - alpha) * xb for xa, xb in zip(da, db)))

    out = ptwt.waverec2(merged, wave)  # [page:2]

    # Safety: ensure exact kernel size (padding modes can yield off-by-1 in some toolchains)
    out = out[..., :kH, :kW]

    out_w = out.reshape(O, I, kH, kW).to(dtype=orig_dtype)
    return out_w

def get_layer_type(shape, kwargs):
    key = kwargs["key"]

    # Prioritize checks for bias and other specific types
    if key.endswith(".bias") or "bias" in key:
        return LayerType.OFFSET

    # Layer Norms
    elif (
        any(x in key for x in [".norm", "layer_norm", "ln_final", "ln_1", "ln_2", "layer_norm1", "layer_norm2", "final_layer_norm"])
        or "norm" in key
        or "logit_scale" in key
        or "position_ids" in key
        or ".in_layers.0.weight" in key
        or ".out_layers.0.weight" in key
    ):
        return LayerType.SCALAR

    # True embeddings (vocabulary mappings)
    elif "token_embedding" in key or "position_embedding" in key or "positional_embedding" in key or "shared.weight" in key:
        return LayerType.EMBEDD

    # Check for attention layers first
    elif any(x in key for x in [".to_q.", ".to_k.", ".to_v.", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", ".in_proj_"]):
        # Add cross-attention check
        if ".attn2." in key:
            return LayerType.CROSS_ATTENTION_QKV
        return LayerType.ATTENTION_QKV

    # Attention Projection (output projection in both CLIP-G and CLIP-L)
    elif any(x in key for x in [".to_out.", ".out_proj"]) and ".weight" in key:
        return LayerType.ATTENTION_PROJ

    # Feed Forward Network (FFN) in Stable Diffusion layers
    elif ".ff.net." in key and ".proj." in key:
        return LayerType.FFN_PROJ
    elif ".ff.net." in key and ".weight" in key:
        return LayerType.FFN_OUT

    # Feed Forward Network (FFN) in CLIP-G and CLIP-L
    elif "mlp.c_fc" in key and ".weight" in key:
        return LayerType.FFN_PROJ
    elif "mlp.c_proj" in key and ".weight" in key:
        return LayerType.FFN_OUT
    elif "mlp.fc1" in key and ".weight" in key:
        return LayerType.FFN_PROJ
    elif "mlp.fc2" in key and ".weight" in key:
        return LayerType.FFN_OUT

    # Matrix Transformation for Embedding-Like Layers (positional embeddings, projections)
    elif any(x in key for x in ["positional_embedding", "text_projection", "label_emb"]):
        return LayerType.MATMUL

    # Convolutional Layers
    elif len(shape) == 4:
        return LayerType.CONV2D

    # Default to matrix transformations
    return LayerType.MATMUL

class LayerType(enum.Enum):
    SCALAR = enum.auto()
    OFFSET = enum.auto()
    CONV2D = enum.auto()
    EMBEDD = enum.auto()
    MATMUL = enum.auto()
    ATTENTION_QKV = enum.auto()
    CROSS_ATTENTION_QKV = enum.auto()  # New type
    ATTENTION_PROJ = enum.auto()
    FFN_PROJ = enum.auto()
    FFN_OUT = enum.auto()

def matrix_is_large(a: Tensor, threshold: int = 1280) -> bool:
    """
    Determines if a matrix is considered "large" based on its dimensions.

    Args:
        A: The input matrix.
        threshold: The threshold for the minimum dimension size to be considered "large."

    Returns:
        True if the matrix is considered large, False otherwise.
    """
    if a.ndim < 2:  # Check if tensor has fewer than 2 dimensions
        return False  # Treat non-2D tensors as "not large"
    m, n = a.shape  # Get the matrix dimensions
    return m >= threshold or n >= threshold  # Check if either dimension exceeds the threshold

def dominant_rotation(a: Tensor, threshold: float = 0.8) -> bool:
    """
    Estimates if a matrix primarily represents a rotation based on its singular values.

    Args:
        A: The input matrix.
        threshold: The threshold for the ratio of the largest singular value to the smallest
                    singular value to be considered "dominant rotation."

    Returns:
        True if the matrix is estimated to have a dominant rotation, False otherwise.
    """
    _, s, _ = torch.linalg.svd(a)  # Compute the singular values of the matrix
    largest_singular_value = s[0]
    smallest_singular_value = s[-1]
    return largest_singular_value / smallest_singular_value >= threshold

def matrix_is_ill_conditioned(a: Tensor, threshold: float = 100) -> bool:
    """
    Determines if a matrix is ill-conditioned based on its condition number.

    Args:
        A: The input matrix.
        threshold: The threshold for the condition number to be considered ill-conditioned.

    Returns:
        True if the matrix is ill-conditioned, False otherwise.
    """
    condition_number = torch.linalg.cond(a)  # Compute the condition number
    return condition_number >= threshold

def orthogonal_procrustes_ml(a, b, cancel_reflection: bool = False):
    # a is u_a_polar, b is u_b_polar
    atb = a.T @ b

    use_lowrank = not cancel_reflection and a.shape[0] + 10 < a.shape[1]

    if use_lowrank:
        svd_driver = "gesvdj" if a.is_cuda else None
        # NEW torch_svd_lowrank returns U, S, Vh_approx
        u_approx, _, vh_approx = torch_svd_lowrank(  # <--- vh_approx IS Vh
            atb,
            q=a.shape[0] + 10,
            driver=svd_driver,
            full_matrices=False,  # Start with False to mimic old V dim if that helps isolate
        )
        # The Procrustes solution R = U @ Vh
        transform = u_approx @ vh_approx  # <--- USE vh_approx DIRECTLY

    else:  # Standard SVD path (this part was already correct)
        svd_driver = "gesvdj" if a.is_cuda else None
        u_full, _, vh_full = torch.linalg.svd(atb, driver=svd_driver)  # vh_full is V_transpose

        final_u_for_transform = u_full
        final_vh_for_transform = vh_full  # Renamed for clarity
        if cancel_reflection:
            final_u_for_transform[:, -1] *= torch.sign(torch.det(final_u_for_transform) * torch.det(final_vh_for_transform))

        transform = final_u_for_transform @ final_vh_for_transform  # U @ Vh

    if not torch.isfinite(transform).all():
        raise ValueError("Orthogonal Procrustes transform is not finite.")
    return transform

def _get_standard_cached_svd(
    matrix: Tensor, cache: dict | None, prefix: str, device: torch.device, dtype: torch.dtype
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Helper to handle standard SVD caching (u, s, vh).
    """
    cache_key_u = f"{prefix}_u"
    cache_key_s = f"{prefix}_s"
    cache_key_vh = f"{prefix}_vh"

    if cache is not None and cache_key_u in cache:
        u = cache[cache_key_u].to(device, dtype)
        s = cache[cache_key_s].to(device, dtype)
        vh = cache[cache_key_vh].to(device, dtype)
    else:
        svd_driver = "gesvdj" if matrix.is_cuda else "none"
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False, driver=svd_driver)

        if cache is not None:
            cache[cache_key_u] = u.to("cpu")
            cache[cache_key_s] = s.to("cpu")
            cache[cache_key_vh] = vh.to("cpu")

    return u, s, vh

# @merge_method
# def merge_layers(
#         a: Parameter(Tensor, "weight"),
#         b: Parameter(Tensor, "weight"),
#         *,
#         alpha: Parameter(Tensor) =0.0,
#         corr_threshold: Parameter(Tensor) =0.5,
#         early_exit: float = 1.0,  # New flag
#
#         **kwargs,
# ) -> Return(Tensor):
#     key = kwargs["key"]
#
#     # Early exit if alpha is 0.0 and flag is above 0.0
#     if early_exit > 0.0 and alpha == 0.0:
#         return a
#
#     if cache is not None:
#         if key not in cache:
#             cache[key] = {}
#         layer_cache = cache[key]
#     else:
#         layer_cache = None
#
#     layer_type = get_layer_type(a.shape, kwargs)
#
#     if layer_type == LayerType.SCALAR:
#         return geometric_sum_full.__wrapped__(a, b, alpha=alpha)
#     elif layer_type == LayerType.OFFSET:
#         return sd_mecha.weighted_sum.__wrapped__(a, b, alpha=alpha)
#     elif layer_type == LayerType.EMBEDD:
#         return clip_embedding_merge_v3(a, b, alpha=alpha)
#     elif layer_type == LayerType.CROSS_ATTENTION_QKV:
#         return merge_cross_attention_qkv(a, b, alpha=alpha, key=key, cache=layer_cache)
#     elif layer_type == LayerType.ATTENTION_QKV:
#         return merge_self_attention_qkv(a, b, alpha, key=key, cache=layer_cache)
#     elif layer_type == LayerType.ATTENTION_PROJ:
#         return merge_attention_output(a, b, alpha, key=key, cache=layer_cache)
#     elif layer_type == LayerType.FFN_PROJ:
#         return merge_ffn_proj(a, b, alpha=alpha, key=key)
#     elif layer_type == LayerType.FFN_OUT:
#         return merge_ffn_out(a, b, alpha=alpha, corr_threshold=corr_threshold, cache=layer_cache)
#     elif layer_type == LayerType.MATMUL:
#         return polar_decomposition(a, b, alpha=alpha, cache=layer_cache)
#     elif layer_type == LayerType.CONV2D:
#         return merge_wavelets(a, b, alpha=alpha)
#     else:
#         return sd_mecha.weighted_sum.__wrapped__(a, b, alpha=alpha)
#
# def polar_decomposition(a: Tensor, b: Tensor, alpha: float,
#                         regularization_eps: float = 1e-6,
#                         cache: Optional[Dict] = None) -> Tensor:
#     device = a.device
#     dtype = a.dtype
#     original_shape = a.shape
#
#     if not original_shape:
#         shape_2d = (1, 1)
#     elif len(a.shape) == 4:
#         shape_2d = (-1, functools.reduce(operator.mul, a.shape[1:]))
#     else:
#         shape_2d = (-1, a.shape[-1])
#     a = a.reshape(*shape_2d)
#     b = b.reshape(*shape_2d)
#
#     def get_cached_svd(matrix: Tensor, prefix: str) -> Tuple[Tensor, Tensor, Tensor]:
#         """Helper to handle SVD caching for either matrix."""
#         if cache is not None and f"{prefix}_polar" in cache:
#             # Cached polar decomposition available
#             u_polar = cache[f"{prefix}_polar"].to(device, dtype)
#             s = cache[f"{prefix}_s"].to(device, dtype)
#             vt = cache[f"{prefix}_vt"].to(device, dtype)
#         else:
#             # Calculate and cache SVD components
#             svd_driver = "gesvdj" if matrix.is_cuda else "none"
#             u, s, vt = torch.linalg.svd(matrix, full_matrices=False, driver=svd_driver)
#             u_polar = u @ vt  # Pre-compute polar component
#
#             if cache is not None:
#                 cache[f"{prefix}_polar"] = u_polar.to("cpu")
#                 cache[f"{prefix}_s"] = s.to("cpu")
#                 cache[f"{prefix}_vt"] = vt.to("cpu")
#
#         return u_polar, s, vt
#
#     # Get decompositions (from cache or compute)
#     u_a_polar, s_a, vt_a = get_cached_svd(a, "a")
#     u_b_polar, s_b, vt_b = get_cached_svd(b, "b")
#
#     # Get or compute alignment transform
#     if cache is not None and "transform" in cache:
#         transform = cache["transform"].to(device, dtype)
#     else:
#         transform = orthogonal_procrustes_ml(u_a_polar, u_b_polar)
#         if cache is not None:
#             cache["transform"] = transform.to("cpu")
#
#     # Align polar decompositions
#     u_b_polar_aligned = u_b_polar @ transform
#
#     # Compute positive semidefinite parts
#     p_a = vt_a.t() @ torch.diag(s_a + regularization_eps) @ vt_a
#     p_b = vt_b.t() @ torch.diag(s_b + regularization_eps) @ vt_b
#
#     # Merge components
#     merged_u = slerp_unitary_taylor(u_a_polar, u_b_polar_aligned, alpha)
#     merged_p = torch.lerp(p_a, p_b, alpha)
#
#     return (merged_u @ merged_p).reshape(original_shape)
#
# # def polar_decomposition(a: Tensor, b: Tensor, alpha: float,
# # regularization_eps: float = 1e-6,
# # cache: Optional[Dict] = None) -> Tensor:
# # device = a.device
# # dtype = a.dtype
# # original_shape = a.shape
#
# # if not original_shape:
# # shape_2d = (1, 1)
# # elif len(a.shape) == 4:
# # shape_2d = (-1, functools.reduce(operator.mul, a.shape[1:]))
# # else:
# # shape_2d = (-1, a.shape[-1])
# # a = a.reshape(*shape_2d)
# # b = b.reshape(*shape_2d)
#
# # def get_cached_svd(matrix: Tensor, prefix: str) -> Tuple[Tensor, Tensor, Tensor]:
# # """Helper to handle SVD caching for either matrix."""
# # if cache is not None and f"{prefix}_polar" in cache:
# # # Cached polar decomposition available
# # u_polar = cache[f"{prefix}_polar"].to(device, dtype)
# # s = cache[f"{prefix}_s"].to(device, dtype)
# # vt = cache[f"{prefix}_vt"].to(device, dtype)
# # else:
# # # Calculate and cache SVD components
# # svd_driver = "gesvdj" if matrix.is_cuda else "gesvd"
# # u, s, vt = torch.linalg.svd(matrix, full_matrices=False, driver=driver)
# # u_polar = u @ vt  # Pre-compute polar component
#
# # if cache is not None:
# # cache[f"{prefix}_polar"] = u_polar.to("cpu")
# # cache[f"{prefix}_s"] = s.to("cpu")
# # cache[f"{prefix}_vt"] = vt.to("cpu")
#
# # return u_polar, s, vt
#
# # # Get decompositions (from cache or compute)
# # u_a_polar, s_a, vt_a = get_cached_svd(a, "a")
# # u_b_polar, s_b, vt_b = get_cached_svd(b, "b")
#
# # # Compute transformation directly
# # transform = u_a_polar.T @ u_b_polar
#
# # # Align polar decompositions
# # u_b_polar_aligned = u_b_polar @ transform
#
# # # Compute positive semidefinite parts
# # p_a = vt_a.t() @ torch.diag(s_a + regularization_eps) @ vt_a
# # p_b = vt_b.t() @ torch.diag(s_b + regularization_eps) @ vt_b
#
# # # Merge components
# # merged_u = slerp_unitary_taylor(u_a_polar, u_b_polar_aligned, alpha)
# # merged_p = torch.lerp(p_a, p_b, alpha)
#
# # return (merged_u @ merged_p).reshape(original_shape)
#
# def slerp_unitary_taylor(A: Tensor, B: Tensor, alpha: float, num_terms: int = 5) -> Tensor:
#     """
#     Performs slerp between two unitary matrices using a Taylor series approximation
#     of the matrix logarithm.
#
#     Args:
#         A: The first unitary matrix.
#         B: The second unitary matrix.
#         alpha: The interpolation factor (0 <= alpha <= 1).
#         num_terms: The number of terms to include in the Taylor series approximation.
#
#     Returns:
#         The interpolated unitary matrix.
#     """
#     if torch.allclose(A, B, atol=1e-6):
#         return A
#     else:
#         # Compute the relative rotation
#         relative_rotation = B @ A.t()
#
#         # Compute X for the Taylor series: X = relative_rotation - I
#         X = relative_rotation - torch.eye(relative_rotation.size(-1), device=A.device)
#
#         # Approximate the logarithm using the Taylor series
#         log_rotation = torch.zeros_like(X)
#         for i in range(1, num_terms + 1):
#             log_rotation += ((-1) ** (i + 1) / i) * torch.linalg.matrix_power(X, i)
#
#         # Interpolate in the tangent space
#         interpolated_log = alpha * log_rotation
#
#         # Map back to the space of unitary matrices
#         interpolated_unitary = torch.linalg.matrix_exp(interpolated_log) @ A
#
#         return interpolated_unitary
#
# def clip_embedding_merge_v3(a: Tensor, b: Tensor, alpha: float = 0.5) -> Tensor:
#     """
#     CLIP embedding merge focused on preserving directional relationships using orthogonal Procrustes.
#     """
#     # 1. Normalize embeddings
#     a_norm = F.normalize(a, p=2, dim=1)
#     b_norm = F.normalize(b, p=2, dim=1)
#
#     # 2. Compute rotation using orthogonal Procrustes
#     rotation = orthogonal_procrustes_ml(a_norm, b_norm)  # Replace SVD-based rotation
#
#     # 3. Apply rotation to b to align directional space
#     b_aligned = torch.mm(b, rotation.T)
#
#     # 4. Simple interpolation in aligned space
#     merged = (1 - alpha) * a + alpha * b_aligned
#
#     # 5. Preserve original norms
#     a_norms = torch.norm(a, dim=1, keepdim=True)
#     b_norms = torch.norm(b, dim=1, keepdim=True)
#     target_norms = (1 - alpha) * a_norms + alpha * b_norms
#
#     current_norms = torch.norm(merged, dim=1, keepdim=True)
#     merged = merged * (target_norms / (current_norms + 1e-8))
#
#     return merged
#
# def merge_cross_attention_qkv(a: Tensor, b: Tensor, alpha: float, key: str,
#                               cache: Optional[Dict] = None) -> Tensor:
#     """
#     Enhanced merge for cross-attention QKV layers with optimized caching for SVD.
#     Handles various architectures and projection types.
#     """
#     device = a.device
#     dtype = a.dtype
#
#     # Handle CLIP-G style concatenated QKV
#     if "in_proj" in key:
#         head_dim = a.shape[0] // 3
#         merged_parts = []
#
#         for i in range(3):
#             start = head_dim * i
#             end = head_dim * (i + 1)
#             part_a = a[start:end]
#             part_b = b[start:end]
#
#             # Use polar decomposition for each part with separate cache entries
#             part_key = f"{key}_part_{i}"
#             part_cache = cache.get(part_key, {}) if cache is not None else None
#             merged = polar_decomposition(part_a, part_b, alpha, cache=part_cache)
#             if cache is not None:
#                 cache[part_key] = part_cache
#
#             merged_parts.append(merged)
#
#         return torch.cat(merged_parts, dim=0)
#
#     # Handle regular CLIP text encoder layers
#     elif any(x in key for x in ["k_proj", "v_proj", "q_proj"]):
#         return merge_self_attention_qkv(a, b, alpha, key)
#
#     # Handle UNet cross-attention
#     else:
#         # For query projections, calculate `adjusted_alpha` without caching
#         if ".to_q." in key:
#             with torch.no_grad():
#                 # Generate some sample data for cosine similarity computation
#                 x = torch.randn(min(100, a.shape[-1]), a.shape[-1], device=device, dtype=dtype)
#                 q_a = x @ a.T
#                 q_b = x @ b.T
#                 sim = F.cosine_similarity(q_a.flatten(), q_b.flatten(), dim=0)
#                 adjusted_alpha = alpha * torch.sigmoid(sim * 0.5)
#
#             # Use polar decomposition with adjusted weight
#             return polar_decomposition(a, b, alpha=adjusted_alpha, cache=cache)
#
#         # For key/value projections (different dimensions), focus caching on SVD and transform
#         def get_cached_svd(matrix: Tensor, prefix: str) -> Tuple[Tensor, Tensor, Tensor]:
#             """Helper to handle SVD caching."""
#             cache_key = f"{key}_{prefix}"
#             if cache is not None and f"{cache_key}_u" in cache:
#                 u = cache[f"{cache_key}_u"].to(device, dtype)
#                 s = cache[f"{cache_key}_s"].to(device, dtype)
#                 vh = cache[f"{cache_key}_vh"].to(device, dtype)
#             else:
#                 svd_driver = "gesvdj" if matrix.is_cuda else "none"
#                 u, s, vh = torch.linalg.svd(matrix, full_matrices=False, driver=svd_driver)
#
#                 if cache is not None:
#                     cache[f"{cache_key}_u"] = u.to('cpu')
#                     cache[f"{cache_key}_s"] = s.to('cpu')
#                     cache[f"{cache_key}_vh"] = vh.to('cpu')
#
#             return u, s, vh
#
#         # Get cached SVD components for matrices `a` and `b`
#         u_a, s_a, vh_a = get_cached_svd(a, "a")
#         u_b, s_b, vh_b = get_cached_svd(b, "b")
#
#         # Interpolate singular values
#         s_merged = torch.lerp(s_a, s_b, alpha)
#
#         # Align spaces using the smaller dimension
#         k = min(vh_a.shape[0], vh_b.shape[0])
#
#         # Get or compute alignment transform
#         transform_key = f"{key}_transform"
#         if cache is not None and transform_key in cache:
#             R = cache[transform_key].to(device, dtype)
#         else:
#             R = orthogonal_procrustes_ml(vh_a[:k], vh_b[:k])
#             if cache is not None:
#                 cache[transform_key] = R.to('cpu')
#
#         vh_merged = torch.lerp(vh_a[:k], vh_b[:k] @ R.T, alpha)
#
#         # Reconstruct while preserving cross-modal relationships
#         merged = (u_a[:, :k] * s_merged[:k]) @ vh_merged
#
#         # Scale to preserve magnitude
#         scale_a = torch.norm(a)
#         scale_b = torch.norm(b)
#         target_scale = (1 - alpha) * scale_a + alpha * scale_b
#         current_scale = torch.norm(merged)
#
#         return merged * (target_scale / (current_scale + 1e-6))
#
# def merge_self_attention_qkv(a: Tensor, b: Tensor, alpha: float, key: str,
#                              cache: Optional[Dict] = None) -> Tensor:
#     """
#     Merge self-attention QKV layers with caching for polar decomposition.
#     Handles separate Q/K/V and concatenated formats for CLIP-G style models.
#     """
#     # Handle CLIP-G style concatenated QKV
#     if "in_proj" in key:
#         head_dim = a.shape[0] // 3
#         merged_parts = []
#
#         # Pre-fetch all cache entries to minimize repeated calls to cache.get
#         part_caches = [cache.get(f"{key}_part_{i}", {}) if cache else None for i in range(3)]
#
#         for i in range(3):
#             start = head_dim * i
#             end = head_dim * (i + 1)
#             part_a = a[start:end]
#             part_b = b[start:end]
#
#             # Use polar decomposition with separate cache namespace for each part
#             merged = polar_decomposition(part_a, part_b, alpha, cache=part_caches[i])
#
#             # Update the main cache after polar decomposition call, if caching is enabled
#             if cache is not None:
#                 cache[f"{key}_part_{i}"] = part_caches[i]
#
#             merged_parts.append(merged)
#
#         return torch.cat(merged_parts, dim=0)
#
#     # Handle separate Q/K/V projections
#     else:
#         # Calculate attention similarity and adjusted alpha (not cached)
#         with torch.no_grad():
#             x = torch.randn(min(100, a.shape[-1]), a.shape[-1], device=a.device, dtype=a.dtype)
#             attn_a = torch.softmax(x @ a.mT / math.sqrt(a.shape[-1]), dim=-1)  # Fix: Use .mT
#             attn_b = torch.softmax(x @ b.mT / math.sqrt(b.shape[-1]), dim=-1)  # Fix: Use .mT
#
#             kl_div = F.kl_div(attn_a.log(), attn_b, reduction='batchmean')
#             adjusted_alpha = alpha * torch.sigmoid(1.0 - kl_div)
#
#         # Call polar_decomposition without caching, due to dynamic adjusted_alpha
#         return polar_decomposition(a, b, alpha=adjusted_alpha, cache=cache)
#
# def merge_attention_output(a: Tensor, b: Tensor, alpha: float, key: str,
#                            cache: Optional[Dict] = None) -> Tensor:
#     """
#     Merge attention output projections while preserving output distribution,
#     without caching for dynamically adjusted alpha values.
#     """
#     with torch.no_grad():
#         # Generate sample inputs
#         x = torch.randn(min(512, a.shape[-1]), a.shape[-1], device=a.device, dtype=a.dtype)
#
#         # Get output representations
#         out_a = x @ a.T
#         out_b = x @ b.T
#
#         # Compute output statistics
#         stats_a = torch.stack([
#             out_a.std(dim=0).mean(),  # Feature variation
#             out_a.abs().mean(),  # Activation magnitude
#             (out_a > 0).float().mean()  # Activation sparsity
#         ])
#         stats_b = torch.stack([
#             out_b.std(dim=0).mean(),
#             out_b.abs().mean(),
#             (out_b > 0).float().mean()
#         ])
#
#         # Adjust merge weight based on output similarity
#         stats_diff = torch.norm(stats_a - stats_b)
#         adjusted_alpha = alpha * torch.sigmoid(1.0 - stats_diff)
#
#     # Call polar_decomposition without caching, due to dynamic adjusted_alpha
#     merged = polar_decomposition(a, b, alpha=adjusted_alpha, cache=cache)
#
#     # Scale to preserve activation magnitude
#     scale_a = torch.norm(out_a) / torch.norm(x)
#     scale_b = torch.norm(out_b) / torch.norm(x)
#     target_scale = (1 - alpha) * scale_a + alpha * scale_b
#
#     with torch.no_grad():
#         current_scale = torch.norm(x @ merged.T) / torch.norm(x)
#
#     return merged * (target_scale / (current_scale + 1e-6))
#
# def merge_ffn_proj(a: torch.Tensor, b: torch.Tensor, alpha: float, key: str) -> torch.Tensor:
#     """
#     Enhanced FFN projection handling that adapts to matrix size.
#     """
#     input_dim = a.shape[-1]  # For proj.weight, this would be 640 or 1280
#     output_dim = a.shape[0]  # For proj.weight, this would be 5120 or 10240
#     expansion_factor = output_dim / input_dim
#
#     if matrix_is_large(a, threshold=2048):  # Adjust threshold as needed
#         return merge_ffn_proj_conservative(a, b, alpha, expansion_factor)
#     else:
#         return merge_ffn_proj_standard(a, b, alpha, expansion_factor)
#
# def merge_ffn_proj_conservative(a: Tensor, b: Tensor, alpha: float,
#                                 expansion_factor: float) -> Tensor:
#     """
#     Conservative merging for larger FFN projections
#     """
#     # Split the large projection into groups
#     group_size = a.shape[-1]  # Input dimension
#     num_groups = int(expansion_factor)
#
#     # Reshape to handle groups separately
#     a_groups = a.reshape(num_groups, -1, a.shape[-1])
#     b_groups = b.reshape(num_groups, -1, b.shape[-1])
#
#     merged_groups = []
#     for i in range(num_groups):
#         # Process each group with attention to activation patterns
#         a_group = a_groups[i]
#         b_group = b_groups[i]
#
#         # Check activation similarity within group
#         with torch.no_grad():
#             test_input = torch.randn(min(100, a_group.shape[-1]),
#                                      a_group.shape[-1],
#                                      device=a.device).to(a.dtype)  # Ensure correct data type
#             a_act = torch.relu(test_input @ a_group.T)
#             b_act = torch.relu(test_input @ b_group.T).to(a.dtype)
#
#             # Compare activation patterns
#             similarity = F.cosine_similarity(
#                 a_act.flatten(),
#                 b_act.flatten(),
#                 dim=0
#             )
#
#         if similarity > 0.5:
#             # Similar activations - interpolate smoothly
#             merged_group = torch.lerp(a_group, b_group, alpha)
#         else:
#             # Different activations - preserve stronger features
#             merged_group = torch.where(
#                 torch.abs(a_group) > torch.abs(b_group),
#                 a_group,
#                 b_group
#             )
#
#         merged_groups.append(merged_group)
#
#     # Recombine groups
#     return torch.cat(merged_groups, dim=0)
#
# def merge_ffn_proj_standard(a: torch.Tensor, b: torch.Tensor, alpha: float,
#                             expansion_factor: float) -> torch.Tensor:
#     """
#     Standard merging for smaller FFN projections
#     """
#     # Normalize matrices
#     a_norm = F.normalize(a, dim=-1)
#     b_norm = F.normalize(b, dim=-1)
#
#     # Compute activation statistics
#     with torch.no_grad():
#         test_input = torch.randn(min(100, a.shape[-1]),
#                                  a.shape[-1],
#                                  device=a.device).to(a.dtype)  # Cast test_input to a.dtype
#         a_act = torch.relu(test_input @ a.T)
#         b_act = torch.relu(test_input @ b.T).to(a.dtype)
#
#         # Calculate activation statistics
#         a_stats = torch.stack([
#             (a_act > 0).float().mean(),  # sparsity
#             a_act[a_act > 0].std()  # activation spread
#         ])
#         b_stats = torch.stack([
#             (b_act > 0).float().mean(),
#             b_act[b_act > 0].std()
#         ])
#
#     # Calculate merge weight based on activation properties
#     stats_diff = torch.norm(a_stats - b_stats)
#     merge_weight = torch.sigmoid(1.0 - stats_diff) * alpha
#
#     # Interpolate with adjusted weight
#     merged = torch.lerp(a, b, merge_weight)
#
#     # Rescale to preserve activation magnitude
#     scale_a = torch.norm(a_act) / torch.norm(test_input)
#     scale_b = torch.norm(b_act) / torch.norm(test_input)
#     target_scale = (1 - alpha) * scale_a + alpha * scale_b
#     current_scale = torch.norm(torch.relu(test_input @ merged.T)) / torch.norm(test_input)
#
#     return merged * (target_scale / (current_scale + 1e-6))
#
# def merge_ffn_out(a: torch.Tensor, b: torch.Tensor, alpha: float, corr_threshold: float,
#                   cache: Optional[Dict[str, Dict[str, torch.Tensor]]] = None) -> torch.Tensor:
#     """
#     Enhanced FFN output merge that preserves feature relationships and activation patterns,
#     optimized with caching for SVD and orthogonal Procrustes alignment.
#     """
#     output_dim, input_dim = a.shape
#     device = a.device
#     dtype = a.dtype
#
#     # Generate sample activations
#     num_samples = min(512, input_dim)
#     with torch.no_grad():
#         x = torch.randn(num_samples, input_dim, device=device, dtype=dtype)
#         x = torch.nn.functional.gelu(x)
#
#         # Get output space representations
#         out_a = x @ a.T
#         out_b = x @ b.T
#
#         # Compute correlation matrices in output space
#         corr_a = torch.corrcoef(out_a.T)
#         corr_b = torch.corrcoef(out_b.T)
#
#     # Identify strongly correlated feature groups
#     groups_a = []
#     groups_b = []
#     used_indices = set()
#
#     # Find feature groups in both matrices
#     for i in range(output_dim):
#         if i in used_indices:
#             continue
#
#         # Find correlated features
#         group_a = torch.where(torch.abs(corr_a[i]) > corr_threshold)[0]
#         group_b = torch.where(torch.abs(corr_b[i]) > corr_threshold)[0]
#
#         if len(group_a) > 1 or len(group_b) > 1:
#             # Ensure we don't exceed the actual group size when storing
#             actual_size = min(len(group_a), len(group_b))
#             groups_a.append(group_a[:actual_size])  # Only take the matching number of indices
#             groups_b.append(group_b[:actual_size])
#             used_indices.update(group_a[:actual_size].tolist())
#
#     # Initialize merged tensor
#     merged = torch.zeros_like(a)
#
#     # Helper function for caching SVD components
#     def get_cached_svd(matrix: torch.Tensor, prefix: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         cache_key = f"{prefix}_svd"
#         if cache is not None and f"{cache_key}_u" in cache:
#             u = cache[f"{cache_key}_u"].to(device, dtype)
#             s = cache[f"{cache_key}_s"].to(device, dtype)
#             vh = cache[f"{cache_key}_vh"].to(device, dtype)
#         else:
#             u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
#             if cache is not None:
#                 cache[f"{cache_key}_u"] = u.cpu()
#                 cache[f"{cache_key}_s"] = s.cpu()
#                 cache[f"{cache_key}_vh"] = vh.cpu()
#         return u, s, vh
#
#     # Process each feature group
#     for group_a, group_b in zip(groups_a, groups_b):
#         # Extract relevant slices
#         slice_a = a[group_a]
#         slice_b = b[group_b]
#
#         # Normalize the slices
#         norm_a = torch.norm(slice_a, dim=1, keepdim=True)
#         norm_b = torch.norm(slice_b, dim=1, keepdim=True)
#         slice_a_norm = slice_a / (norm_a + 1e-8)
#         slice_b_norm = slice_b / (norm_b + 1e-8)
#
#         # Get SVD components with caching
#         u_a, s_a, v_a = get_cached_svd(slice_a_norm, f"{group_a}_a")
#         u_b, s_b, v_b = get_cached_svd(slice_b_norm, f"{group_b}_b")
#
#         # Use minimum number of components for alignment
#         k = min(v_a.shape[1], v_b.shape[1])
#
#         # Use orthogonal Procrustes for alignment with caching
#         if k > 0:
#             procrustes_key = f"procrustes_{len(group_a)}_{len(group_b)}"
#             if cache is not None and procrustes_key in cache:
#                 R = cache[procrustes_key].to(device, dtype)
#             else:
#                 R = orthogonal_procrustes_ml(v_a[:, :k], v_b[:, :k])
#                 if cache is not None:
#                     cache[procrustes_key] = R.cpu()
#
#             v_b_aligned = v_b[:, :k] @ R.T
#         else:
#             v_b_aligned = v_b[:, :k]
#
#         # Align and interpolate
#         v_merged = torch.lerp(v_a[:, :k], v_b_aligned, alpha)
#         s_merged = torch.exp((1 - alpha) * torch.log(s_a[:k] + 1e-8) + alpha * torch.log(s_b[:k] + 1e-8))
#
#         # Interpolate norms
#         norm_merged = (1 - alpha) * norm_a + alpha * norm_b
#
#         # Reconstruct and check shape before assignment
#         group_result = (u_a[:, :k] * s_merged.unsqueeze(0)) @ v_merged * norm_merged
#
#         # Ensure the reconstructed group_result has the correct shape for assignment
#         expected_shape = merged[group_a].shape
#         if group_result.shape != expected_shape:
#             # Apply padding or trimming to match expected shape
#             if group_result.shape[0] < expected_shape[0]:
#                 # Pad group_result to match the expected shape
#                 padding = (0, 0, 0, expected_shape[0] - group_result.shape[0])
#                 group_result = torch.nn.functional.pad(group_result, padding)
#             elif group_result.shape[0] > expected_shape[0]:
#                 # Trim group_result to match the expected shape
#                 group_result = group_result[:expected_shape[0]]
#
#         merged[group_a] = group_result
#
#     # Handle uncorrelated features
#     uncorrelated = list(set(range(output_dim)) - used_indices)
#     if uncorrelated:
#         merged[uncorrelated] = torch.lerp(a[uncorrelated], b[uncorrelated], alpha)
#
#     # Scale adjustment
#     with torch.no_grad():
#         out_merged = x @ merged.T
#         scale_a = torch.norm(out_a) / torch.norm(x)
#         scale_b = torch.norm(out_b) / torch.norm(x)
#         target_scale = (1 - alpha) * scale_a + alpha * scale_b
#         current_scale = torch.norm(out_merged) / torch.norm(x)
#         merged = merged * (target_scale / (current_scale + 1e-8))
#
#     return merged
#
# @merge_method
# def geometric_sum_full(
#         a: Parameter(Tensor, "weight"),
#         b: Parameter(Tensor, "weight"),
#         *,
#         alpha: Parameter(Tensor) =0.5,
#         **kwargs,
# ) -> Return(Tensor):
#     a = torch.complex(a, torch.zeros_like(a))
#     b = torch.complex(b, torch.zeros_like(b))
#     res = a ** (1 - alpha) * b ** alpha
#     return res.real
#
# def merge_wavelets(a: Tensor, b: Tensor, alpha: float, wave: str = 'db4',
#                    levels: int = None) -> Tensor:
#     """
#     Merges two convolutional layers using a multi-level wavelet transform
#     while attempting to preserve original sizes. Kernels are reshaped to 2D
#     before the transform, and explicit padding is removed.
#
#     Args:
#     - a, b: Input tensors (convolutional kernels)
#     - alpha: Blending factor (0 to 1)
#     - wave: Wavelet to use (default: 'db3')
#     - levels: Number of decomposition levels
#     """
#     original_size = a.shape
#
#     # Reshape tensors to 2D based on kernel size
#     is_conv_3x3 = len(a.shape) == 4 and a.shape[-1] != 1
#     is_conv_1x1 = len(a.shape) == 4 and a.shape[-1] == 1
#     if is_conv_3x3:
#         shape_2d = (-1, functools.reduce(operator.mul, a.shape[1:]))
#     elif is_conv_1x1:
#         shape_2d = (-1, functools.reduce(operator.mul, a.shape[1:]))
#     elif not a.shape:
#         shape_2d = (1, 1)
#     else:
#         shape_2d = (-1, a.shape[-1])
#
#     a = a.reshape(*shape_2d)
#     b = b.reshape(*shape_2d)
#
#     # Determine the number of levels if not specified
#     if levels is None:
#         levels = min(4, (max(shape_2d) - 1).bit_length() - 1)  # Adaptive J
#
#     # Initialize wavelet transform
#     dwt = DWTForward(J=levels, wave=wave, mode='zero')
#     idwt = DWTInverse(wave=wave, mode='zero')
#     dwt = dwt.to(device=a.device, dtype=a.dtype)
#     idwt = idwt.to(device=a.device, dtype=a.dtype)
#
#     # Perform forward DWT (on 2D matrices)
#     a_ll, a_h = dwt(a.unsqueeze(0).unsqueeze(0))  # Add batch and channel dimensions
#     b_ll, b_h = dwt(b.unsqueeze(0).unsqueeze(0))  # Add batch and channel dimensions
#
#     # Merge the low-frequency components
#     merged_ll = alpha * a_ll + (1 - alpha) * b_ll
#
#     # Merge the high-frequency components
#     merged_h = []
#     for a_h_level, b_h_level in zip(a_h, b_h):
#         merged_h_level = alpha * a_h_level + (1 - alpha) * b_h_level
#         merged_h.append(merged_h_level)
#
#     # Perform inverse DWT
#     merged = idwt((merged_ll, merged_h)).squeeze(0).squeeze(0)  # Remove batch and channel dimensions
#
#     # Reshape back to original size (no cropping needed)
#     return merged.reshape(original_size)
#
# def slerp_interp(a: Tensor, b: Tensor, alpha: float) -> Tensor:
#     """
#     Spherical linear interpolation (slerp) between two tensors `a` and `b`.
#     Args:
#         a: The first tensor, normalized along the appropriate dimension.
#         b: The second tensor, same shape as `a`.
#         alpha: The interpolation factor (0 <= alpha <= 1).
#     Returns:
#         Interpolated tensor in the same shape as `a` and `b`.
#     """
#     # Normalize input tensors along the feature dimension
#     a_norm = a / a.norm(dim=-1, keepdim=True)
#     b_norm = b / b.norm(dim=-1, keepdim=True)
#
#     # Dot product between the normalized tensors to calculate the angle
#     dot_product = torch.clamp((a_norm * b_norm).sum(dim=-1, keepdim=True), -1.0, 1.0)
#     theta = torch.acos(dot_product)
#
#     # Spherical interpolation formula
#     sin_theta = torch.sin(theta)
#     slerp_factor_a = torch.sin((1 - alpha) * theta) / sin_theta
#     slerp_factor_b = torch.sin(alpha * theta) / sin_theta
#
#     # Calculate and return the interpolated tensor
#     return slerp_factor_a * a + slerp_factor_b * b
#
# def get_layer_type(shape, kwargs):
#     key = kwargs["key"]
#
#     # Prioritize checks for bias and other specific types
#     if key.endswith(".bias") or "bias" in key:
#         return LayerType.OFFSET
#
#     # Layer Norms
#     elif any(x in key for x in [".norm", "layer_norm", "ln_final", "ln_1", "ln_2", "layer_norm1", "layer_norm2",
#                                 "final_layer_norm"]) or "norm" in key:
#         return LayerType.SCALAR
#
#     # Scalar Layer (like `logit_scale` in CLIP models)
#     elif "logit_scale" in key:
#         return LayerType.SCALAR
#
#     # True embeddings (vocabulary mappings)
#     elif "token_embedding" in key or "shared.weight" in key:
#         return LayerType.EMBEDD
#
#     # Check for attention layers first
#     elif any(x in key for x in
#              [".to_q.", ".to_k.", ".to_v.", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
#               ".in_proj_"]):
#         # Add cross-attention check
#         if ".attn2." in key:
#             return LayerType.CROSS_ATTENTION_QKV
#         return LayerType.ATTENTION_QKV
#
#     # Attention Projection (output projection in both CLIP-G and CLIP-L)
#     elif any(x in key for x in [".to_out.", "self_attn.out_proj"]) and ".weight" in key:
#         return LayerType.ATTENTION_PROJ
#
#     # Feed Forward Network (FFN) in Stable Diffusion layers
#     elif ".ff.net." in key and ".proj." in key:
#         return LayerType.FFN_PROJ
#     elif ".ff.net." in key and ".weight" in key:
#         return LayerType.FFN_OUT
#
#     # Feed Forward Network (FFN) in CLIP-G and CLIP-L
#     elif "mlp.c_fc" in key and ".weight" in key:
#         return LayerType.FFN_PROJ
#     elif "mlp.c_proj" in key and ".weight" in key:
#         return LayerType.FFN_OUT
#     elif "mlp.fc1" in key and ".weight" in key:
#         return LayerType.FFN_PROJ
#     elif "mlp.fc2" in key and ".weight" in key:
#         return LayerType.FFN_OUT
#
#     # Matrix Transformation for Embedding-Like Layers (positional embeddings, projections)
#     elif any(x in key for x in ["positional_embedding", "text_projection", "label_emb"]):
#         return LayerType.MATMUL
#
#     # Convolutional Layers
#     elif len(shape) == 4:
#         return LayerType.CONV2D
#
#     # Default to matrix transformations
#     return LayerType.MATMUL
#
# class LayerType(enum.Enum):
#     SCALAR = enum.auto()
#     OFFSET = enum.auto()
#     CONV2D = enum.auto()
#     EMBEDD = enum.auto()
#     MATMUL = enum.auto()
#     ATTENTION_QKV = enum.auto()
#     CROSS_ATTENTION_QKV = enum.auto()  # New type
#     ATTENTION_PROJ = enum.auto()
#     FFN_PROJ = enum.auto()
#     FFN_OUT = enum.auto()
#
# def matrix_is_large(A: Tensor, threshold: int = 1280) -> bool:
#     """
#     Determines if a matrix is considered "large" based on its dimensions.
#
#     Args:
#         A: The input matrix.
#         threshold: The threshold for the minimum dimension size to be considered "large."
#
#     Returns:
#         True if the matrix is considered large, False otherwise.
#     """
#     if A.ndim < 2:  # Check if tensor has fewer than 2 dimensions
#         return False  # Treat non-2D tensors as "not large"
#     m, n = A.shape  # Get the matrix dimensions
#     return m >= threshold or n >= threshold  # Check if either dimension exceeds the threshold
#
# def dominant_rotation(A: Tensor, threshold: float = 0.8) -> bool:
#     """
#     Estimates if a matrix primarily represents a rotation based on its singular values.
#
#     Args:
#         A: The input matrix.
#         threshold: The threshold for the ratio of the largest singular value to the smallest
#                    singular value to be considered "dominant rotation."
#
#     Returns:
#         True if the matrix is estimated to have a dominant rotation, False otherwise.
#     """
#     _, S, _ = torch.linalg.svd(A)  # Compute the singular values of the matrix
#     largest_singular_value = S[0]
#     smallest_singular_value = S[-1]
#     return largest_singular_value / smallest_singular_value >= threshold
#
# def matrix_is_ill_conditioned(A: Tensor, threshold: float = 100) -> bool:
#     """
#     Determines if a matrix is ill-conditioned based on its condition number.
#
#     Args:
#         A: The input matrix.
#         threshold: The threshold for the condition number to be considered ill-conditioned.
#
#     Returns:
#         True if the matrix is ill-conditioned, False otherwise.
#     """
#     condition_number = torch.linalg.cond(A)  # Compute the condition number
#     return condition_number >= threshold
#
# def orthogonal_procrustes_ml(a, b, cancel_reflection: bool = False):
#     # Compute A^T @ B once since it's used in both branches
#     atb = a.T @ b
#
#     use_lowrank = not cancel_reflection and a.shape[0] + 10 < a.shape[1]
#     if use_lowrank:
#         svd_driver = "gesvdj" if a.is_cuda else None
#         u, _, v = sd_mecha.merge_methods.svd.torch_svd_lowrank(atb, driver=svd_driver, q=a.shape[0] + 10)
#         vt = v.T
#         del v
#     else:
#         svd_driver = "gesvdj" if a.is_cuda else None
#         u, _, vt = torch.linalg.svd(atb, driver=svd_driver)
#         if cancel_reflection:
#             u[:, -1] *= torch.sign(torch.det(u) * torch.det(vt))  # More numerically stable
#
#     transform = u @ vt
#
#     if not torch.isfinite(transform).all():  # Check the transform instead of just u
#         raise ValueError(
#             f"determinant error: {torch.det(transform)}. "
#             'This can happen when merging on the CPU with the "rotate" method. '
#             "Consider merging on a cuda device, "
#             "or try setting `alignment` to 1 for the problematic blocks. "
#             "See this related discussion for more info: "
#             "https://github.com/s1dlx/meh/pull/50#discussion_r1429469484"
#         )
#
#     return transform
#
# def get_svd_cached(tensor: Tensor, cache: Optional[Dict], key: str, suffix: str = "") -> Tuple[
#     Tensor, Tensor, Tensor]:
#     """Standardized SVD caching for hierarchical cache structure."""
#     device = tensor.device
#     dtype = tensor.dtype
#     # Create a standardized key including both parameters
#     cache_key = f"svd_{key}_{suffix}" if suffix else f"svd_{key}"
#
#     if cache is not None and cache_key in cache:
#         # Unpack cached SVD
#         cached_svd = cache[cache_key]
#         u = cached_svd["u"].to(device, dtype)
#         s = cached_svd["s"].to(device, dtype)
#         vh = cached_svd["vh"].to(device, dtype)
#         return u, s, vh
#
#     # Compute SVD (use optimal driver for device)
#     svd_driver = "gesvdj" if tensor.is_cuda else "gesvd"
#     u, s, vh = torch.linalg.svd(tensor, full_matrices=False, driver=svd_driver)
#
#     # Cache the results (on CPU to save GPU memory)
#     if cache is not None:
#         cache[cache_key] = {
#             "u": u.cpu(),
#             "s": s.cpu(),
#             "vh": vh.cpu()
#         }
#
#     return u, s, vh
#
# def get_procrustes_cached(matrix_a: Tensor, matrix_b: Tensor,
#                           cache: Optional[Dict], key: str,
#                           suffix: str = "") -> Tensor:
#     """Standardized orthogonal Procrustes caching for hierarchical cache."""
#     device = matrix_a.device
#     dtype = matrix_a.dtype
#     # Use consistent key pattern with get_svd_cached
#     cache_key = f"proc_{key}_{suffix}" if suffix else f"proc_{key}"
#
#     if cache is not None and cache_key in cache:
#         return cache[cache_key].to(device, dtype)
#
#     # Compute Procrustes alignment
#     R = orthogonal_procrustes_ml(matrix_a, matrix_b)
#
#     # Cache the result
#     if cache is not None:
#         cache[cache_key] = R.cpu()
#
#     return R
