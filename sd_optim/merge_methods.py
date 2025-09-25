import re
import sys
import warnings

import pywt
import sd_mecha
import functools
import pathlib
import gc
import enum
import operator
import torch
import math
import safetensors.torch
import torch.nn.functional as F
import numpy as np
import fnmatch
import geoopt

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from scipy.linalg import sqrtm
from scipy.stats import binom, rankdata
from torch import Tensor, polar
from torch.utils import checkpoint
from typing import Optional, Callable, Dict, Tuple, TypeVar, Generic, get_type_hints, get_origin, Union, get_args, List,  Set, Iterable
from pytorch_wavelets import DWTForward, DWTInverse
from sd_mecha import Parameter, Return, merge_method  # Import Parameter and Return

from sd_optim.TALON import TALON
from sd_optim.svd import torch_svd_lowrank  # you need to make your own or use the one from mecha
from sd_mecha.extensions.builtin.merge_methods.svd import svd_lowrank, stiefel_interpolate, fractional_orthogonal_matrix_power
from torch import Tensor  # Import Tensor

try:
    import cupy as cp
    from cupy.cuda import cusolver

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

EPSILON = 1e-10


class MergeMethods:

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

        if key:  # Only print if key is available
            # The 'alpha' variable here is ALREADY the specific float value for this key
            print(f"[merge_layers] Key: {key} -- Using alpha: {alpha:.4f}")

        # --- ADDED NaN/Inf CHECK ---
        a_is_finite = torch.isfinite(a).all()
        b_is_finite = torch.isfinite(b).all()

        if not a_is_finite or not b_is_finite:
            warning_msg = f"({key}): Non-finite values detected in input tensors! "
            if not a_is_finite: warning_msg += "Input 'a' has NaNs/Infs. "
            if not b_is_finite: warning_msg += "Input 'b' has NaNs/Infs. "
            warning_msg += "Returning input 'a' as fallback."
            # Use your logging system here if you have one, otherwise print
            print(warning_msg, file=sys.stderr)  # Or logpy.warning(warning_msg)
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

        layer_type = MergeMethods.get_layer_type(a.shape, kwargs)

        if layer_type == MergeMethods.LayerType.SCALAR:
            return MergeMethods.geometric_sum_full.__wrapped__(a, b, alpha=alpha)
        elif layer_type == MergeMethods.LayerType.OFFSET:
            return torch.lerp(a, b, alpha)
        elif layer_type == MergeMethods.LayerType.EMBEDD:
            return MergeMethods.clip_embedding_merge_v3(a, b, alpha=alpha)
        elif layer_type == MergeMethods.LayerType.CROSS_ATTENTION_QKV:
            return MergeMethods.merge_cross_attention_qkv(a, b, alpha=alpha, key=key, cache=layer_cache)
        elif layer_type == MergeMethods.LayerType.ATTENTION_QKV:
            return MergeMethods.merge_self_attention_qkv(a, b, alpha, key=key, cache=layer_cache)
        elif layer_type == MergeMethods.LayerType.ATTENTION_PROJ:
            return MergeMethods.merge_attention_output(a, b, alpha, key=key, cache=layer_cache)
        elif layer_type == MergeMethods.LayerType.FFN_PROJ:
            return MergeMethods.merge_ffn_proj(a, b, alpha=alpha, key=key)
        elif layer_type == MergeMethods.LayerType.FFN_OUT:
            return MergeMethods.merge_ffn_out(a, b, alpha=alpha, corr_threshold=corr_threshold, cache=layer_cache)
        elif layer_type == MergeMethods.LayerType.MATMUL:
            return MergeMethods.polar_decomposition(a, b, alpha=alpha, cache=layer_cache)
        elif layer_type == MergeMethods.LayerType.CONV2D:
            return MergeMethods.merge_wavelets(a, b, alpha=alpha)
        else:
            return torch.lerp(a, b, alpha)

    @staticmethod
    def polar_decomposition(a: Tensor, b: Tensor, alpha: float,
                            regularization_eps: float = 1e-6,
                            cache: Optional[Dict] = None,
                            key_prefix: str = "polar") -> Tensor:
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

        def get_cached_svd(matrix: torch.Tensor, name_suffix: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            svd_cache_key_prefix = f"{key_prefix}_{name_suffix}"
            u_svd, s_svd, vt_svd = MergeMethods._get_standard_cached_svd(matrix, cache, svd_cache_key_prefix,
                                                                         device, dtype)
            u_polar = u_svd @ vt_svd  # Orthogonal factor (closest orthogonal matrix)
            return u_polar, s_svd, vt_svd

        u_a_polar, s_a, vt_a = get_cached_svd(a_2d, "a")
        u_b_polar, s_b, vt_b = get_cached_svd(b_2d, "b")

        # Align orthogonal factors using Procrustes
        transform_cache_key = f"{key_prefix}_transform"
        if cache is not None and transform_cache_key in cache:
            transform = cache[transform_cache_key].to(device, dtype)
        else:
            transform = MergeMethods.orthogonal_procrustes_ml(u_a_polar, u_b_polar)
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
            merged_u = MergeMethods.slerp_grassmann(u_a_polar, u_b_polar_aligned, alpha,
                                                    cache=slerp_internal_cache,
                                                    key_prefix=f"{key_prefix}_grassmann")
        else:  # Tall or square matrices
            merged_u = MergeMethods.slerp_stiefel(u_a_polar, u_b_polar_aligned, alpha,
                                                  cache=slerp_internal_cache,
                                                  key_prefix=f"{key_prefix}_stiefel")

        if cache is not None and slerp_internal_cache:
            cache[slerp_sub_cache_key] = slerp_internal_cache

        # LERP on positive factors
        merged_p = torch.lerp(p_a, p_b, alpha)

        # Reconstruct result
        result = (merged_u @ merged_p).reshape(original_shape)

        return result

    @staticmethod
    def slerp_grassmann(  # Version 1.0 from user, minimally modified
            u_a: Tensor, u_b: Tensor, alpha: float,
            cache: Optional[Dict] = None, key_prefix: str = "grassmann"
    ) -> Tensor:
        # Based on Edelman, Arias, Smith (1998) "The Geometry of Algorithms with Orthogonality Constraints", Eq (2.4)
        # Adapted for U_A, U_B being M x N with Orthonormal Rows (ONR)
        # Original formula is for N x P with Orthonormal Columns (ONC)

        if alpha == 0.0: return u_a
        if alpha == 1.0: return u_b

        if torch.allclose(u_a, u_b, atol=1e-6):
            return u_a

        device, dtype, M, N = u_a.device, u_a.dtype, u_a.shape[0], u_a.shape[1]

        if M == N:  # Square matrix - delegate to specialized function
            return MergeMethods.slerp_square_unitary(u_a, u_b, alpha,
                                                     cache=cache,
                                                     key_prefix=f"{key_prefix}_as_sq_unitary")

        C_matrix = u_a @ u_b.T  # M x M

        svd_C_key_v, svd_C_key_s, svd_C_key_w_t = f"{key_prefix}_svd_C_v", f"{key_prefix}_svd_C_s", f"{key_prefix}_svd_C_w_t"
        if cache is not None and svd_C_key_v in cache:
            v_c, s_c_diag, w_c_t = cache[svd_C_key_v].to(device, dtype), cache[svd_C_key_s].to(device, dtype), cache[
                svd_C_key_w_t].to(device, dtype)
        else:
            svd_driver = "gesvda" if u_a.is_cuda else None
            v_c, s_c_diag, w_c_t = torch.linalg.svd(C_matrix, driver=svd_driver)
            if cache is not None: cache[svd_C_key_v], cache[svd_C_key_s], cache[
                svd_C_key_w_t] = v_c.cpu(), s_c_diag.cpu(), w_c_t.cpu()

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
            if cache is not None: cache[q_onc_key] = q_onc.cpu()

        cos_interp_theta, sin_interp_theta = torch.cos(alpha * theta_s), torch.sin(alpha * theta_s)
        term1_cols = u_a.T @ w_c_t.T @ torch.diag(cos_interp_theta) @ v_c.T
        term2_cols = q_onc @ torch.diag(sin_interp_theta) @ v_c.T
        x_interp_cols = term1_cols + term2_cols

        u_interp = x_interp_cols.T

        return u_interp

    @staticmethod
    def slerp_stiefel(a: Tensor, b: Tensor, alpha: float,
                      cache: Optional[Dict] = None, key_prefix: str = "stiefel") -> Tensor:
        """Complete Stiefel manifold interpolation"""
        if alpha == 0.0: return a
        if alpha == 1.0: return b

        m, n = a.shape
        if m == n:
            return MergeMethods.slerp_square_unitary(a, b, alpha,
                                                     cache=cache,
                                                     key_prefix=f"{key_prefix}_as_sq_unitary")

        if torch.allclose(a, b, atol=1e-6):
            return a

        try:  # Primary path with proper caching
            tangent_vector = MergeMethods.log_stiefel(a, b, cache=cache, key_prefix=f"{key_prefix}_log")
            scaled_tangent = alpha * tangent_vector
            result = MergeMethods.exp_stiefel(a, scaled_tangent)
            return result
        except Exception as e_logexp:  # Fallback path
            print(f"Warning: slerp_stiefel fallback triggered for {key_prefix}. "
                  f"Reason: {type(e_logexp).__name__}. Using direct SVD method.", file=sys.stderr)

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

    @staticmethod
    def log_stiefel(a, b, tau=None, max_iter=30, cache: Optional[Dict] = None,
                    key_prefix: str = "log_stiefel"):
        assert max_iter >= 1

        log_stiefel_key = f"{key_prefix}_result_log_stiefel_cpu"
        if cache is not None and log_stiefel_key in cache:
            return cache[log_stiefel_key].to(a.device, a.dtype)

        n, p = a.shape
        device, dtype = a.device, a.dtype
        tau = tau or 100 * torch.finfo(dtype).eps * p

        m_mat = a.T @ b
        b_minus_am = b - a @ m_mat
        q_orth, n_mat = MergeMethods.qr_pos(b_minus_am)

        v_intermediate_cat = torch.cat((m_mat, n_mat), dim=0)
        v = MergeMethods.orthogonal_complete(v_intermediate_cat)

        r_svd, sigma_svd, r_hat_t_svd = torch.linalg.svd(v[p:, p:], driver="gesvda" if v.is_cuda else None)
        q_orth @= r_svd
        v[p:, :p] = r_svd.T @ n_mat
        v[:p, p:] @= r_hat_t_svd.T
        p_arange = torch.arange(p, 2 * p, device=device)
        v[p:, p:].zero_()
        v[p_arange, p_arange] = sigma_svd
        del r_svd, sigma_svd, r_hat_t_svd, p_arange

        for i in range(max_iter):
            l_iter = MergeMethods.logm(v)
            c_syl, lh_block = l_iter[p:, p:], l_iter[p:, :p]
            c_norm = torch.linalg.matrix_norm(c_syl)
            if c_norm <= tau:
                break
            s_syl = (lh_block @ lh_block.mH) / 12.0 - torch.eye(p, device=device, dtype=dtype) / 2.0
            g_syl = MergeMethods.solve_symmetric_sylvester(s_syl, c_syl)
            v_update_exp = torch.linalg.matrix_exp(g_syl)
            v[:, p:] @= v_update_exp

        # Robust final computation - always recompute after iterations
        l_final = MergeMethods.logm(v)
        delta = a @ l_final[:p, :p] + q_orth @ l_final[p:, :p]
        if cache is not None:
            cache[log_stiefel_key] = delta.to("cpu")

        return delta

    @staticmethod
    def logm(m: Tensor) -> Tensor:  # Removed unused key_prefix
        """Matrix logarithm using eigendecomposition with numerically stable reconstruction"""
        original_dtype = m.dtype

        # Promote to complex for eig if real, as eigenvalues/vectors can be complex
        compute_dtype = m.dtype if m.is_complex() else (
            torch.complex64 if m.dtype == torch.float32 else torch.complex128)
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

    @staticmethod
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

        q2 = torch.linalg.householder_product(*torch.linalg.qr(projected, mode='raw'))

        return torch.cat([q_ortho, q2[:, :n - k]], dim=1)

    @staticmethod
    def solve_symmetric_sylvester(s, c):
        """Solve symmetric Sylvester equation AX + XA = C where A=s, C=c"""
        v, vs = torch.linalg.eigh(s)  # s is symmetric/Hermitian

        # Transform to diagonal coordinates: vs.mH @ c @ vs
        c_t = vs.mH @ c @ vs

        # Denominator matrix: λ_i + λ_j for all pairs
        d = v.unsqueeze(0) + v.unsqueeze(1)

        # Check for singularity (λ_i + λ_j ≈ 0)
        if torch.any(torch.abs(d) < 1e-12):
            print("Warning: Singular Sylvester operator: some λ_i+λ_j ≈ 0", file=sys.stderr)
            # Could regularize with: d[torch.abs(d) < 1e-12] = 1e-12 * torch.sign(d[torch.abs(d) < 1e-12])

        # Solve in diagonal coordinates
        g_t = c_t / d

        # Transform back: vs @ g_t @ vs.mH
        g = vs @ g_t @ vs.mH  # Fixed: consistent use of .mH

        return g

    @staticmethod
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

    @staticmethod
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
            q2 = q[:, p:2 * p]
        else:
            # When n < 2p, pad q2 with zeros
            q2 = torch.zeros(n, p, device=a.device, dtype=a.dtype)
            available_cols = q.shape[1] - p
            if available_cols > 0:
                q2[:, :available_cols] = q[:, p:p + available_cols]

        # Extract R blocks safely
        r12 = r[:p, p:2 * p]

        # Handle r22 extraction for edge cases
        min_dim = min(r.shape[0], 2 * p)
        if min_dim >= 2 * p:
            r22 = r[p:2 * p, p:2 * p]
        else:
            # Degenerate case: pad with identity to avoid singularity
            r22 = torch.eye(p, device=a.device, dtype=a.dtype)
            if min_dim > p:
                actual_size = min_dim - p
                r22[:actual_size, :actual_size] = r[p:min_dim, p:2 * p][:, :actual_size]

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

    @staticmethod
    def _matrix_logarithm_eig(matrix: torch.Tensor,
                              cache: Optional[Dict] = None,
                              key_prefix: str = "logm_eig_default") -> torch.Tensor:
        log_eig_cache_key = f"{key_prefix}_result_cpu"

        if cache is not None and log_eig_cache_key in cache:
            cached_log_A = cache[log_eig_cache_key].to(device=matrix.device, dtype=matrix.dtype)
            return cached_log_A

        if not (matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]):
            raise ValueError(f"Matrix logarithm expects a square matrix. Got shape: {matrix.shape}")

        original_dtype = matrix.dtype
        compute_dtype = matrix.dtype if matrix.is_complex() else (
            torch.complex64 if matrix.dtype == torch.float32 else torch.complex128)
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

    @staticmethod
    def slerp_square_unitary(
            A: torch.Tensor, B: torch.Tensor, alpha: float,
            cache: Optional[Dict] = None,
            key_prefix: str = "sq_unitary_default"
    ) -> torch.Tensor:
        """
        SLERP for square unitary/orthogonal matrices using matrix logarithm.
        Caches the expensive, alpha-independent matrix logarithm computation.
        """
        if alpha == 0.0: return A
        if alpha == 1.0: return B
        if torch.allclose(A, B, atol=1e-6): return A

        device, original_dtype = A.device, A.dtype
        compute_c_dtype = torch.complex64 if original_dtype in [torch.float32, torch.complex64] else torch.complex128
        was_real_input = not A.is_complex()
        A_c, B_c = A.to(compute_c_dtype), B.to(compute_c_dtype)

        try:
            relative_rotation = B_c @ A_c.mH

            # Cache the expensive matrix logarithm (alpha-independent)
            log_R_key_prefix_for_helper = f"{key_prefix}_log_rel_rot"
            log_R = MergeMethods._matrix_logarithm_eig(
                relative_rotation,
                cache=cache,
                key_prefix=log_R_key_prefix_for_helper
            )

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
            print(f"Warning: slerp_square_unitary fallback triggered for {key_prefix}. "
                  f"Reason: {type(e).__name__}. Using LERP+SVD.", file=sys.stderr)
            lerped_val = torch.lerp(A, B, alpha)
            try:
                u_lerp, _, vh_lerp = torch.linalg.svd(lerped_val, full_matrices=False)
                fallback_result = (u_lerp @ vh_lerp).to(original_dtype)
                return fallback_result
            except Exception as e2:
                print(f"Warning: SVD fallback also failed for {key_prefix}. "
                      f"Reason: {type(e2).__name__}. Using raw LERP.", file=sys.stderr)
                return lerped_val

    @staticmethod
    def clip_embedding_merge_v3(a: Tensor, b: Tensor, alpha: float = 0.5) -> Tensor:
        """
        CLIP embedding merge focused on preserving directional relationships using orthogonal Procrustes.
        """
        # 1. Normalize embeddings
        a_norm = F.normalize(a, p=2, dim=1)
        b_norm = F.normalize(b, p=2, dim=1)

        # 2. Compute rotation using orthogonal Procrustes
        rotation = MergeMethods.orthogonal_procrustes_ml(a_norm, b_norm)  # Replace SVD-based rotation

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

    @staticmethod
    def merge_cross_attention_qkv(a: Tensor, b: Tensor, alpha: float, key: str,
                                  cache: Optional[Dict] = None) -> Tensor:
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
                merged = MergeMethods.polar_decomposition(part_a, part_b, alpha, cache=part_cache)
                if cache is not None:
                    cache[part_key] = part_cache

                merged_parts.append(merged)

            return torch.cat(merged_parts, dim=0)

        # Handle regular CLIP text encoder layers
        elif any(x in key for x in ["k_proj", "v_proj", "q_proj"]):
            return MergeMethods.merge_self_attention_qkv(a, b, alpha, key)

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
                return MergeMethods.polar_decomposition(a, b, alpha=adjusted_alpha.item(), cache=cache)

            # Get cached SVD components for matrices `a` and `b` using the centralized helper
            u_a, s_a, vh_a = MergeMethods._get_standard_cached_svd(a, cache, f"{key}_a", device, dtype)
            u_b, s_b, vh_b = MergeMethods._get_standard_cached_svd(b, cache, f"{key}_b", device, dtype)

            # Interpolate singular values
            s_merged = torch.lerp(s_a, s_b, alpha)

            # Align spaces using the smaller dimension
            k = min(vh_a.shape[0], vh_b.shape[0])

            # Get or compute alignment transform
            transform_key = f"{key}_transform"
            if cache is not None and transform_key in cache:
                R = cache[transform_key].to(device, dtype)
            else:
                R = MergeMethods.orthogonal_procrustes_ml(vh_a[:k], vh_b[:k])
                if cache is not None:
                    cache[transform_key] = R.to('cpu')

            vh_merged = torch.lerp(vh_a[:k], vh_b[:k] @ R.T, alpha)

            # Reconstruct while preserving cross-modal relationships
            merged = (u_a[:, :k] * s_merged[:k]) @ vh_merged

            # Scale to preserve magnitude
            scale_a = torch.norm(a)
            scale_b = torch.norm(b)
            target_scale = (1 - alpha) * scale_a + alpha * scale_b
            current_scale = torch.norm(merged)

            return merged * (target_scale / (current_scale + 1e-6))

    @staticmethod
    def merge_self_attention_qkv(a: Tensor, b: Tensor, alpha: float, key: str,
                                 cache: Optional[Dict] = None) -> Tensor:
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
                merged = MergeMethods.polar_decomposition(part_a, part_b, alpha, cache=part_caches[i])

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

                kl_div = F.kl_div(attn_a.log(), attn_b, reduction='batchmean')
                adjusted_alpha = alpha * torch.sigmoid(1.0 - kl_div)

            # Call polar_decomposition without caching, due to dynamic adjusted_alpha
            return MergeMethods.polar_decomposition(a, b, alpha=adjusted_alpha.item(), cache=cache)

    @staticmethod
    def merge_attention_output(a: Tensor, b: Tensor, alpha: float, key: str,
                               cache: Optional[Dict] = None) -> Tensor:
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
            stats_a = torch.stack([
                out_a.std(dim=0).mean(),  # Feature variation
                out_a.abs().mean(),  # Activation magnitude
                (out_a > 0).float().mean()  # Activation sparsity
            ])
            stats_b = torch.stack([
                out_b.std(dim=0).mean(),
                out_b.abs().mean(),
                (out_b > 0).float().mean()
            ])

            # Adjust merge weight based on output similarity
            stats_diff = torch.norm(stats_a - stats_b)
            adjusted_alpha = alpha * torch.sigmoid(1.0 - stats_diff)

        # Call polar_decomposition without caching, due to dynamic adjusted_alpha
        merged = MergeMethods.polar_decomposition(a, b, alpha=adjusted_alpha.item(), cache=cache)

        # Scale to preserve activation magnitude
        scale_a = torch.norm(out_a) / torch.norm(x)
        scale_b = torch.norm(out_b) / torch.norm(x)
        target_scale = (1 - alpha) * scale_a + alpha * scale_b

        with torch.no_grad():
            current_scale = torch.norm(x @ merged.T) / torch.norm(x)

        return merged * (target_scale / (current_scale + 1e-6))

    @staticmethod
    def merge_ffn_proj(a: Tensor, b: Tensor, alpha: float, key: str) -> torch.Tensor:
        """
        Enhanced FFN projection handling that adapts to matrix size.
        """
        input_dim = a.shape[-1]  # For proj.weight, this would be 640 or 1280
        output_dim = a.shape[0]  # For proj.weight, this would be 5120 or 10240
        expansion_factor = output_dim / input_dim

        if MergeMethods.matrix_is_large(a, threshold=2048):  # Adjust threshold as needed
            return MergeMethods.merge_ffn_proj_conservative(a, b, alpha, expansion_factor)
        else:
            return MergeMethods.merge_ffn_proj_standard(a, b, alpha, expansion_factor)

    @staticmethod
    def merge_ffn_proj_conservative(a: Tensor, b: Tensor, alpha: float,
                                    expansion_factor: float) -> Tensor:
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
                test_input = torch.randn(min(100, a_group.shape[-1]),
                                         a_group.shape[-1],
                                         device=a.device).to(a.dtype)  # Ensure correct data type
                a_act = torch.relu(test_input @ a_group.T)
                b_act = torch.relu(test_input @ b_group.T).to(a.dtype)

                # Compare activation patterns
                similarity = F.cosine_similarity(
                    a_act.flatten(),
                    b_act.flatten(),
                    dim=0
                )

            if similarity > 0.5:
                # Similar activations - interpolate smoothly
                merged_group = torch.lerp(a_group, b_group, alpha)
            else:
                # Different activations - preserve stronger features
                merged_group = torch.where(
                    torch.abs(a_group) > torch.abs(b_group),
                    a_group,
                    b_group
                )

            merged_groups.append(merged_group)

        # Recombine groups
        return torch.cat(merged_groups, dim=0)

    @staticmethod
    def merge_ffn_proj_standard(a: Tensor, b: Tensor, alpha: float,
                                expansion_factor: float) -> Tensor:
        """
        Standard merging for smaller FFN projections
        """
        # Normalize matrices
        a_norm = F.normalize(a, dim=-1)
        b_norm = F.normalize(b, dim=-1)

        # Compute activation statistics
        with torch.no_grad():
            test_input = torch.randn(min(100, a.shape[-1]),
                                     a.shape[-1],
                                     device=a.device).to(a.dtype)  # Cast test_input to a.dtype
            a_act = torch.relu(test_input @ a.T)
            b_act = torch.relu(test_input @ b.T).to(a.dtype)

            # Calculate activation statistics
            a_stats = torch.stack([
                (a_act > 0).float().mean(),  # sparsity
                a_act[a_act > 0].std()  # activation spread
            ])
            b_stats = torch.stack([
                (b_act > 0).float().mean(),
                b_act[b_act > 0].std()
            ])

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

    @staticmethod
    def merge_ffn_out(a: Tensor, b: Tensor, alpha: float, corr_threshold: float,
                      cache: Optional[Dict[str, Dict[str, Tensor]]] = None) -> Tensor:
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
                # u_a, s_a, v_a = MergeMethods._get_standard_cached_svd(slice_a_norm, cache, f"{group_a}_a", device, dtype)
                # u_b, s_b, v_b = MergeMethods._get_standard_cached_svd(slice_b_norm, cache, f"{group_b}_b", device, dtype)

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
                    #     r = MergeMethods.orthogonal_procrustes_ml(v_a[:, :k], v_b[:, :k])
                    #     if cache is not None:
                    #         cache[procrustes_key] = r.cpu()

                    # Direct Procrustes computation without caching
                    r = MergeMethods.orthogonal_procrustes_ml(v_a[:, :k], v_b[:, :k])
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
                        group_result = group_result[:expected_shape[0]]

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
            a: Parameter(Tensor, "weight"),
            b: Parameter(Tensor, "weight"),
            alpha: Parameter(Tensor) = 0.5,
            **kwargs
    ) -> Return(Tensor, "weight"):
        key = kwargs["key"]
        if key:  # Only print if key is available
            # The 'alpha' variable here is ALREADY the specific float value for this key
            print(f"[geosum] Key: {key} -- Using alpha: {alpha:.4f}")
        a = torch.complex(a, torch.zeros_like(a))
        b = torch.complex(b, torch.zeros_like(b))
        res = a ** (1 - alpha) * b ** alpha
        return res.real

    @staticmethod
    def merge_wavelets(a: Tensor, b: Tensor, alpha: float, wave: str = 'db4',
                       levels: int = None) -> Tensor:
        """
        Merges two convolutional layers using a multi-level wavelet transform
        while attempting to preserve original sizes. Kernels are reshaped to 2D
        before the transform, and explicit padding is removed.

        Args:
        - a, b: Input tensors (convolutional kernels)
        - alpha: Blending factor (0 to 1)
        - wave: Wavelet to use (default: 'db3')
        - levels: Number of decomposition levels
        """
        original_size = a.shape

        # Reshape tensors to 2D based on kernel size
        is_conv_3x3 = len(a.shape) == 4 and a.shape[-1] != 1
        is_conv_1x1 = len(a.shape) == 4 and a.shape[-1] == 1
        if is_conv_3x3:
            shape_2d = (-1, functools.reduce(operator.mul, a.shape[1:]))
        elif is_conv_1x1:
            shape_2d = (-1, functools.reduce(operator.mul, a.shape[1:]))
        elif not a.shape:
            shape_2d = (1, 1)
        else:
            shape_2d = (-1, a.shape[-1])

        a = a.reshape(*shape_2d)
        b = b.reshape(*shape_2d)

        # Determine the number of levels if not specified
        if levels is None:
            levels = min(4, (max(shape_2d) - 1).bit_length() - 1)  # Adaptive J

        # Initialize wavelet transform
        dwt = DWTForward(J=levels, wave=wave, mode='zero')
        idwt = DWTInverse(wave=wave, mode='zero')
        dwt = dwt.to(device=a.device, dtype=a.dtype)
        idwt = idwt.to(device=a.device, dtype=a.dtype)

        # Perform forward DWT (on 2D matrices)
        a_ll, a_h = dwt(a.unsqueeze(0).unsqueeze(0))  # Add batch and channel dimensions
        b_ll, b_h = dwt(b.unsqueeze(0).unsqueeze(0))  # Add batch and channel dimensions

        # Merge the low-frequency components
        merged_ll = alpha * a_ll + (1 - alpha) * b_ll

        # Merge the high-frequency components
        merged_h = []
        for a_h_level, b_h_level in zip(a_h, b_h):
            merged_h_level = alpha * a_h_level + (1 - alpha) * b_h_level
            merged_h.append(merged_h_level)

        # Perform inverse DWT
        merged = idwt((merged_ll, merged_h)).squeeze(0).squeeze(0)  # Remove batch and channel dimensions

        # Reshape back to original size (no cropping needed)
        return merged.reshape(original_size)

    @staticmethod
    def get_layer_type(shape, kwargs):
        key = kwargs["key"]

        # Prioritize checks for bias and other specific types
        if key.endswith(".bias") or "bias" in key:
            return MergeMethods.LayerType.OFFSET

        # Layer Norms
        elif any(x in key for x in [".norm", "layer_norm", "ln_final", "ln_1", "ln_2", "layer_norm1", "layer_norm2",
                                    "final_layer_norm"]) or "norm" in key:
            return MergeMethods.LayerType.SCALAR

        # Scalar Layer (like `logit_scale` in CLIP models)
        elif "logit_scale" in key or "position_ids" in key:
            return MergeMethods.LayerType.SCALAR

        elif ".in_layers.0.weight" in key or ".out_layers.0.weight" in key:  # Specific to ResBlock norm weights
            return MergeMethods.LayerType.SCALAR

        # True embeddings (vocabulary mappings)
        elif "token_embedding" in key or "position_embedding" in key or "positional_embedding" in key or "shared.weight" in key:
            return MergeMethods.LayerType.EMBEDD

        # Check for attention layers first
        elif any(x in key for x in
                 [".to_q.", ".to_k.", ".to_v.", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                  ".in_proj_"]):
            # Add cross-attention check
            if ".attn2." in key:
                return MergeMethods.LayerType.CROSS_ATTENTION_QKV
            return MergeMethods.LayerType.ATTENTION_QKV

        # Attention Projection (output projection in both CLIP-G and CLIP-L)
        elif any(x in key for x in [".to_out.", ".out_proj"]) and ".weight" in key:
            return MergeMethods.LayerType.ATTENTION_PROJ

        # Feed Forward Network (FFN) in Stable Diffusion layers
        elif ".ff.net." in key and ".proj." in key:
            return MergeMethods.LayerType.FFN_PROJ
        elif ".ff.net." in key and ".weight" in key:
            return MergeMethods.LayerType.FFN_OUT

        # Feed Forward Network (FFN) in CLIP-G and CLIP-L
        elif "mlp.c_fc" in key and ".weight" in key:
            return MergeMethods.LayerType.FFN_PROJ
        elif "mlp.c_proj" in key and ".weight" in key:
            return MergeMethods.LayerType.FFN_OUT
        elif "mlp.fc1" in key and ".weight" in key:
            return MergeMethods.LayerType.FFN_PROJ
        elif "mlp.fc2" in key and ".weight" in key:
            return MergeMethods.LayerType.FFN_OUT

        # Matrix Transformation for Embedding-Like Layers (positional embeddings, projections)
        elif any(x in key for x in ["positional_embedding", "text_projection", "label_emb"]):
            return MergeMethods.LayerType.MATMUL

        # Convolutional Layers
        elif len(shape) == 4:
            return MergeMethods.LayerType.CONV2D

        # Default to matrix transformations
        return MergeMethods.LayerType.MATMUL

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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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
                full_matrices=False  # Start with False to mimic old V dim if that helps isolate
            )
            # The Procrustes solution R = U @ Vh
            transform = u_approx @ vh_approx  # <--- USE vh_approx DIRECTLY

        else:  # Standard SVD path (this part was already correct)
            svd_driver = "gesvdj" if a.is_cuda else None
            u_full, _, vh_full = torch.linalg.svd(atb, driver=svd_driver)  # vh_full is V_transpose

            final_u_for_transform = u_full
            final_vh_for_transform = vh_full  # Renamed for clarity
            if cancel_reflection:
                final_u_for_transform[:, -1] *= torch.sign(
                    torch.det(final_u_for_transform) * torch.det(final_vh_for_transform))

            transform = final_u_for_transform @ final_vh_for_transform  # U @ Vh

        if not torch.isfinite(transform).all():
            raise ValueError("Orthogonal Procrustes transform is not finite.")
        return transform

    @staticmethod
    def _get_standard_cached_svd(matrix: Tensor, cache: Optional[Dict], prefix: str,
                                 device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor, Tensor]:
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
                cache[cache_key_u] = u.to('cpu')
                cache[cache_key_s] = s.to('cpu')
                cache[cache_key_vh] = vh.to('cpu')

        return u, s, vh

    @merge_method
    def orthonorm(
            a: Parameter(Tensor, merge_space="delta"),  # orig_model - base
            *models: Parameter(Tensor, merge_space="delta"),  # b,c,d... - base
            iterative_alpha: Parameter(float) = 0.0,
            **kwargs,
    ) -> Return(Tensor, merge_space="delta"):  # Returns A with models perpendicularly added in their given order.
        """
        Projects the model diff g to be orthogonal to the current diff w.

        g_orth = g - ( (w·g)/(w·w + eps) ) * w

        And then re-scales g_orth to have the same norm as g.

        Modified function to use atan2 instead of an epsilon
        """
        total_res = torch.zeros_like(a)  # 0
        alpha = iterative_alpha
        if alpha == 0:
            alpha = 1 / (math.sqrt(len(models)))

        for m in models:
            w = (a + total_res).view(-1)
            g = m.view(-1)

            proj = torch.dot(w, g).atan2_(torch.dot(w, w)).mul_(1.27323954474)  # Scale by 1 / atan(1) (~1.27)
            g_orth = g.to(dtype=torch.float32, copy=True).sub_(w, alpha=proj)
            g_orth_scaled = g_orth.mul_(g.norm(2).clamp_min_(1e-6).div_(g_orth.norm(2).clamp_min_(1e-6)))

            total_res += g_orth_scaled.view(m.shape) * alpha
        # print(total_res)
        return total_res

    # V1.0 - Updated to use @sd_mecha.merge_method and type hints
    @staticmethod
    @merge_method  # Explicit identifier recommended
    def weighted_sum_0_filtered(
            # Use Parameter() to specify type and optionally merge space.
            # Assuming inputs 'a' and 'b' are expected in weight space.
            a: Parameter(Tensor, "weight"),
            b: Parameter(Tensor, "weight"),
            *,
            # alpha has a default, so it's implicitly 'param' space.
            # We can just use float, or Parameter(float). Parameter(float) is slightly more explicit.
            alpha: Parameter(Tensor) = 0.0,
            # **kwargs is essential to receive the 'key' and other metadata
            **kwargs,
    ) -> Return(Tensor, "weight"):  # Return type is Tensor, in weight space
        """
        Performs weighted sum (1-alpha)*a + alpha*b, but ONLY for keys matching
        specific patterns. Returns 'a' unmodified for non-matching keys.
        Effectively sets alpha=0 for non-matching keys.
        """
        # YES, the 'key' is still available in kwargs! sd-mecha injects it.
        key = kwargs.get("key")  # Use .get() for safety, though 'key' should always be there
        if key is None:
            # This shouldn't happen in normal sd-mecha execution, but good to handle
            print("Merge method 'weighted_sum_0_filtered' called without 'key' in kwargs.")
            return a  # Fallback to returning 'a'

        # The patterns to apply the weighted sum to
        patterns = [
            "*.norm.*",
            "*.norm1.*",
            "*.norm2.*",
            "*.norm3.*",
            "*.emb_layers.*",
            "model.diffusion_model.out.*",
            "*.conv.*",  # Careful, this might be too broad?
            "*.skip_connection.*",
            "model.diffusion_model.output_blocks.8.0.out_layers.3.*",  # Example specific key
            "model.diffusion_model.input_blocks.0.0.*",  # Example specific key
        ]

        # Check if the current key matches any pattern
        if any(fnmatch.fnmatch(key, pattern) for pattern in patterns):
            # logger.debug(f"Applying weighted sum (alpha={alpha}) for key: {key}") # Optional logging
            # Perform the actual weighted sum
            # Ensure alpha is compatible type if needed (Parameter(float) gives float)
            # If alpha could be a Tensor: alpha_val = alpha.item() if alpha.numel() == 1 else alpha
            return (1.0 - alpha) * a + alpha * b
        else:
            # logger.debug(f"Returning unmodified 'a' for key: {key}") # Optional logging
            # Return 'a' unmodified for all other keys
            return a

    # @staticmethod
    # @merge_method
    # def weighted_sum_01(
    #         a: Parameter(Tensor, "weight"),
    #         b: Parameter(Tensor, "weight"),
    #         *,
    #         alpha: Parameter(Tensor) =0.0,  # to 0 out non-selected blocks
    #         **kwargs,
    # ) -> Return(Tensor):
    #     return (1 - alpha) * a + alpha * b
    #
    # @staticmethod
    # @merge_method
    # def parallel_component(
    #         a: Parameter(Tensor, "weight"),
    #         b: Parameter(Tensor, "weight"),
    #         **kwargs,
    # ) -> Return(Tensor):
    #     """
    #     Calculates the component of tensor 'b' that is parallel to tensor 'a'.
    #
    #     Returns a tensor that represents the projection of 'b' onto 'a'.
    #     If the result contains NaN values, returns a zero tensor of the same shape.
    #     """
    #     norm_a = torch.linalg.norm(a)
    #     res = a * (a / norm_a * (b / norm_a)).sum()
    #     if res.isnan().any():
    #         return torch.zeros_like(a)
    #     return res
    #
    # @staticmethod
    # @merge_method
    # def determinant_sum(
    #         a: Parameter(Tensor, "weight"),
    #         b: Parameter(Tensor, "weight"),
    #         *,
    #         alpha: Parameter(Tensor) =0.5,
    #
    #         **kwargs,
    # ) -> Return(Tensor):
    #     key = kwargs["key"]
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
    #             vs.append(MergeMethods.determinant_sum.__wrapped__(k_a, k_b, **k_kwargs))
    #         return torch.cat(vs)
    #
    #     if key.endswith("bias"):
    #         return sd_mecha.merge_methods.weighted_sum.__wrapped__(a, b, alpha=alpha)
    #
    #     is_conv_3x3 = len(a.shape) == 4 and a.shape[-1] != 1
    #     is_conv_1x1 = len(a.shape) == 4 and a.shape[-1] == 1
    #     original_shape = a.shape
    #     if is_conv_3x3:
    #         shape_2d = (-1, functools.reduce(operator.mul, a.shape[1:]))
    #     elif is_conv_1x1:
    #         shape_2d = (-1, functools.reduce(operator.mul, a.shape[1:]))
    #     elif not a.shape:
    #         shape_2d = (1, 1)
    #     else:
    #         shape_2d = (-1, a.shape[-1])
    #
    #     a_neurons = a.reshape(*shape_2d)
    #     b_neurons = b.reshape(*shape_2d)
    #
    #     svd_driver = "gesvdj" if a.is_cuda else None
    #
    #     # Cache handling
    #     if cache is not None:
    #         key = kwargs["key"]
    #         if key not in cache:
    #             cache[key] = {}
    #         cache = cache[key]
    #
    #     if cache is not None and "a_s" in cache and "b_s" in cache:
    #         a_s = cache["a_s"].to(a.device, a.dtype)
    #         b_s = cache["b_s"].to(a.device, a.dtype)
    #     else:
    #         a_s = torch.linalg.svdvals(a_neurons, driver=svd_driver)
    #         b_s = torch.linalg.svdvals(b_neurons, driver=svd_driver)
    #
    #         if cache is not None:
    #             cache["a_s"] = a_s.to("cpu")
    #             cache["b_s"] = b_s.to("cpu")
    #
    #     ab_neurons = a_neurons * (1 - alpha) + b_neurons * alpha
    #     ab_s = torch.linalg.svdvals(ab_neurons, driver=svd_driver)
    #
    #     def pdet(s):
    #         return (s.log().sum() / len(s)).exp()
    #
    #     a_pdet = pdet(a_s)
    #     b_pdet = pdet(b_s)
    #     ab_pdet = pdet(ab_s)
    #
    #     ab_rescale = torch.nan_to_num(a_pdet ** (1 - alpha) * b_pdet ** alpha / ab_pdet, nan=1, posinf=1)
    #
    #     return (a * (1 - alpha) + b * alpha) * ab_rescale
    #
    # @staticmethod
    # @merge_method
    # def wavelet_merge(
    #         a: Parameter(Tensor, "weight"),
    #         b: Parameter(Tensor, "weight"),
    #         *,
    #         alpha: Parameter(Tensor) =0.5,
    #         **kwargs,
    # ) -> Return(Tensor):
    #     key = kwargs["key"]
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
    #             vs.append(MergeMethods.wavelet_merge.__wrapped__(k_a, k_b, **k_kwargs))
    #         return torch.cat(vs)
    #
    #     # Reshape tensors to 2D
    #     is_conv_3x3 = len(a.shape) == 4 and a.shape[-1] != 1
    #     is_conv_1x1 = len(a.shape) == 4 and a.shape[-1] == 1
    #     original_shape = a.shape
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
    #     dwt = DWTForward(J=4, wave='db4', mode='zero')
    #     idwt = DWTInverse(wave='db4', mode='zero')
    #
    #     dwt.to(device=a.device, dtype=a.dtype)
    #     idwt.to(device=a.device, dtype=a.dtype)
    #
    #     a_yl, a_yh = dwt(a.unsqueeze(0).unsqueeze(0))
    #     b_yl, b_yh = dwt(b.unsqueeze(0).unsqueeze(0))
    #
    #     merged_detail = alpha * a_yl + (1 - alpha) * b_yl, [alpha * aa + (1 - alpha) * bb for aa, bb in zip(a_yh, b_yh)]
    #
    #     merged_tensor = idwt(merged_detail).squeeze(0).squeeze(0)
    #     merged_tensor = merged_tensor[:shape_2d[0], :shape_2d[1]]
    #
    #     return merged_tensor.reshape(original_shape)
    #
    # @staticmethod
    # @merge_method
    # def multi_domain_alignment(
    #         a: Parameter(Tensor, "weight"),
    #         b: Parameter(Tensor, "weight"),
    #         c: Parameter(Tensor, "weight"),
    #         *,
    #         alpha: Parameter(Tensor) =0.5,
    #         beta: Parameter(Tensor) =0.5,
    #         kernel_size: Parameter(int) = 3,
    #         centroid_margin_factor: Parameter(Tensor) =0.08,
    #         frequency_weight: Parameter(Tensor) =0.4,
    #         use_cross_attention: float = 1.0,
    #         **kwargs,
    # ) -> Return(Tensor):
    #     try:
    #         with torch.no_grad():  # Prevent gradient graph building for intermediate calculations
    #             if not (0 <= alpha <= 1 and 0 <= beta <= 1):
    #                 raise ValueError("Alpha and beta must be between 0 and 1")
    #
    #             key = kwargs["key"]
    #             if key.endswith(("in_proj_weight", "in_proj_bias")):
    #                 return MergeMethods.handle_attention_projection(a, b, c, alpha, beta, kwargs)
    #
    #             original_shape = a.shape
    #
    #             # Step 1: Frequency domain alignment in isolated context
    #             freq_aligned_b = torch.utils.checkpoint.checkpoint(
    #                 MergeMethods.frequency_selective_alignment,
    #                 a, b, c,
    #                 centroid_margin_factor,
    #                 use_reentrant=False
    #             )
    #
    #             # Step 2: Spatial domain processing
    #             shape_2d = MergeMethods.determine_reshape_dimensions(a)
    #             a_2d = a.reshape(*shape_2d)
    #             b_2d = b.reshape(*shape_2d)
    #             c_2d = c.reshape(*shape_2d)
    #             freq_aligned_b_2d = freq_aligned_b.reshape(*shape_2d)
    #
    #             # Calculate importance weights using cross-attention if enabled
    #             if use_cross_attention > 0 and min(shape_2d) > 1:
    #                 importance_weights = torch.utils.checkpoint.checkpoint(
    #                     MergeMethods.calculate_cross_attention,
    #                     a_2d.detach(), b_2d.detach(), c_2d.detach(),
    #                     use_reentrant=False
    #                 )
    #             else:
    #                 importance_weights = torch.ones_like(a_2d)
    #
    #             # Calculate dissimilarity with anchor using checkpointing
    #             dissimilarity = torch.utils.checkpoint.checkpoint(
    #                 MergeMethods.calculate_dissimilarity,
    #                 a_2d.detach(), b_2d.detach(), c_2d.detach(),
    #                 use_reentrant = False
    #             )
    #
    #             dissimilarity = MergeMethods.gaussian_blur(dissimilarity, kernel_size)
    #
    #             # Combine frequency and spatial information
    #             b_combined = (
    #                     freq_aligned_b_2d * frequency_weight +
    #                     b_2d * (1 - frequency_weight)
    #             )
    #
    #             # Vectorized SLERP implementation
    #             effective_alpha = alpha * importance_weights
    #
    #             # Normalize vectors
    #             a_norm = F.normalize(a_2d, p=2, dim=-1)
    #             b_norm = F.normalize(b_combined, p=2, dim=-1)
    #
    #             # Compute dot product
    #             dot_product = (a_norm * b_norm).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    #             omega = torch.acos(dot_product)
    #
    #             # Handle small angles to prevent numerical instability
    #             small_angle_mask = omega < 1e-4
    #             sin_omega = torch.sin(omega).clamp_min(1e-6)
    #
    #             # Compute SLERP coefficients
    #             slerp_a = torch.where(small_angle_mask,
    #                                   1.0 - effective_alpha,
    #                                   torch.sin((1.0 - effective_alpha) * omega) / sin_omega)
    #             slerp_b = torch.where(small_angle_mask,
    #                                   effective_alpha,
    #                                   torch.sin(effective_alpha * omega) / sin_omega)
    #
    #             # Compute merged result
    #             merged = slerp_a * a_2d + slerp_b * b_combined
    #
    #             # Apply anchor-based adjustment
    #             anchor_adjustment = (b_combined - c_2d) * beta * dissimilarity
    #             merged = merged + anchor_adjustment * importance_weights
    #
    #             result = merged.reshape(original_shape)
    #
    #             # Ensure all intermediate tensors are cleared
    #             del (freq_aligned_b, a_2d, b_2d, c_2d, freq_aligned_b_2d, importance_weights,
    #                  dissimilarity, b_combined, a_norm, b_norm, dot_product, omega,
    #                  small_angle_mask, sin_omega, slerp_a, slerp_b, merged, anchor_adjustment)
    #
    #             return result
    #
    #     finally:
    #         # Clear any CUDA cache if using GPU
    #         if torch.cuda.is_available():
    #             torch.cuda.empty_cache()
    #
    # def handle_attention_projection(
    #         a: Tensor,
    #         b: Tensor,
    #         c: Tensor,
    #         alpha: float,
    #         beta: float,
    #         kwargs: dict
    # ) -> Tensor:
    #     """Handle the special case of attention projection layers."""
    #     vs = []
    #     for i, k in enumerate(("to_q", "to_k", "to_v")):
    #         k_kwargs = kwargs.copy()
    #         k_kwargs["key"] = kwargs["key"].replace("in_proj_", f"{k}.")
    #         dim = a.shape[0] // 3
    #         t_start = dim * i
    #         t_end = dim * (i + 1)
    #         vs.append(
    #             MergeMethods.multi_domain_alignment.__wrapped__(
    #                 a[t_start:t_end],
    #                 b[t_start:t_end],
    #                 c[t_start:t_end],
    #                 alpha=alpha,
    #                 beta=beta,
    #                 **k_kwargs
    #             )
    #         )
    #     return torch.cat(vs)
    #
    # def determine_reshape_dimensions(tensor: Tensor) -> tuple:
    #     """Determine the appropriate reshape dimensions based on tensor type."""
    #     if not tensor.shape:
    #         return (1, 1)
    #
    #     is_conv = len(tensor.shape) == 4
    #     if is_conv:
    #         return (-1, functools.reduce(operator.mul, tensor.shape[1:]))
    #     return (-1, tensor.shape[-1])
    #
    # def calculate_cross_attention(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    #     """Calculate feature importance using cross-attention mechanism."""
    #     # Normalize inputs
    #     a_norm = F.normalize(a, dim=-1)
    #     b_norm = F.normalize(b, dim=-1)
    #     c_norm = F.normalize(c, dim=-1)
    #
    #     # Calculate attention scores
    #     attn_ab = torch.matmul(a_norm, b_norm.transpose(-2, -1))
    #     attn_ac = torch.matmul(a_norm, c_norm.transpose(-2, -1))
    #     attn_bc = torch.matmul(b_norm, c_norm.transpose(-2, -1))
    #
    #     # Softmax for probability distribution
    #     attn_ab = F.softmax(attn_ab / math.sqrt(a.size(-1)), dim=-1)
    #     attn_ac = F.softmax(attn_ac / math.sqrt(a.size(-1)), dim=-1)
    #     attn_bc = F.softmax(attn_bc / math.sqrt(a.size(-1)), dim=-1)
    #
    #     # Calculate feature importance based on attention patterns
    #     importance = (
    #                          torch.sum(attn_ab, dim=-1, keepdim=True) +
    #                          torch.sum(attn_ac, dim=-1, keepdim=True) +
    #                          torch.sum(attn_bc, dim=-1, keepdim=True)
    #                  ) / 3.0
    #
    #     # Normalize importance scores
    #     importance = F.normalize(importance, dim=0)
    #     return importance
    #
    # def calculate_dissimilarity(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    #     """Calculate dissimilarity between tensors with anchor guidance."""
    #     diff_a_c = a - c
    #     diff_b_c = b - c
    #
    #     norm_a = diff_a_c.norm(dim=1, keepdim=True)
    #     norm_b = diff_b_c.norm(dim=1, keepdim=True)
    #
    #     # Use maximum of norms for normalization
    #     threshold = torch.max(norm_a, norm_b)
    #
    #     # Calculate cosine similarity with improved numerical stability
    #     cos_sim = torch.nan_to_num(
    #         (diff_a_c * diff_b_c).sum(dim=1, keepdim=True) / (threshold ** 2 + EPSILON),
    #         nan=0
    #     )
    #
    #     return (1 - cos_sim) / 2
    #
    # @staticmethod
    # def frequency_selective_alignment(
    #         a: Tensor,
    #         b: Tensor,
    #         c: Tensor,
    #         centroid_margin_factor: float = 0.1
    # ) -> Tensor:
    #     """Frequency selective alignment with improved memory management."""
    #     with torch.no_grad():
    #         # Reshape tensors
    #         a_flat = a.reshape(-1).float()
    #         b_flat = b.reshape(-1).float()
    #         c_flat = c.reshape(-1).float()
    #
    #         # Compute FFTs one at a time to reduce peak memory usage
    #         a_dft = torch.fft.rfft(a_flat)
    #         b_dft = torch.fft.rfft(b_flat)
    #         c_dft = torch.fft.rfft(c_flat)
    #
    #         fft_size = a_dft.shape[0]
    #
    #         # Calculate centroids sequentially
    #         centroids = {
    #             'a': MergeMethods.calculate_spectral_centroid(a_dft),
    #             'b': MergeMethods.calculate_spectral_centroid(b_dft),
    #             'c': MergeMethods.calculate_spectral_centroid(c_dft)
    #         }
    #
    #         # Calculate phase coherence
    #         phase_coherence = MergeMethods.calculate_phase_coherence(a_dft, b_dft, c_dft)
    #
    #         # Dynamic beta calculation
    #         freq_dissimilarity = abs(centroids['a'] - centroids['b'])
    #         dynamic_beta = torch.cos(torch.tensor(math.pi / 2) * freq_dissimilarity).item()
    #         dynamic_beta = dynamic_beta * phase_coherence
    #
    #         # Define frequency bands
    #         margin = int(centroid_margin_factor * fft_size)
    #         passband_end = int(min(centroids['a'], centroids['c']) * fft_size - margin)
    #         stopband_start = int(max(centroids['a'], centroids['c']) * fft_size + margin)
    #
    #         passband_end = max(0, min(passband_end, fft_size - margin))
    #         stopband_start = min(fft_size, max(stopband_start, margin))
    #
    #         # Adjust frequency components
    #         result = MergeMethods.adjust_frequency_components(
    #             a_dft, b_dft, c_dft,
    #             passband_end, stopband_start,
    #             dynamic_beta
    #         )
    #
    #         # Clean up FFT tensors explicitly
    #         del a_dft, b_dft, c_dft
    #
    #         return torch.fft.irfft(result, a_flat.shape[0]).reshape(a.shape)
    #
    # def calculate_phase_coherence(a_dft: Tensor, b_dft: Tensor, c_dft: Tensor) -> float:
    #     """Calculate phase coherence between three signals."""
    #     phase_a = torch.angle(a_dft)
    #     phase_b = torch.angle(b_dft)
    #     phase_c = torch.angle(c_dft)
    #
    #     # Calculate phase differences
    #     diff_ab = torch.abs(torch.angle(torch.exp(1j * (phase_a - phase_b))))
    #     diff_ac = torch.abs(torch.angle(torch.exp(1j * (phase_a - phase_c))))
    #     diff_bc = torch.abs(torch.angle(torch.exp(1j * (phase_b - phase_c))))
    #
    #     # Average phase coherence
    #     coherence = torch.mean(torch.cos(diff_ab) + torch.cos(diff_ac) + torch.cos(diff_bc)) / 3
    #     return coherence.item()
    #
    # def adjust_frequency_components(
    #         a_dft: Tensor,
    #         b_dft: Tensor,
    #         c_dft: Tensor,  # Keep parameter for API consistency, but use minimally
    #         passband_end: int,
    #         stopband_start: int,
    #         dynamic_beta: float
    # ) -> Tensor:
    #     """
    #     Adjust magnitude and phase of frequency components.
    #     The anchor tensor (c_dft) is used only for band definition in the caller,
    #     not for direct magnitude/phase adjustment.
    #     """
    #     # Separate magnitude and phase
    #     mag_b = torch.abs(b_dft)
    #     phase_b = torch.angle(b_dft)
    #
    #     # Get reference magnitudes
    #     mag_a = torch.abs(a_dft)
    #
    #     # Calculate weighted magnitude
    #     weighted_mag = torch.where(
    #         torch.arange(mag_b.shape[0], device=mag_b.device) < passband_end,
    #         (1 - dynamic_beta) * mag_a + dynamic_beta * mag_b,
    #         mag_b
    #     )
    #
    #     # Apply smooth transition only if there's a valid transition range
    #     transition_range = stopband_start - passband_end
    #     if transition_range > 0:
    #         transition = torch.cos(
    #             torch.linspace(0, math.pi / 2, transition_range, device=mag_b.device)
    #         )
    #         weighted_mag[passband_end:stopband_start] *= transition
    #
    #     return torch.polar(weighted_mag, phase_b)
    #
    # def calculate_spectral_centroid(dft: Tensor) -> float:
    #     """
    #     Calculates the spectral centroid of a tensor in the frequency domain.
    #     Returns a normalized centroid value between 0 and 1.
    #     """
    #     fft_size = dft.shape[0]
    #     frequencies = torch.arange(fft_size, device=dft.device) / fft_size  # Normalize frequencies to [0, 1]
    #     magnitudes = torch.abs(dft)
    #     centroid = (frequencies * magnitudes).sum() / (magnitudes.sum() + EPSILON)
    #     return centroid.item()
    #
    # def gaussian_blur(a: Tensor, kernel_size: int) -> Tensor:
    #     """
    #     Apply 1D Gaussian blur to tensor with handling for small tensors.
    #     Automatically adjusts kernel size for small inputs to prevent padding errors.
    #     """
    #     # Ensure input is at least 2D
    #     if len(a.shape) == 1:
    #         a = a.unsqueeze(0)
    #
    #     # Adjust kernel size if it's too large for the input
    #     min_dim = min(a.shape)
    #     if kernel_size > min_dim:
    #         # Use the largest odd number that's smaller than the minimum dimension
    #         kernel_size = max(3, min_dim - (min_dim % 2 == 0))
    #
    #     # Ensure kernel size is odd
    #     if kernel_size % 2 == 0:
    #         kernel_size -= 1
    #
    #     # Skip blur for very small tensors
    #     if kernel_size < 3:
    #         return a.squeeze() if len(a.shape) > 1 else a
    #
    #     sigma = kernel_size / 3
    #     x = torch.arange(kernel_size, device=a.device) - (kernel_size - 1) / 2
    #     kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    #     kernel = kernel / kernel.sum()
    #     kernel = kernel.view(1, 1, -1)
    #
    #     pad_size = kernel_size // 2
    #     # Use replication padding for very small tensors where reflection wouldn't work
    #     padding_mode = 'replicate' if min_dim <= pad_size * 2 else 'reflect'
    #
    #     padded = F.pad(a.unsqueeze(1), (pad_size, pad_size), mode=padding_mode)
    #     blurred = F.conv1d(padded.double(), kernel.double()).squeeze(1)
    #
    #     # Return to original shape
    #     return blurred.squeeze() if len(a.shape) == 1 else blurred
    #
    # def calculate_feature_importance(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    #     """Calculate feature importance using attention and gradient information."""
    #     # Normalize inputs
    #     a_norm = F.normalize(a, dim=-1)
    #     b_norm = F.normalize(b, dim=-1)
    #     c_norm = F.normalize(c, dim=-1)
    #
    #     # Calculate attention scores
    #     attn_ab = torch.matmul(a_norm, b_norm.transpose(-2, -1))
    #     attn_ac = torch.matmul(a_norm, c_norm.transpose(-2, -1))
    #
    #     # Calculate feature importance
    #     importance = (
    #                          torch.sum(torch.abs(attn_ab), dim=-1) +
    #                          torch.sum(torch.abs(attn_ac), dim=-1)
    #                  ) / 2
    #
    #     # Normalize importance scores
    #     return F.softmax(importance, dim=-1).unsqueeze(-1)

    @merge_method
    def pop_lora(
            a: Parameter(Tensor, "weight"),
            b: Parameter(Tensor, "weight"),
            *,
            alpha: Parameter(Tensor) = 0.5,
            rank_ratio: Parameter(float) = 0.25,
            early_exit: Parameter(bool) = True,
            **kwargs
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
                w = (v @ R_sub)
                R[k:, k:] = R_sub - (v.unsqueeze(1) @ (tau * w).unsqueeze(0))

                Q_sub = Q[:, k:]
                wq = Q_sub @ v
                Q[:, k:] = Q_sub - (wq.unsqueeze(1) @ (tau * v).unsqueeze(0))

                if k + 1 < m:
                    R[k + 1:, k] = 0

                if k + 1 < n:
                    col_norms[k + 1:] = torch.clamp(col_norms[k + 1:] - R[k, k + 1:].to(torch.float64).pow(2), min=0.0)
                    if (k % 8) == 7 or k == 0:
                        col_norms[k + 1:] = (R[k:, k + 1:].to(torch.float32).pow(2).sum(dim=0)).to(torch.float64)

            return Q, R, piv

        def _unpivot_R(R: torch.Tensor, piv: torch.Tensor, n_cols: int) -> torch.Tensor:
            # FIX: Use proper torch.zeros dimensions
            R_unp = torch.zeros(R.size(0), n_cols, device=R.device, dtype=R.dtype)
            R_unp[:, piv] = R
            return R_unp

        # 1) Basis from a (QR) - NO CACHE, recompute each time
        Qa, _Ra = torch.linalg.qr(a_2d, mode='reduced')

        # 2) Project difference - NO CACHE, recompute each time
        diff = b_2d - a_2d
        projected_diff = Qa @ (Qa.T @ diff) if Qa.shape[1] > 0 else torch.zeros_like(diff)

        # 3) CPQR on projected difference - CACHE ONLY THIS
        if projected_diff.numel() == 0:
            low_rank_diff = torch.zeros_like(projected_diff)
        else:
            if layer_cache is not None and 'Qd' in layer_cache and 'R_unp' in layer_cache:
                Qd = layer_cache['Qd'].to(device=a.device, dtype=a.dtype)
                R_unp = layer_cache['R_unp'].to(device=a.device, dtype=a.dtype)
            else:
                Qd, Rd, piv = _cpqr(projected_diff)
                R_unp = _unpivot_R(Rd, piv, projected_diff.shape[1])
                if layer_cache is not None:
                    layer_cache['Qd'] = Qd.detach().cpu()
                    layer_cache['R_unp'] = R_unp.detach().cpu()

            max_rank = min(Qd.shape[1], R_unp.shape[0], R_unp.shape[1])
            r = max(1, int(max_rank * float(rank_ratio)))
            r = min(r, max_rank)
            low_rank_diff = Qd[:, :r] @ R_unp[:r, :]

        alpha_f = float(alpha) if isinstance(alpha, torch.Tensor) else alpha
        merged = a_2d + alpha_f * low_rank_diff
        return merged.reshape(original_shape)

    @merge_method
    def delta_widen(
            *deltas: Parameter(Tensor, "delta"),  # subtract
            magnitude_ratio: Parameter(float) = 2.0,  # weight of magnitude divergence
            direction_ratio: Parameter(float) = 2.0,  # weight of directional divergence
            temperature: Parameter(float) = 1.0,  # softmax sharpness
            critical_quantile: Parameter(float) = 0.80,  # pooled per-parameter threshold across models
            topk: Parameter(int) = 0,  # 0=off; 1..M enables per-column top-k gating
            rank_blend: Parameter(float) = 0.0,  # 0=off; 0.3–0.6 blends rank with raw divergences
            baseline_index: Parameter(int) = -1,  # -1 = zero baseline; >=0 = anchor model
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
            raise ValueError("At least one model delta is required. [delta_widen_sd]")

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
                mags.append(Wdk.abs().reshape(-1))  # magnitude-only for 1D
                dirs.append(None)
            else:
                mcol, D = components_dk(Wdk)
                mags.append(mcol)
                dirs.append(D)

        ref_dk, ref_meta = to_dk(ref)
        if ref_meta[1] in ("scalar", "vec"):
            ref_mag = ref_dk.abs().reshape(-1)
            ref_dir = None
        else:
            ref_mag, ref_dir = components_dk(ref_dk)

        # Compute divergences per column j
        mag_divs, dir_divs = [], []
        for i in range(M):
            md = (mags[i] - ref_mag).abs()
            mag_divs.append(md)
            if dirs[i] is None or ref_dir is None:
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
            mask = torch.full_like(logits, float('-inf'))
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

    @merge_method
    def rams(
            *deltas: Parameter(Tensor, "delta"),
            outlier_tolerance: Parameter(float) = 2.5,
            outlier_intensity: Parameter(float) = 1.0,
            memory_safety_margin: Parameter(float) = 0.85,
            use_adaptive_tolerance: Parameter(float) = 1.0,
            use_geometric_median: Parameter(float) = 0.0,
            **kwargs,
    ) -> Return(Tensor, "delta"):
        """
        Identifies outlier parameters and blends them based on a robust statistical
        framework, while giving the user a clear, powerful choice for how to handle
        the most extreme disagreements.

        Args:
            *deltas (Tensor): A variable number of input tensors (deltas) for the same layer.
            core_indices (list[int]): A list of indices specifying the 'trusted' models that
                                      form the statistical baseline for the merge.
            outlier_tolerance (float): The base sensitivity for outlier detection. Higher values
                                       are more tolerant, leading to fewer outliers.
            outlier_intensity (float): Controls the influence of outlier parameters. Higher values
                                       give outliers more strength in the final blend.
            memory_safety_margin (float): The percentage of free VRAM to use for processing chunks.
                                          Prevents CUDA OOM errors automatically.
            use_adaptive_tolerance (bool): If True, automatically adjusts `outlier_tolerance` based
                                           on the statistical variance of each layer.
            use_geometric_median (bool): The primary user control. If False (default), uses a fast,
                                         weighted influence blend for outliers. If True, uses the
                                         slower but ultra-robust geometric median for outliers.
            **kwargs: Catches any unused parameters from the framework.

        Returns:
            Tensor: The final, merged tensor for the layer.
        """
        # A little safety net!
        if not deltas:
            raise ValueError("Onii-chan, you have to give me tensors to merge!")

        core_indices = [1, 3, 4, 6, 7, 8, 9, 10]
        device = deltas[0].device
        epsilon_tensor = torch.tensor(1e-8, device=device)

        # --- 1. The Correct Architecture: Pre-allocate the Canvas ---
        final_merged_tensor = torch.zeros_like(deltas[0])

        # --- 2. Dynamic Chunk Size Calculation (No Multiplier!) ---
        # We go back to the simple calculation because we're cleaning as we go.
        # The largest single allocation will be the chunk_stack.
        if device.type == 'cuda':
            available_vram, _ = torch.cuda.mem_get_info(device)
            memory_budget = available_vram * memory_safety_margin
            cost_per_element_in_stack = len(deltas) * deltas[0].element_size()
            dynamic_chunk_size = max(1, int(memory_budget // cost_per_element_in_stack))
        else:
            dynamic_chunk_size = 2 ** 22

        total_elements = deltas[0].numel()

        # --- 3. Main Processing Loop (with Aggressive Cleanup) ---
        for i in range(0, total_elements, dynamic_chunk_size):
            end = min(i + dynamic_chunk_size, total_elements)
            # We operate on slices of the input deltas.
            chunk_deltas = [d.flatten()[i:end] for d in deltas]
            chunk_stack = torch.stack(chunk_deltas)
            del chunk_deltas

            core_chunks = chunk_stack[core_indices]
            core_median, _ = torch.median(core_chunks, dim=0)

            current_tolerance = outlier_tolerance
            if use_adaptive_tolerance > 0.0:
                layer_complexity = torch.std(core_chunks, dim=0)
                adaptive_factor = torch.clamp(1 + 0.1 * torch.log(layer_complexity + epsilon_tensor), min=0.5, max=3.0)
                current_tolerance *= adaptive_factor
                del layer_complexity, adaptive_factor

            q1, q3 = torch.quantile(core_chunks, torch.tensor([0.25, 0.75], device=device), dim=0)
            iqr_core = q3 - q1
            del q1, q3
            mad_core, _ = torch.median(torch.abs(core_chunks - core_median), dim=0)

            lower_bound = core_median - (current_tolerance * iqr_core)
            upper_bound = core_median + (current_tolerance * iqr_core)
            mad_bound = current_tolerance * 1.4826 * mad_core
            del mad_core

            is_outlier_iqr = (chunk_stack < lower_bound) | (chunk_stack > upper_bound)
            del lower_bound, upper_bound
            is_outlier_mad = torch.abs(chunk_stack - core_median) > mad_bound.clamp(min=epsilon_tensor)
            del mad_bound
            is_outside_bounds = is_outlier_iqr | is_outlier_mad
            del is_outlier_iqr, is_outlier_mad

            if torch.mean(is_outside_bounds.float()) < 0.01:
                final_chunk_masked = core_median
            else:
                disagreement = chunk_stack - core_median
                robust_z_score = disagreement.div(iqr_core + epsilon_tensor)
                del disagreement
                influence_scores = outlier_intensity * torch.tanh(torch.abs(robust_z_score))
                del robust_z_score
                masked_influence_scores = influence_scores * is_outside_bounds.float()

                if use_geometric_median > 0.0:
                    outlier_mask = torch.any(is_outside_bounds, dim=1)
                    if torch.any(outlier_mask):
                        outlier_points = chunk_stack[outlier_mask]
                        outlier_blend = MergeMethods._rams_geometric_median(outlier_points)
                        del outlier_points
                    else:
                        outlier_blend = core_median
                    del outlier_mask
                else:
                    sum_of_masked_influences = torch.sum(masked_influence_scores, dim=0)
                    masked_weighted_deltas = chunk_stack * masked_influence_scores
                    sum_of_masked_contributions = torch.sum(masked_weighted_deltas, dim=0)
                    outlier_blend = sum_of_masked_contributions / (sum_of_masked_influences + epsilon_tensor)
                    del sum_of_masked_influences, masked_weighted_deltas, sum_of_masked_contributions

                avg_outlier_influence = torch.sum(masked_influence_scores, dim=0) / (
                    torch.sum(is_outside_bounds.float(), dim=0).clamp(min=1))
                del masked_influence_scores, influence_scores

                final_chunk = (1.0 - avg_outlier_influence) * core_median + avg_outlier_influence * outlier_blend
                del avg_outlier_influence, outlier_blend

                disagreement_mask = torch.any(is_outside_bounds, dim=0)
                final_chunk_masked = torch.where(disagreement_mask, final_chunk, core_median)
                del final_chunk, disagreement_mask

            final_merged_tensor.flatten()[i:end] = final_chunk_masked

            # --- Final Manual Cleanup at the end of each chunk iteration ---
            del chunk_stack, core_chunks, core_median, iqr_core, is_outside_bounds, final_chunk_masked
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        return final_merged_tensor.reshape(deltas[0].shape)

    @staticmethod
    def _rams_geometric_median(
            points: Tensor, eps: float = 1e-8, maxiter: int = 100, ftol: float = 1e-5, chunk_size: int = 1024
    ) -> Tensor:
        """
        Computes the geometric median for a set of tensors with robust optimizations.

        The geometric median is the point minimizing the sum of Euclidean distances to the
        sample points. It's a highly robust estimator of central tendency. This implementation
        is optimized for CUDA environments with chunking and specific edge cases.

        Args:
            points (Tensor): A tensor of points, where the first dimension is the number of points.
            eps (float): A small epsilon for numerical stability.
            maxiter (int): The maximum number of iterations for the Weiszfeld algorithm.
            ftol (float): The tolerance for convergence.
            chunk_size (int): The number of points to process in each chunk for memory efficiency.

        Returns:
            Tensor: The geometric median of the input points.
        """
        # --- Edge Case Handling ---
        n_points, *dims = points.shape
        if n_points == 0:
            return torch.empty((0, *dims), device=points.device)
        if n_points == 1:
            return points[0]
        # Pro Optimization: The median of 2 points is their midpoint.
        if n_points == 2:
            return torch.mean(points, dim=0)

        device = points.device
        # Use reshape for safety, as it handles non-contiguous tensors automatically.
        median = torch.mean(points.reshape(n_points, -1), dim=0)

        # --- Iterative Weiszfeld's Algorithm ---
        for _ in range(maxiter):
            prev_median = median.clone()
            weighted_sum = torch.zeros_like(median)
            weight_sum = torch.zeros_like(median)

            # Process in chunks to save VRAM
            for i in range(0, n_points, chunk_size):
                chunk = points[i:i + chunk_size].reshape(-1, median.shape[0])
                chunk_dist = torch.norm(chunk - median, dim=1)
                # Pro Optimization: Improved numerical stability for weights
                weights = 1.0 / (chunk_dist + eps)

                weighted_sum.add_(torch.sum(chunk * weights[:, None], dim=0))
                weight_sum.add_(torch.sum(weights))

            median = weighted_sum / weight_sum.clamp(min=eps)

            # Check for convergence
            if torch.norm(median - prev_median) < ftol:
                break

        return median.reshape(*dims)

    @merge_method
    def rams_pro(
            *deltas: Parameter(Tensor, "delta"),
            outlier_tolerance: Parameter(float) = 2.5,
            outlier_intensity: Parameter(float) = 1.0,
            memory_safety_margin: Parameter(float) = 0.85,
            use_adaptive_tolerance: Parameter(float) = 1.0,
            use_geometric_median: Parameter(float) = 1.0,
            **kwargs,
    ) -> Return(Tensor, "delta"):
        """
        Production-ready RAMS with robust memory management and hanging prevention.
        Incorporates lessons from enterprise-grade merge methods.
        """
        if not deltas:
            raise ValueError("At least one delta tensor must be provided!")

        if outlier_tolerance == 0.0 and outlier_intensity == 0.0:
            return torch.zeros_like(deltas[0])

        core_indices = [1, 3, 4, 6, 7, 8, 9, 10]
        device = deltas[0].device
        epsilon_tensor = torch.tensor(1e-8, device=device)

        # 🔧 ONLY FIX: Better chunk size calculation with minimum bounds
        if device.type == 'cuda':
            available_vram, _ = torch.cuda.mem_get_info(device)
            memory_budget = available_vram * memory_safety_margin
            cost_per_element_in_stack = len(deltas) * deltas[0].element_size()
            calculated_chunk_size = max(1, int(memory_budget // cost_per_element_in_stack))

            # 🛡️ Prevent tiny chunks that cause hanging
            min_chunk = max(1024, deltas[0].nelement() // 50000)  # Max 50K chunks
            dynamic_chunk_size = max(min_chunk, calculated_chunk_size)
        else:
            dynamic_chunk_size = max(2 ** 20, deltas[0].nelement() // 10000)

        total_elements = deltas[0].numel()

        # 🔧 MINIMAL FIX: Safer median calculation (only change here)
        def safe_median(core_chunks):
            # Just add a simple fallback for tiny chunks
            if core_chunks.shape[1] < 5:
                return torch.mean(core_chunks, dim=0)

            # Original median calculation
            core_median, _ = torch.median(core_chunks, dim=0)
            return core_median

        # 🔧 BACK TO ORIGINAL: GPU-based result tensor (like your working version)
        final_merged_tensor = torch.zeros_like(deltas[0])

        for i in range(0, total_elements, dynamic_chunk_size):
            end = min(i + dynamic_chunk_size, total_elements)

            # 🔧 EXACTLY like your original working version
            chunk_deltas = [d.flatten()[i:end] for d in deltas]
            chunk_stack = torch.stack(chunk_deltas)
            del chunk_deltas

            core_chunks = chunk_stack[core_indices]

            # 🔧 ONLY CHANGE: Use safe median instead of direct median
            core_median = safe_median(core_chunks)

            # 🔧 EVERYTHING ELSE: Exactly like your original working version
            current_tolerance = outlier_tolerance
            if use_adaptive_tolerance > 0.0:
                layer_complexity = torch.std(core_chunks, dim=0)
                adaptive_factor = torch.clamp(1 + 0.1 * torch.log(layer_complexity + epsilon_tensor), min=0.5, max=3.0)
                current_tolerance *= adaptive_factor
                del layer_complexity, adaptive_factor

            q1, q3 = torch.quantile(core_chunks, torch.tensor([0.25, 0.75], device=device), dim=0)
            iqr_core = q3 - q1
            del q1, q3
            mad_core, _ = torch.median(torch.abs(core_chunks - core_median), dim=0)

            lower_bound = core_median - (current_tolerance * iqr_core)
            upper_bound = core_median + (current_tolerance * iqr_core)
            mad_bound = current_tolerance * 1.4826 * mad_core
            del mad_core

            is_outlier_iqr = (chunk_stack < lower_bound) | (chunk_stack > upper_bound)
            del lower_bound, upper_bound
            is_outlier_mad = torch.abs(chunk_stack - core_median) > mad_bound.clamp(min=epsilon_tensor)
            del mad_bound
            is_outside_bounds = is_outlier_iqr | is_outlier_mad
            del is_outlier_iqr, is_outlier_mad

            if torch.mean(is_outside_bounds.float()) < 0.01:
                final_chunk_masked = core_median
            else:
                disagreement = chunk_stack - core_median
                robust_z_score = disagreement.div(iqr_core + epsilon_tensor)
                del disagreement
                influence_scores = outlier_intensity * torch.tanh(torch.abs(robust_z_score))
                del robust_z_score
                masked_influence_scores = influence_scores * is_outside_bounds.float()

                if use_geometric_median > 0.0:
                    outlier_mask = torch.any(is_outside_bounds, dim=1)
                    if torch.any(outlier_mask):
                        outlier_points = chunk_stack[outlier_mask]
                        outlier_blend = MergeMethods._rams_geometric_median_safe(outlier_points)
                        del outlier_points
                    else:
                        outlier_blend = core_median
                    del outlier_mask
                else:
                    sum_of_masked_influences = torch.sum(masked_influence_scores, dim=0)
                    masked_weighted_deltas = chunk_stack * masked_influence_scores
                    sum_of_masked_contributions = torch.sum(masked_weighted_deltas, dim=0)
                    outlier_blend = sum_of_masked_contributions / (sum_of_masked_influences + epsilon_tensor)
                    del sum_of_masked_influences, masked_weighted_deltas, sum_of_masked_contributions

                avg_outlier_influence = torch.sum(masked_influence_scores, dim=0) / (
                    torch.sum(is_outside_bounds.float(), dim=0).clamp(min=1))
                del masked_influence_scores, influence_scores

                final_chunk = (1.0 - avg_outlier_influence) * core_median + avg_outlier_influence * outlier_blend
                del avg_outlier_influence, outlier_blend

                disagreement_mask = torch.any(is_outside_bounds, dim=0)
                final_chunk_masked = torch.where(disagreement_mask, final_chunk, core_median)
                del final_chunk, disagreement_mask

            # 🔧 EXACTLY like your original working version
            final_merged_tensor.flatten()[i:end] = final_chunk_masked

            # 🔧 EXACTLY like your original working version
            del chunk_stack, core_chunks, core_median, iqr_core, is_outside_bounds, final_chunk_masked
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        # 🔧 EXACTLY like your original working version - NO value sanitization
        return final_merged_tensor.reshape(deltas[0].shape)

    @staticmethod
    def _rams_geometric_median_safe(points: Tensor, eps: float = 1e-8, maxiter: int = 150,
                                    ftol: float = 1e-5) -> Tensor:
        """Geometric median with just timeout protection - no other changes."""
        n_points, *dims = points.shape
        if n_points == 0:
            return torch.empty((0, *dims), device=points.device)
        if n_points == 1:
            return points[0]
        if n_points == 2:
            return torch.mean(points, dim=0)

        device = points.device
        median = torch.mean(points.reshape(n_points, -1), dim=0)

        for iteration in range(maxiter):
            prev_median = median.clone()
            weighted_sum = torch.zeros_like(median)
            weight_sum = torch.zeros_like(median)

            chunk_size = min(1024, n_points)
            for i in range(0, n_points, chunk_size):
                chunk = points[i:i + chunk_size].reshape(-1, median.shape[0])
                chunk_dist = torch.norm(chunk - median, dim=1) + eps
                weights = 1.0 / chunk_dist

                weighted_sum.add_(torch.sum(chunk * weights[:, None], dim=0))
                weight_sum.add_(torch.sum(weights))

            median = weighted_sum / weight_sum.clamp(min=eps)

            # 🔧 ONLY CHANGE: Timeout protection
            if torch.norm(median - prev_median) < ftol:
                break

            if iteration > 10:  # Reduced timeout
                break

        return median.reshape(*dims)

    # @staticmethod
    # @merge_method
    # def synthetic_fisher_merge(
    #         a: Parameter(Tensor, "weight"),
    #         b: Parameter(Tensor, "weight"),
    #         *,
    #         num_samples: Parameter(Tensor) =256,
    #         noise_scale: Parameter(Tensor) =1.5,
    #         epsilon: Parameter(Tensor) =1e-8,
    #         use_diverse: float = 1.0,
    #         **kwargs
    # ) -> Return(Tensor):
    #     """
    #     v1.51 Fisher merge using synthetic data with bias mitigation
    #
    #     Features:
    #     - Adversarial noise generation
    #     - Entropy maximization
    #     - Gradient clipping
    #     - Sign alignment
    #     """
    #     key = kwargs["key"]
    #
    #     # Handle scalar parameters
    #     if a.dim() == 0 or b.dim() == 0:
    #         print("Scalar parameter detected, using simple average")
    #         return 0.5 * a + 0.5 * b
    #
    #     # Ensure gradient tracking
    #     a = a.detach().clone().requires_grad_(True)
    #     b = b.detach().clone().requires_grad_(True)
    #
    #     if use_diverse == 1.0:
    #         synthetic_input = MergeMethods.generate_diverse_input(a, num_samples, a.device).to(dtype=a.dtype)
    #     else:
    #         # Dimension-aware input generation
    #         if a.dim() == 4:  # Conv2D
    #             print(f"Conv2D tensor detected: {a.shape}")
    #             # For conv layers: [batch, in_channels, kernel_h, kernel_w]
    #             synthetic_input = torch.randn(num_samples, a.size(1),
    #                                           max(8, a.size(2) * 2), max(8, a.size(3) * 2),
    #                                           device=a.device, dtype=a.dtype) * noise_scale
    #         elif a.dim() == 2:  # Linear
    #             print(f"Linear tensor detected: {a.shape}")
    #             # For linear layers: [batch, in_features] - IMPORTANT: Use size(1) not size(0)
    #             synthetic_input = torch.randn(num_samples, a.size(1),
    #                                           device=a.device, dtype=a.dtype) * noise_scale
    #         elif a.dim() == 1:  # Bias
    #             print(f"Bias tensor detected: {a.shape}")
    #             # For bias terms: [batch, out_features]
    #             synthetic_input = torch.randn(num_samples, a.size(0),
    #                                           device=a.device, dtype=a.dtype) * noise_scale
    #         else:
    #             raise ValueError(f"Unsupported dimension {a.dim()}")
    #
    #     print(f"Generated synthetic input: {synthetic_input.shape}")
    #
    #     # Safe feature extraction with dimension checking
    #     try:
    #         if a.dim() == 4:
    #             # For conv layers, use proper padding
    #             feats_a = F.conv2d(synthetic_input, a, padding=a.size(2) // 2)
    #             feats_b = F.conv2d(synthetic_input, b, padding=b.size(2) // 2)
    #         elif a.dim() == 2:
    #             # Check matrix dimensions
    #             if a.size(0) != synthetic_input.size(1):
    #                 print(f"Matrix dimension mismatch: {synthetic_input.shape} @ {a.shape}")
    #                 # For SDXL transformer weights, we need to handle [out_features, in_features] format
    #                 if a.size(1) == synthetic_input.size(1):
    #                     # This is correct - we want to multiply with the transpose
    #                     feats_a = synthetic_input @ a.t()
    #                     feats_b = synthetic_input @ b.t()
    #                 else:
    #                     # If dimensions still don't match, regenerate with correct size
    #                     synthetic_input = MergeMethods.generate_statistical_input(a, num_samples,
    #                                                                  a.device) if use_statistical else torch.randn(
    #                         num_samples, a.size(1), device=a.device, dtype=a.dtype) * noise_scale
    #                     print(f"Regenerated input: {synthetic_input.shape}")
    #                     feats_a = synthetic_input @ a.t()
    #                     feats_b = synthetic_input @ b.t()
    #             else:
    #                 feats_a = synthetic_input @ a
    #                 feats_b = synthetic_input @ b
    #         else:  # Bias terms
    #             # For bias, we add the bias to pre-activations
    #             feats_a = synthetic_input + a.unsqueeze(0)
    #             feats_b = synthetic_input + b.unsqueeze(0)
    #     except RuntimeError as e:
    #         print(f"Error in feature extraction: {e}")
    #         print(f"Falling back to simple average for this tensor")
    #         return 0.5 * a.detach() + 0.5 * b.detach()
    #
    #     print(f"Feature shapes: A={feats_a.shape}, B={feats_b.shape}")
    #
    #     def compute_fisher_improved(param, feats_a, feats_b, key=""):  # Pass feats_a and feats_b
    #         """Enhanced Fisher computation with layer-aware adaptation, dynamic scaling and normalization."""
    #         param.grad = None
    #
    #         with torch.enable_grad():
    #
    #             # Use feats_a and feats_b to compute pseudo_labels
    #             comparison_noise = torch.randn_like(feats_a) * 0.3  # noise must have the same shape as feats_a
    #             pseudo_labels = (feats_a > feats_b + comparison_noise).float()  # must have same shape as feats_a and b
    #             print(f"Generated {pseudo_labels.sum().item()}/{feats_a.numel()} positive pseudo-labels")
    #
    #             focal_loss = F.binary_cross_entropy_with_logits(
    #                 feats_a,  # [B, ...] Use original features, remove reduction. Same for feats_b
    #                 pseudo_labels,  # [B, ...] Use per-element labels
    #                 reduction='none'  # Keep per-sample loss, and per-element
    #             )
    #             loss = focal_loss.mean()  # Average over the batch
    #             print(f"Loss: {loss.item():.4f}")
    #
    #         loss.backward(retain_graph=True)
    #         grad = param.grad
    #
    #         avg_grad_magnitude = torch.mean(torch.abs(grad.detach())).item()
    #         if avg_grad_magnitude < 1e-10:  # Increased threshold slightly
    #             scale_factor = 1.0
    #         else:
    #             scale_factor = 1.0 / avg_grad_magnitude
    #
    #         # Layer-type specific adaptation (Keep as is)
    #         layer_scale = 1.0
    #         if param.dim() == 4:  # Convolutional layers
    #             if 'op' in key or 'skip_connection' in key:
    #                 layer_scale = 0.7  # Down-weight skip connections
    #             elif 'out.2' in key:  # Final output conv
    #                 layer_scale = 1.2
    #             else:
    #                 layer_scale = 1.2  # Other conv layers
    #
    #         elif param.dim() == 2:  # Linear layers
    #             if 'ff.net' in key:
    #                 layer_scale = 1.4 if 'proj' in key else 1.2
    #             elif 'attn1' or 'attn2' in key:
    #                 layer_scale = 1.6 if 'to_out' in key else 1.5
    #             elif 'emb_layers' in key:  # Added emb layers
    #                 layer_scale = 0.9
    #             elif 'time_embed' in key:  # time embed, linear
    #                 layer_scale = 1.0
    #             elif 'out.0' in key:
    #                 layer_scale = 1.0  # final output block linear
    #             else:
    #                 layer_scale = 1.0
    #
    #         elif param.dim() == 1:  # Bias terms
    #             if 'norm' in key:
    #                 layer_scale = 0.6
    #             elif 'time_embed' in key:
    #                 layer_scale = 0.8  # time_embed bias
    #             else:
    #                 layer_scale = 0.8  # Slightly down-weight biases
    #
    #         # --- Fisher Calculation and Normalization ---
    #         fisher = grad.pow(2) * scale_factor * layer_scale
    #
    #         # Numerical stability (Adjust clamping, potentially making it less aggressive)
    #         clamp_min = 0.0  # Fisher should be non-negative
    #         clamp_max = 5e2  # Allow higher values for differentiation, though maybe don't need it
    #         fisher = torch.clamp(fisher, min=clamp_min, max=clamp_max)  # Keep clamp
    #
    #         print(
    #             f"Layer: {param.shape} | Type: {layer_scale}x | Avg Grad Mag: {avg_grad_magnitude:.4e} | Scale Factor: {scale_factor:.4e} | Fisher Mean: {fisher.mean().item():.4f} ±{fisher.std().item():.4f}")
    #
    #         return fisher
    #
    #     def diagonal_rescaling(fisher_diag, param, scaling_factor=1.0):  # CHANGED scaling_factor
    #         param_scale = torch.var(param.detach(), dim=0, keepdim=True) + 1e-6
    #         rescaled_fisher = fisher_diag / (param_scale * scaling_factor + 1e-6)
    #         return rescaled_fisher  # REMOVED max normalization
    #
    #     print("Computing Fisher information...")
    #     fisher_a = compute_fisher_improved(a, feats_a, feats_b, key=key)
    #     fisher_b = compute_fisher_improved(b, feats_b, feats_a, key=key)  # FIXED
    #
    #     # After computing fisher_a and fisher_b
    #     fisher_a = diagonal_rescaling(fisher_a, a)
    #     fisher_b = diagonal_rescaling(fisher_b, b)
    #
    #     # Cleanup gradients
    #     # Clear gradients more thoroughly
    #     def clear_grads(*tensors):
    #         for t in tensors:
    #             if t.grad is not None:
    #                 t.grad.detach_()
    #                 t.grad = None
    #
    #     clear_grads(a, b, synthetic_input)
    #
    #     sign_mask = (a * b) > 0
    #     sign_agreement = sign_mask.float().mean().item() * 100
    #     print(f"Sign agreement: {sign_agreement:.1f}%")
    #
    #     # Adaptive merging with improved weighting
    #     raw_weights = fisher_b / (fisher_a + fisher_b + epsilon)
    #     clamped_weights = MergeMethods.adaptive_clamp(raw_weights, sign_agreement)
    #     weights = torch.where(sign_mask, clamped_weights, 0.5)
    #
    #     if raw_weights.numel() == 1:
    #         print(f"[{key}] Scalar weight: {raw_weights.item():.4f}")
    #     else:
    #         # Calculate statistics with sampling
    #         mean_val = raw_weights.mean().item()
    #         std_val = raw_weights.std().item() if raw_weights.numel() > 1 else 0.0
    #
    #         # Quantile calculation with sampling
    #         if raw_weights.numel() > 10000:
    #             flat = raw_weights.view(-1)
    #             sample = flat[torch.randperm(flat.size(0), device=flat.device)[:10000]]
    #             q25, q75 = torch.quantile(sample,
    #                                       torch.tensor([0.25, 0.75], device=sample.device, dtype=raw_weights.dtype))
    #         else:
    #             q25, q75 = torch.quantile(raw_weights, torch.tensor([0.25, 0.75], device=raw_weights.device,
    #                                                                 dtype=raw_weights.dtype))
    #
    #         # Formatting (keep existing)
    #         print(f"""
    #         Weight Analysis:
    #           - Distribution: μ={mean_val:.4f} ±{std_val:.4f}
    #           - Quartiles: 25%={q25:.4f} | 75%={q75:.4f}
    #           - Range: [{raw_weights.min().item():.4f}, {raw_weights.max().item():.4f}]
    #           - Fisher Ratio: {fisher_b.mean().item() / (fisher_a.mean().item() + 1e-8):.2f}
    #         """)
    #
    #     merged = torch.where(sign_mask, (1 - weights) * a + weights * b, 0.5 * (a + b))
    #     print("Weights:", weights)
    #
    #     return merged
    #
    # def adaptive_clamp(weights, sign_agreement):
    #     agreement = sign_agreement / 100
    #     # Allow mild extrapolation for high agreement cases
    #     extrapolation_scale = 0.2  # 20% beyond 0-1 at 100% agreement
    #
    #     lower_bound = 0.0 - extrapolation_scale * agreement ** 3
    #     upper_bound = 1.0 + extrapolation_scale * agreement ** 3
    #
    #     return torch.clamp(weights, lower_bound, upper_bound)
    #
    # def generate_diverse_input(param, num_samples=256, device='cuda'):
    #     """Generate highly diverse synthetic inputs with multiple modes"""
    #     if param.dim() == 4:  # Conv layers
    #         n_channels = param.size(1)
    #
    #         # Create multi-modal distribution (4 distinct clusters)
    #         clusters = []
    #         for i in range(4):
    #             # Each cluster has different mean and variance
    #             mean_shift = torch.randn(1, n_channels, 1, 1, device=device) * (i + 1) / 2
    #             var_scale = 0.5 + i * 0.5  # Different variance per cluster
    #             cluster = torch.randn(num_samples // 4, n_channels,
    #                                   max(8, param.size(2) * 2), max(8, param.size(3) * 2),
    #                                   device=device) * var_scale + mean_shift
    #             clusters.append(cluster)
    #
    #         return torch.cat(clusters, dim=0)
    #
    #     elif param.dim() <= 2:  # Linear/bias layers
    #         n_features = param.size(-1) if param.dim() == 2 else param.size(0)
    #
    #         # Create adversarial inputs that maximize differences
    #         base = torch.randn(num_samples // 4, n_features, device=device)
    #
    #         # Create 4 distinct input distributions
    #         diverse_inputs = [
    #             base,  # Standard normal
    #             base * 2.0,  # Scaled inputs
    #             torch.sign(base) * torch.abs(base).sqrt(),  # Non-linear transformation
    #             torch.sin(base * 3.14159)  # Periodic transformation
    #         ]
    #
    #         return torch.cat(diverse_inputs, dim=0)

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
    #             vs.append(MergeMethods.butterfly_merge.__wrapped__(k_a, k_b, **k_kwargs))
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
    #     Q_a = MergeMethods.butterfly_orthogonalize(a_2d, lora_dim, constraint, a.device)
    #
    #     # Project the difference using the butterfly orthogonal basis
    #     diff = b_2d - a_2d
    #     projected_diff = Q_a @ (Q_a.T @ diff)
    #
    #     # Apply butterfly orthogonalization to the projected difference
    #     Q_diff = MergeMethods.butterfly_orthogonalize(projected_diff, lora_dim, constraint, a.device)
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
    #         block_size, block_num = MergeMethods.butterfly_factor(out_dim, lora_dim)
    #         boft_m = sum(int(i) for i in f"{block_num - 1:b}") + 1
    #
    #         # If all parameters are calculated successfully, proceed.
    #         oft_blocks = MergeMethods.initialize_butterfly_blocks(x, boft_m, block_num, block_size, device)
    #
    #         # The return for the happy path is INSIDE the try block.
    #         return MergeMethods.apply_butterfly_transform(x, oft_blocks, boft_m, block_size, block_num, constraint,
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
            **kwargs
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
                vs.append(MergeMethods.butterfly_projection.__wrapped__(
                    k_a, k_b,
                    alpha=alpha,
                    rank_ratio=rank_ratio,
                    lora_dim=lora_dim,
                    constraint=constraint,
                    boft_iters=boft_iters,
                    boft_step_scale=boft_step_scale,
                    projector_eps=projector_eps,
                    seed=seed,
                    early_exit=early_exit,
                    **k_kwargs
                ))
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
        padded_dim = MergeMethods._next_pow2(out_dim)
        lora_dim_pow2 = MergeMethods._clamp_lora_dim_pow2(lora_dim, padded_dim)

        # Use the clamped dimension for consistency with butterfly_orthogonalize
        subspace_dim = min(lora_dim_pow2, min(a_2d.shape))

        # More stable seed generation
        seed_a = seed if seed is not None else MergeMethods.stable_seed_from_tensor(a_2d, key)

        # **UPDATED: Cache key now includes lora_dim_pow2 for consistency**
        if cache is not None:
            cache_key = f"{key}_lora{lora_dim_pow2}"  # Include clamped lora_dim in cache key
            if cache_key not in cache:
                cache[cache_key] = {}
            layer_cache = cache[cache_key]
        else:
            layer_cache = None

        if layer_cache is not None and 'Q_a_full' in layer_cache:
            Q_a_full = layer_cache['Q_a_full'].to(device=a.device, dtype=a.dtype)
        else:
            # Apply data-aligned butterfly orthogonalization to create subspace basis
            # **UPDATED: Pass lora_dim_pow2 for consistency**
            Q_a_full = MergeMethods.butterfly_orthogonalize(
                a_2d, key, lora_dim_pow2, constraint, a.device,
                seed=seed_a, guide=a_2d, iters=boft_iters, step=constraint * boft_step_scale)
            if layer_cache is not None:
                layer_cache['Q_a_full'] = Q_a_full.cpu()

        # Create projection matrix for subspace restriction
        P = Q_a_full[:, :subspace_dim]  # [out_dim, subspace_dim]
        diff = b_2d - a_2d

        # Modern QR-based projector with Cholesky fallback - CONVERT TO FP64
        k = P.shape[1]
        P_fp64 = P.to(torch.float64)
        diff_fp64 = diff.to(torch.float64)

        # Primary approach: QR-based projector (more numerically stable)
        try:
            Q, _ = torch.linalg.qr(P_fp64, mode='reduced')
            # Compute projection coefficients: E = Q^T diff
            E = (Q.T @ diff_fp64).to(P.dtype)

        except RuntimeError as qr_error:
            # Fallback to Cholesky if QR fails (very rare)
            print(f"QR failed, using Cholesky fallback: {qr_error}")
            try:
                I_k = torch.eye(k, device=P.device, dtype=torch.float64)
                G = (P_fp64.T @ P_fp64) + projector_eps * I_k
                L = torch.linalg.cholesky(G)

                # Compute E using modern triangular solves
                rhs = P_fp64.T @ diff_fp64
                y = torch.linalg.solve_triangular(L, rhs, upper=False)
                E = torch.linalg.solve_triangular(L.T, y, upper=True).to(P.dtype)

            except RuntimeError as chol_error:
                print(f"Both QR and Cholesky failed, using pseudo-inverse: {chol_error}")
                # Last resort: pseudo-inverse
                P_pinv = torch.linalg.pinv(P_fp64)
                E = (P_pinv @ diff_fp64).to(P.dtype)

        # Cap rank by subspace dimension
        r = min(
            max(1, int(min(E.shape) * rank_ratio)),
            subspace_dim
        )

        diff_seed = MergeMethods.stable_seed_from_tensor(E, f"{key}_diff")

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

    @staticmethod
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
            str(tensor.device)
        ]

        # Combine with key string
        hash_input = f"{key}_{tensor_stats}"
        hash_obj = hashlib.md5(hash_input.encode())
        return int(hash_obj.hexdigest()[:8], 16) % 2147483647

    @staticmethod
    def butterfly_orthogonalize(x, key, lora_dim=4, constraint=0.01, device=None,
                                seed=None, guide=None, iters=1, step=None):
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
            padded_dim = MergeMethods._next_pow2(out_dim)
            lora_dim_pow2 = MergeMethods._clamp_lora_dim_pow2(lora_dim, padded_dim)
            block_size, block_num = MergeMethods.butterfly_factor(padded_dim, lora_dim_pow2)
            boft_m = int(math.log2(block_num))  # Now guaranteed to be valid

            # Initialize butterfly blocks with input-informed scaling
            oft_blocks = MergeMethods.initialize_butterfly_blocks(x, boft_m, block_num, block_size, device, gen)

            # Apply data alignment sweep - returns orthogonal blocks directly
            if iters > 0 and step > 0:
                r_blocks = MergeMethods.align_butterfly_blocks(
                    oft_blocks, guide2d, boft_m, block_num, block_size, iters, step, device, key)
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
            result = MergeMethods.apply_butterfly_transform(x, r_blocks, boft_m, block_size, block_num, device, key)

            # **FIX: Ensure final result matches input dtype**
            return result.to(dtype=original_dtype)

        except Exception as e:
            print(
                f"Butterfly factorization failed for dimension {out_dim}, falling back to QR decomposition. Error: {e}")

            # Proper fallback that creates basis from guide
            Q, _ = torch.linalg.qr(guide2d, mode='reduced')

            if constraint > 0:
                # Apply constraint as magnitude scaling
                Q_norm = torch.norm(Q, dim=0, keepdim=True)
                constraint_value = constraint * torch.sqrt(
                    torch.tensor(guide2d.shape[0], dtype=original_dtype, device=device))  # **ADD DTYPE**
                scale = torch.clamp(constraint_value / (Q_norm + 1e-8), max=1.0)
                Q = Q * scale

            # **FIX: Ensure fallback result matches input dtype**
            return Q.to(dtype=original_dtype)

    @staticmethod
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

        blocks = torch.zeros(boft_m, block_num, block_size, block_size,
                             device=device, dtype=dtype)  # **ADD DTYPE**

        with torch.no_grad():
            # Scale initialization based on input tensor characteristics
            x_mean = torch.mean(torch.abs(x))

            # Adaptive scaling based on input statistics
            base_scale = min(0.01, x_mean.item() * 0.01)

            for i in range(boft_m):
                for j in range(block_num):
                    # Create anti-symmetric matrices with input-informed scaling
                    random_vals = torch.randn(block_size, block_size,
                                              device=device, dtype=dtype,
                                              generator=generator) * base_scale  # **ADD DTYPE**

                    # Decay scale with butterfly layer depth for stability
                    layer_scale = 1.0 / (i + 1)

                    # Make anti-symmetric (Q = -Q^T) and scale
                    blocks[i, j] = (random_vals - random_vals.T) * layer_scale / 2.0

        return blocks

    @staticmethod
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
                            target_Q, _ = torch.linalg.qr(guide_block[:, :actual_block_size], mode='reduced')

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
                                        I_loc = torch.eye(actual_block_size, device=current_Q.device,
                                                          dtype=current_Q.dtype)

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
                                        current_Q.to(torch.float64),
                                        target_Q.to(torch.float64),
                                        stage_step,
                                        eps=1e-8,
                                        max_iters=50
                                    ).to(current_Q.dtype)

                                # **FIX: Ensure aligned_Q has correct dtype before assignment**
                                r[i, j, :actual_block_size, :actual_block_size] = aligned_Q.to(dtype=original_dtype)

                            except (RuntimeError, AssertionError):
                                # Fallback: keep current block
                                r[i, j, :actual_block_size, :actual_block_size] = current_Q

            # **FIX: Ensure final result has correct dtype**
            return r.to(dtype=original_dtype)

    @staticmethod
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

    @staticmethod
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

    # @staticmethod
    # @merge_method
    # def add_difference_var_clip(
    #         a: Parameter(Tensor, "weight"),
    #         b: Parameter(Tensor, "weight"),
    #         c: Parameter(Tensor, "weight"),
    #         *,
    #         alpha: Parameter(Tensor) =1.0,
    #         **kwargs,
    # ) -> Return(Tensor):
    #     bc_corr = torch.corrcoef(torch.stack([
    #         (b - b.mean()).flatten(),
    #         (c - c.mean()).flatten()
    #     ], dim=0))[0, 1]
    #
    #     b_var = b.var(correction=0)
    #     c_var = c.var(correction=0)
    #
    #     bc_cov = bc_corr * torch.sqrt(b_var * c_var)
    #
    #     min_corr = 0.9999
    #     if bc_corr < min_corr:
    #         bc_scale = torch.sqrt(b_var + c_var - 2 * min_corr * torch.sqrt(b_var * c_var)) / torch.sqrt(
    #             b_var + c_var - 2 * bc_cov)
    #     else:
    #         bc_scale = 1.0
    #
    #     bc = b - c
    #     bc = (bc - bc.mean()) * bc_scale + bc.mean()
    #     res = a + alpha * bc
    #     return (res - res.mean()) * a.std(correction=0) / res.std(correction=0) + a.mean()
    #
    # @staticmethod
    # @merge_method
    # def gram_schmidt_ortho(
    #         a: Parameter(Tensor, "weight"),
    #         b: Parameter(Tensor, "weight"),
    #         c: Parameter(Tensor, "weight"),
    #         *,
    #         alpha: Parameter(Tensor) =1.0,
    #         **kwargs,
    # ) -> Return(Tensor):
    #     # Calculate the vectors
    #     vector_a = a - c
    #     vector_b = b - c
    #
    #     # Calculate the projection of B onto A
    #     projection_b_on_a = (torch.dot(vector_b.flatten(), vector_a.flatten()) / torch.dot(vector_a.flatten(),
    #                                                                                        vector_a.flatten())) * vector_a
    #
    #     # Magnitude adjustment based on the difference between A and C
    #     magnitude_ratio = torch.norm(projection_b_on_a) / torch.norm(vector_a)
    #     adjusted_projection = projection_b_on_a * (1 + alpha * (magnitude_ratio - 1))
    #
    #     # Add the adjusted projection to the base model
    #     return a + adjusted_projection
    #
    # @staticmethod
    # @merge_method
    # def orth_pro(
    #         a: Parameter(Tensor, "weight"),
    #         b: Parameter(Tensor, "weight"),
    #         c: Parameter(Tensor, "weight"),
    #         *,
    #         alpha: Parameter(Tensor) =1.0,
    #         use_perp: Parameter(Tensor) =0,
    #         ab_only: Parameter(Tensor) =0,
    #         noisy_c: Parameter(Tensor) =0,
    #         noisy_c_sgn_flt: Parameter(Tensor) =0,
    #         **kwargs,
    # ) -> Return(Tensor):
    #     """
    #     Merges tensors 'a' and 'b' using Orthogonal Procrustes alignment with options for perpendicular
    #     component projection, noise injection, and control over alignment scope.
    #
    #     Args:
    #         a (Tensor): The first tensor.
    #         b (Tensor): The second tensor.
    #         c (Tensor): The anchor tensor.
    #         alpha (float): The interpolation factor between the original tensor 'b' and the mapped
    #                        tensor (0 <= alpha <= 1).
    #         use_perp (bool): If True, projects 'a' onto the perpendicular component of 'b' before alignment.
    #         ab_only (bool): If True, performs alignment only between 'a' and 'b', ignoring 'c'.
    #         noisy_c (float): The standard deviation of Gaussian noise added to 'c' (0 for no noise).
    #         noisy_c_sgn_flt (bool): If True, filters the noise added to 'c' to match the sign of 'c'.
    #         **kwargs: Additional keyword arguments.
    #
    #     Returns:
    #         Tensor: The merged tensor.
    #     """
    #     # Reshape tensors to 2D
    #     is_conv_3x3 = len(a.shape) == 4 and a.shape[-1] != 1
    #     is_conv_1x1 = len(a.shape) == 4 and a.shape[-1] == 1
    #     original_shape = a.shape
    #     if is_conv_3x3:
    #         shape_2d = (-1, functools.reduce(operator.mul, a.shape[1:]))
    #     elif is_conv_1x1:
    #         shape_2d = (-1, functools.reduce(operator.mul, a.shape[1:]))
    #     elif not a.shape:
    #         shape_2d = (1, 1)
    #     else:
    #         shape_2d = (-1, a.shape[-1])
    #
    #     a = a.reshape(shape_2d)
    #     b = b.reshape(shape_2d)
    #     c = c.reshape(shape_2d) if not noisy_c else MergeMethods.create_noisy_tensor(c.reshape(shape_2d),
    #                                                                                  sign_filter=noisy_c_sgn_flt,
    #                                                                                  seed=0)
    #     ac = a if ab_only else (a - c)
    #     bc = b if ab_only else (b - c)
    #
    #     if use_perp:
    #         norm_bc = torch.linalg.norm(bc) + 1e-20
    #         ac = ac - bc * (bc / norm_bc * (ac / norm_bc)).sum()
    #
    #     res = MergeMethods.orthogonal_procrustes(ac, bc)
    #     if ab_only:
    #         return torch.lerp(b.reshape(original_shape), res.reshape(original_shape), alpha)
    #     else:
    #         return torch.lerp(b.reshape(original_shape), (c + res).reshape(original_shape), alpha)
    #
    # def orthogonal_procrustes(a: Tensor, b: Tensor):
    #     # Compute the QR decomposition of (a - c)
    #     Q, R = torch.qr(a)
    #
    #     # Compute the mapping matrix
    #     mapping_matrix = torch.mm(Q.t(), b)
    #
    #     # Map (a - c) to (b - c)
    #     mapped_tensor = torch.mm(Q, mapping_matrix)
    #
    #     return mapped_tensor
    #
    # def create_noisy_tensor(
    #         a: Tensor,
    #         seed=218,
    #         sign_filter=False,
    # ) -> Tensor:
    #     torch.manual_seed(seed)
    #
    #     dist = torch.normal(a.mean(), a.std(correction=0, keepdim=True))
    #
    #     if sign_filter:
    #         signs = torch.sign(dist)
    #
    #         final_sign = torch.sign(a)
    #
    #         delta_filters = (signs == final_sign).float()
    #
    #         param_counts = torch.sum(delta_filters, dim=0)
    #
    #         filtered_delta = (dist * delta_filters)
    #
    #         filtered_delta = filtered_delta.sum(dim=0)
    #
    #         dist = torch.nan_to_num(filtered_delta / param_counts)
    #
    #     return dist
    #
    # @staticmethod
    # @merge_method
    # def lu_merge(
    #         a: Parameter(Tensor, "weight"),
    #         b: Parameter(Tensor, "weight"),
    #         c: Parameter(Tensor, "weight"),
    #         *,
    #         alpha: Parameter(Tensor) =1.0,
    #         theta: Parameter(Tensor) =1.0,
    #         use_perp: Parameter(Tensor) =0,
    #         ab_only: Parameter(Tensor) =0,
    #         **kwargs,
    # ) -> Return(Tensor):
    #     """
    #     Merges tensors 'a' and 'b' using LU decomposition interpolation with optional alignment adjustments.
    #
    #     Args:
    #         a (Tensor): The first tensor.
    #         b (Tensor): The second tensor.
    #         c (Tensor): The anchor tensor.
    #         alpha (float): The interpolation factor between the original tensor 'a' and the merged
    #                        tensor (0 <= alpha <= 1).
    #         theta (float): The interpolation factor for the LU decomposition components of 'a' and 'b'
    #                        (0 <= theta <= 1).
    #         use_perp (bool): If True, projects 'a' onto the perpendicular component of 'b' before merging.
    #         ab_only (bool): If True, performs merging only between 'a' and 'b', ignoring 'c'.
    #         **kwargs: Additional keyword arguments.
    #
    #     Returns:
    #         Tensor: The merged tensor.
    #     """
    #     # Reshape tensors to 2D
    #     is_conv_3x3 = len(a.shape) == 4 and a.shape[-1] != 1
    #     is_conv_1x1 = len(a.shape) == 4 and a.shape[-1] == 1
    #     original_shape = a.shape
    #     if is_conv_3x3:
    #         shape_2d = (-1, functools.reduce(operator.mul, a.shape[1:]))
    #     elif is_conv_1x1:
    #         shape_2d = (-1, functools.reduce(operator.mul, a.shape[1:]))
    #     elif not a.shape:
    #         shape_2d = (1, 1)
    #     else:
    #         shape_2d = (-1, a.shape[-1])
    #
    #     a = a.reshape(shape_2d)
    #     b = b.reshape(shape_2d)
    #     c = c.reshape(shape_2d)
    #
    #     # Calculate difference tensors if not ab_only
    #     ac = a if ab_only else (a - c)
    #     bc = b if ab_only else (b - c)
    #
    #     # Project 'ac' onto the perpendicular component of 'bc' if use_perp is True
    #     if use_perp:
    #         norm_bc = torch.linalg.norm(bc) + 1e-20
    #         ac = ac - bc * (bc / norm_bc * (ac / norm_bc)).sum()
    #
    #     # Perform LU decomposition-based merging
    #     res = MergeMethods.lu_decompose(ac, bc, theta)
    #
    #     # Interpolate between original tensor 'A' and the merged result based on 'alpha'
    #     if ab_only:
    #         return torch.lerp(a.reshape(original_shape), res.reshape(original_shape), alpha)
    #     else:
    #         return torch.lerp(a.reshape(original_shape), (c + res).reshape(original_shape), alpha)
    #
    # def lu_decompose(a, b, t=1.0):
    #     """
    #     Performs LU decomposition-based interpolation between tensors a and b.
    #
    #     Args:
    #         a (Tensor): The first tensor (2D).
    #         b (Tensor): The second tensor (2D).
    #         t (float): Interpolation factor (0 <= t <= 1).
    #
    #     Returns:
    #         Tensor: Interpolated tensor based on LU decomposition.
    #     """
    #     # Compute LU decomposition for tensors a and b
    #     P_A, L_A, U_A = torch.linalg.lu(a)
    #     P_B, L_B, U_B = torch.linalg.lu(b)
    #
    #     # Interpolate L and U matrices
    #     L_interpolated = (1 - t) * L_A + t * L_B
    #     U_interpolated = (1 - t) * U_A + t * U_B
    #
    #     # Combine interpolated matrices
    #     A_interpolated = P_A @ L_interpolated @ U_interpolated
    #
    #     return A_interpolated
    #
    # @staticmethod
    # @merge_method
    # def clyb_merge(
    #         a: Parameter(Tensor, "weight"),
    #         b: Parameter(Tensor, "weight"),
    #         c: Parameter(Tensor, "weight"),
    #         *,
    #         alpha: Parameter(Tensor) =1.0,
    #         use_perp: Parameter(Tensor) =1.0,
    #         ab_only: Parameter(Tensor) =0,
    #
    #         **kwargs,
    # ) -> Return(Tensor):
    #     """
    #     Merges tensors 'a' and 'b' using a combination of low-rank approximation, orthogonal projection,
    #     and optional perpendicular component projection and alignment adjustments.
    #
    #     Args:
    #         a (Tensor): The source tensor.
    #         b (Tensor): The target tensor.
    #         c (Tensor): The reference tensor.
    #         alpha (float): The interpolation factor between the original tensor 'b' and the merged
    #                        tensor (0 <= alpha <= 1).
    #         use_perp (bool): If True, projects 'a' onto the perpendicular component of 'b' before merging.
    #         ab_only (bool): If True, performs merging only between 'a' and 'b', ignoring 'c'.
    #         cache (dict): Cache svd and qr for performance
    #         **kwargs: Additional keyword arguments.
    #
    #     Returns:
    #         Tensor: The merged tensor (in fp16).
    #     """
    #     # Reshape tensors to 2D
    #     is_conv_3x3 = len(a.shape) == 4 and a.shape[-1] != 1
    #     is_conv_1x1 = len(a.shape) == 4 and a.shape[-1] == 1
    #     original_shape = a.shape
    #     if is_conv_3x3:
    #         shape_2d = (-1, functools.reduce(operator.mul, a.shape[1:]))
    #     elif is_conv_1x1:
    #         shape_2d = (-1, functools.reduce(operator.mul, a.shape[1:]))
    #     elif not a.shape:
    #         shape_2d = (1, 1)
    #     else:
    #         shape_2d = (-1, a.shape[-1])
    #
    #     a = a.reshape(shape_2d)
    #     b = b.reshape(shape_2d)
    #     c = c.reshape(shape_2d)
    #
    #     # Calculate difference tensors if not ab_only
    #     ac = a if ab_only else (a - c)
    #     bc = b if ab_only else (b - c)
    #
    #     # Project 'ac' onto the perpendicular component of 'bc' if use_perp is True
    #     if use_perp:
    #         norm_bc = torch.linalg.norm(bc) + 1e-20
    #         ac = ac - bc * (bc / norm_bc * (ac / norm_bc)).sum()
    #
    #     # Perform the core merging operation
    #     res = MergeMethods.clyb_align(ac, bc, cache=cache, **kwargs)
    #
    #     # Interpolate between original tensor 'b' and the merged result based on 'alpha'
    #     if ab_only:
    #         return torch.lerp(b.reshape(original_shape), res.reshape(original_shape), alpha)
    #     else:
    #         return torch.lerp(b.reshape(original_shape), (c + res).reshape(original_shape), alpha)
    #
    # def clyb_align(a, b,  **kwargs):
    #     """
    #     Performs the core merging operation using QR decomposition, low-rank approximation, and orthogonal projection.
    #
    #     Args:
    #         a (Tensor): The source tensor (2D).
    #         b (Tensor): The target tensor (2D).
    #         cache (dict): A dictionary for caching intermediate results.
    #
    #     Returns:
    #         Tensor: The merged tensor (2D).
    #     """
    #     if cache is not None:
    #         key = kwargs["key"]
    #         if key not in cache:
    #             cache[key] = {}
    #         cache = cache[key]
    #
    #         if "Qb" in cache:
    #             Qb = cache["Qb"].to(b.device, b.dtype)  # Reuse cached Qb
    #         else:
    #             Qb, _ = torch.qr(b)  # Calculate and cache Qb
    #             cache["Qb"] = Qb.to("cpu")
    #
    #         if "Ua" in cache and "Sa" in cache and "Va" in cache:
    #             Ua = cache["Ua"].to(a.device, a.dtype)  # Reuse cached Ua
    #             Sa = cache["Sa"].to(a.device, a.dtype)  # Reuse cached Sa
    #             Va = cache["Va"].to(a.device, a.dtype)  # Reuse cached Va
    #         else:
    #             compression = 16
    #             q_size = max(int(torch.linalg.matrix_rank(Qb)) // compression, 1)
    #             iters = min(max(int(math.exp(math.log(640 / q_size))), 2), 64)
    #             Ua, Sa, Va = torch.svd_lowrank(a, q=q_size, niter=iters)  # Calculate and cache SVD components
    #             cache["Ua"] = Ua.to("cpu")
    #             cache["Sa"] = Sa.to("cpu")
    #             cache["Va"] = Va.to("cpu")
    #     else:  # No caching, calculate everything
    #         compression = 16
    #         Qb, _ = torch.linalg.qr(b)
    #         q_size = max(int(torch.linalg.matrix_rank(Qb)) // compression, 1)
    #         iters = min(max(int(math.exp(math.log(640 / q_size))), 2), 64)
    #         Ua, Sa, Va = torch.svd_lowrank(a, q=q_size, niter=iters)
    #
    #     a_lowrank = torch.mm(Ua, torch.mm(torch.diag(Sa), Va.t()))
    #     a_diff = a - a_lowrank
    #
    #     # Project the difference onto the space spanned by Qb (orthogonal basis of B)
    #     a_diff_projected = torch.mm(Qb, torch.mm(Qb.t(), a_diff))
    #
    #     return b + a_diff_projected
    #
    # @staticmethod
    # @merge_method
    # def decompose_merge(
    #         a: Parameter(Tensor, "weight"),
    #         b: Parameter(Tensor, "weight"),
    #         *,
    #         alpha: Parameter(Tensor) =1.0,
    #         **kwargs,
    # ) -> Return(Tensor):
    #     """
    #     Merges tensors 'a' and 'b' by decomposing 'b' using SVD, aligning the difference
    #     between 'b' and its low-rank approximation to 'a', and adding the scaled result to 'a'.
    #
    #     Args:
    #         a (Tensor): The first tensor, serving as the base model.
    #         b (Tensor): The second tensor, whose components will be blended into 'a'.
    #         alpha (float): The scaling factor for the projected difference, controlling the strength of 'b's
    #                        influence on the merged result (0 <= alpha <= 1).
    #         **kwargs: Additional keyword arguments.
    #
    #     Returns:
    #         Tensor: The merged tensor.
    #     """
    #     # Reshape tensors to 2D
    #     is_conv_3x3 = len(a.shape) == 4 and a.shape[-1] != 1
    #     is_conv_1x1 = len(a.shape) == 4 and a.shape[-1] == 1
    #     original_shape = a.shape
    #     if is_conv_3x3:
    #         shape_2d = (-1, functools.reduce(operator.mul, a.shape[1:]))
    #     elif is_conv_1x1:
    #         shape_2d = (-1, functools.reduce(operator.mul, a.shape[1:]))
    #     elif not a.shape:
    #         shape_2d = (1, 1)
    #     else:
    #         shape_2d = (-1, a.shape[-1])
    #
    #     a = a.reshape(shape_2d)
    #     b = b.reshape(shape_2d)
    #
    #     # Perform the core merging operation
    #     res = MergeMethods.decompose_align(a, b, alpha)
    #
    #     return res.reshape(original_shape)
    #
    # def decompose_align(a, b, alpha):
    #     """
    #     Performs the core merging operation using SVD and orthogonal projection.
    #
    #     Args:
    #         a (Tensor): The base tensor (2D).
    #         b (Tensor): The tensor whose components will be blended into 'a' (2D).
    #         alpha (float): The scaling factor for the projected difference.
    #
    #     Returns:
    #         Tensor: The merged tensor (2D).
    #     """
    #     Ua, Sa, Va = torch.linalg.svd(a, full_matrices=False, driver="gesvd")
    #     Ub, Sb, Vb = torch.linalg.svd(b, full_matrices=False, driver="gesvd")
    #
    #     # Reconstruct a low-rank approximation of B using the singular values from A
    #     b_lowrank = torch.mm(Ub, torch.mm(torch.diag(Sa), Vb))
    #     b_diff = b - b_lowrank
    #
    #     # Project the difference (B - B_lowrank) onto the space spanned by Ua (orthogonal basis of A)
    #     b_diff_projected = torch.mm(Ua, torch.mm(Ua.t(), b_diff))
    #
    #     # Add the scaled projected difference to A to create the merged tensor
    #     return a + b_diff_projected * alpha
    #
    # @staticmethod
    # @merge_method
    # def svd_replace_merge(
    #         a: Parameter(Tensor, "weight"),
    #         b: Parameter(Tensor, "weight"),
    #         *,
    #         alpha: Parameter(Tensor) =1.0,
    #
    #         **kwargs,
    # ) -> Return(Tensor):
    #     """
    #     Merges tensors 'a' and 'b' using Singular Value Decomposition (SVD) by replacing the singular values
    #     of 'b' with those from 'a' and reconstructing 'b' using the modified singular values.
    #
    #     Args:
    #         a (Tensor): The first tensor, whose singular values will be used to modify 'b'.
    #         b (Tensor): The second tensor, whose structure will be retained but modified with 'a's singular values .
    #         alpha (float): The interpolation factor between the original tensor 'b' and the merged tensor (0 <= alpha <= 1).
    #         cache: Cache svd results to reuse and skip computation in subsequent iterations.
    #         **kwargs: Additional keyword arguments.
    #
    #     Returns:
    #         Tensor: The merged tensor.
    #     """
    #     # Reshape tensors to 2D
    #     is_conv_3x3 = len(a.shape) == 4 and a.shape[-1] != 1
    #     is_conv_1x1 = len(a.shape) == 4 and a.shape[-1] == 1
    #     original_shape = a.shape
    #     if is_conv_3x3:
    #         shape_2d = (-1, functools.reduce(operator.mul, a.shape[1:]))
    #     elif is_conv_1x1:
    #         shape_2d = (-1, functools.reduce(operator.mul, a.shape[1:]))
    #     elif not a.shape:
    #         shape_2d = (1, 1)
    #     else:
    #         shape_2d = (-1, a.shape[-1])
    #
    #     a = a.reshape(shape_2d)
    #     b = b.reshape(shape_2d)
    #
    #     res = MergeMethods.SVD_Replace(a, b, alpha, cache=cache, **kwargs)
    #
    #     return res.reshape(original_shape)
    #
    # def SVD_Replace(a, b, alpha,  **kwargs):
    #     """
    #     Performs the core merging operation using SVD, with caching for optimization.
    #
    #     Args:
    #         a (Tensor): The source tensor (2D), providing the singular values.
    #         b (Tensor): The target tensor (2D), whose structure is retained.
    #         alpha (float): The interpolation factor.
    #         cache:  A dictionary for caching SVD results.
    #
    #     Returns:
    #         Tensor: The merged tensor (2D).
    #     """
    #
    #     # Check if merged_tensor is already cached BEFORE performing SVD
    #     if cache is not None:
    #         key = kwargs["key"]
    #         if key not in cache:
    #             cache[key] = {}
    #         cache = cache[key]
    #
    #     if cache is not None and "merged_tensor" in cache:
    #         merged_tensor = cache["merged_tensor"].to(a.device, a.dtype)
    #     else:
    #         # Determine the SVD driver based on CUDA availability
    #         svd_driver = "gesvdj" if a.is_cuda else "gesvd"
    #         Ua, Sa, Va = torch.linalg.svd(a, full_matrices=False, driver=svd_driver)
    #         Ub, Sb, Vb = torch.linalg.svd(b, full_matrices=False, driver=svd_driver)
    #
    #         # Reconstruct 'b' using the singular values from 'a' (Vb is already transposed)
    #         merged_tensor = torch.mm(Ub, torch.mm(torch.diag(Sa), Vb))
    #         if cache is not None:
    #             cache["merged_tensor"] = merged_tensor.to("cpu")
    #
    #     # Interpolate between the original tensor 'b' and the merged tensor
    #     return torch.lerp(b, merged_tensor, alpha)
    #
    # @staticmethod
    # @merge_method
    # def weighted_sum_projection_v2(
    #         a: Parameter(Tensor, "weight"),
    #         b: Parameter(Tensor, "weight"),
    #         c: Parameter(Tensor, "weight"),
    #         *,
    #         perplexity: Parameter(Tensor) =0.0,
    #         **kwargs,
    # ) -> Return(Tensor):
    #
    #     key = kwargs["key"]
    #     if key.endswith(("in_proj_weight", "in_proj_bias")):
    #         vs = []
    #         for i, k in enumerate(("to_q", "to_k", "to_v")):
    #             k_kwargs = kwargs.copy()
    #             k_kwargs["key"] = key.replace("in_proj_", f"{k}.")
    #             dim = a.shape[0] // 3
    #             t_start = dim * i
    #             t_end = dim * (i + 1)
    #             k_a = a[t_start:t_end]
    #             k_b = b[t_start:t_end]
    #             k_c = c[t_start:t_end]
    #             vs.append(MergeMethods.weighted_sum_projection_v2.__wrapped__(k_a, k_b, k_c, **k_kwargs))
    #         return torch.cat(vs)
    #
    #     # Reshape tensors to 2D
    #     is_conv_3x3 = len(a.shape) == 4 and a.shape[-1] != 1
    #     is_conv_1x1 = len(a.shape) == 4 and a.shape[-1] == 1
    #     original_shape = a.shape
    #     if is_conv_3x3:
    #         shape_2d = (-1, functools.reduce(operator.mul, a.shape[1:]))
    #     elif is_conv_1x1:
    #         shape_2d = (-1, functools.reduce(operator.mul, a.shape[1:]))
    #     elif not a.shape:
    #         shape_2d = (1, 1)
    #     else:
    #         shape_2d = (-1, a.shape[-1])
    #
    #     a = a.reshape(shape_2d)
    #     b = b.reshape(shape_2d)
    #     c = c.reshape(shape_2d)
    #
    #     ba = b - a
    #     ca = c - a
    #
    #     # Calculate alpha values at different levels of granularity
    #     key_alpha = torch.nan_to_num((ba * ca).sum() / (ba ** 2).sum(), nan=0, posinf=0, neginf=0)
    #     neuron_alpha = torch.nan_to_num((ba * ca).sum(dim=1, keepdim=True) / (ba ** 2).sum(dim=1, keepdim=True), nan=0,
    #                                     posinf=0, neginf=0)
    #     param_alpha = torch.nan_to_num((ba * ca) / (ba ** 2), nan=0, posinf=0, neginf=0)
    #
    #     # Interpolate between alpha values based on perplexity
    #     alpha = torch.lerp(torch.lerp(key_alpha, neuron_alpha, 2 * perplexity),
    #                        torch.lerp(neuron_alpha, param_alpha, 2 * perplexity - 1), perplexity)
    #
    #     # Perform weighted sum using the interpolated alpha
    #     return ((1 - alpha) * a + alpha * b).reshape(original_shape)
    #
    # @staticmethod
    # @merge_method
    # def neuron_train_difference(
    #     a: Parameter(Tensor, "weight"),
    #     b: Parameter(Tensor, "weight"),
    #     c: Parameter(Tensor, "weight"),
    #     *,
    #     alpha: Parameter(Tensor) =1.0,
    #     **kwargs,
    # ) -> Return(Tensor):
    #     key = kwargs["key"]
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
    #             k_c = c[t_start:t_end]
    #             vs.append(MergeMethods.neuron_train_difference.__wrapped__(k_a, k_b, k_c, **k_kwargs))
    #         return torch.cat(vs)
    #
    #     if key.endswith("bias"):
    #         return sd_mecha.merge_methods.weighted_sum.__wrapped__(a, b, alpha=alpha)
    #
    #     # Reshape tensors to 2D
    #     is_conv_3x3 = len(a.shape) == 4 and a.shape[-1] != 1
    #     is_conv_1x1 = len(a.shape) == 4 and a.shape[-1] == 1
    #     original_shape = a.shape
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
    #     c = c.reshape(*shape_2d)
    #
    #     threshold = torch.maximum((a - c).norm(dim=1, keepdim=True), (b - c).norm(dim=1, keepdim=True))
    #     dissimilarity = (1 - torch.nan_to_num(((a - c) * (b - c)).sum(dim=1, keepdim=True) / threshold**2, nan=0)) / 2
    #
    #     res = a + (b - c) * alpha * dissimilarity
    #     return res.reshape(original_shape)
    #
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
    #     layer_type = MergeMethods.get_layer_type(a.shape, kwargs)
    #
    #     if layer_type == MergeMethods.LayerType.SCALAR:
    #         return MergeMethods.geometric_sum_full.__wrapped__(a, b, alpha=alpha)
    #     elif layer_type == MergeMethods.LayerType.OFFSET:
    #         return sd_mecha.weighted_sum.__wrapped__(a, b, alpha=alpha)
    #     elif layer_type == MergeMethods.LayerType.EMBEDD:
    #         return MergeMethods.clip_embedding_merge_v3(a, b, alpha=alpha)
    #     elif layer_type == MergeMethods.LayerType.CROSS_ATTENTION_QKV:
    #         return MergeMethods.merge_cross_attention_qkv(a, b, alpha=alpha, key=key, cache=layer_cache)
    #     elif layer_type == MergeMethods.LayerType.ATTENTION_QKV:
    #         return MergeMethods.merge_self_attention_qkv(a, b, alpha, key=key, cache=layer_cache)
    #     elif layer_type == MergeMethods.LayerType.ATTENTION_PROJ:
    #         return MergeMethods.merge_attention_output(a, b, alpha, key=key, cache=layer_cache)
    #     elif layer_type == MergeMethods.LayerType.FFN_PROJ:
    #         return MergeMethods.merge_ffn_proj(a, b, alpha=alpha, key=key)
    #     elif layer_type == MergeMethods.LayerType.FFN_OUT:
    #         return MergeMethods.merge_ffn_out(a, b, alpha=alpha, corr_threshold=corr_threshold, cache=layer_cache)
    #     elif layer_type == MergeMethods.LayerType.MATMUL:
    #         return MergeMethods.polar_decomposition(a, b, alpha=alpha, cache=layer_cache)
    #     elif layer_type == MergeMethods.LayerType.CONV2D:
    #         return MergeMethods.merge_wavelets(a, b, alpha=alpha)
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
    #         transform = MergeMethods.orthogonal_procrustes_ml(u_a_polar, u_b_polar)
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
    #     merged_u = MergeMethods.slerp_unitary_taylor(u_a_polar, u_b_polar_aligned, alpha)
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
    #     rotation = MergeMethods.orthogonal_procrustes_ml(a_norm, b_norm)  # Replace SVD-based rotation
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
    #             merged = MergeMethods.polar_decomposition(part_a, part_b, alpha, cache=part_cache)
    #             if cache is not None:
    #                 cache[part_key] = part_cache
    #
    #             merged_parts.append(merged)
    #
    #         return torch.cat(merged_parts, dim=0)
    #
    #     # Handle regular CLIP text encoder layers
    #     elif any(x in key for x in ["k_proj", "v_proj", "q_proj"]):
    #         return MergeMethods.merge_self_attention_qkv(a, b, alpha, key)
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
    #             return MergeMethods.polar_decomposition(a, b, alpha=adjusted_alpha, cache=cache)
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
    #             R = MergeMethods.orthogonal_procrustes_ml(vh_a[:k], vh_b[:k])
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
    #             merged = MergeMethods.polar_decomposition(part_a, part_b, alpha, cache=part_caches[i])
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
    #         return MergeMethods.polar_decomposition(a, b, alpha=adjusted_alpha, cache=cache)
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
    #     merged = MergeMethods.polar_decomposition(a, b, alpha=adjusted_alpha, cache=cache)
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
    #     if MergeMethods.matrix_is_large(a, threshold=2048):  # Adjust threshold as needed
    #         return MergeMethods.merge_ffn_proj_conservative(a, b, alpha, expansion_factor)
    #     else:
    #         return MergeMethods.merge_ffn_proj_standard(a, b, alpha, expansion_factor)
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
    #                 R = MergeMethods.orthogonal_procrustes_ml(v_a[:, :k], v_b[:, :k])
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
    #         return MergeMethods.LayerType.OFFSET
    #
    #     # Layer Norms
    #     elif any(x in key for x in [".norm", "layer_norm", "ln_final", "ln_1", "ln_2", "layer_norm1", "layer_norm2",
    #                                 "final_layer_norm"]) or "norm" in key:
    #         return MergeMethods.LayerType.SCALAR
    #
    #     # Scalar Layer (like `logit_scale` in CLIP models)
    #     elif "logit_scale" in key:
    #         return MergeMethods.LayerType.SCALAR
    #
    #     # True embeddings (vocabulary mappings)
    #     elif "token_embedding" in key or "shared.weight" in key:
    #         return MergeMethods.LayerType.EMBEDD
    #
    #     # Check for attention layers first
    #     elif any(x in key for x in
    #              [".to_q.", ".to_k.", ".to_v.", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    #               ".in_proj_"]):
    #         # Add cross-attention check
    #         if ".attn2." in key:
    #             return MergeMethods.LayerType.CROSS_ATTENTION_QKV
    #         return MergeMethods.LayerType.ATTENTION_QKV
    #
    #     # Attention Projection (output projection in both CLIP-G and CLIP-L)
    #     elif any(x in key for x in [".to_out.", "self_attn.out_proj"]) and ".weight" in key:
    #         return MergeMethods.LayerType.ATTENTION_PROJ
    #
    #     # Feed Forward Network (FFN) in Stable Diffusion layers
    #     elif ".ff.net." in key and ".proj." in key:
    #         return MergeMethods.LayerType.FFN_PROJ
    #     elif ".ff.net." in key and ".weight" in key:
    #         return MergeMethods.LayerType.FFN_OUT
    #
    #     # Feed Forward Network (FFN) in CLIP-G and CLIP-L
    #     elif "mlp.c_fc" in key and ".weight" in key:
    #         return MergeMethods.LayerType.FFN_PROJ
    #     elif "mlp.c_proj" in key and ".weight" in key:
    #         return MergeMethods.LayerType.FFN_OUT
    #     elif "mlp.fc1" in key and ".weight" in key:
    #         return MergeMethods.LayerType.FFN_PROJ
    #     elif "mlp.fc2" in key and ".weight" in key:
    #         return MergeMethods.LayerType.FFN_OUT
    #
    #     # Matrix Transformation for Embedding-Like Layers (positional embeddings, projections)
    #     elif any(x in key for x in ["positional_embedding", "text_projection", "label_emb"]):
    #         return MergeMethods.LayerType.MATMUL
    #
    #     # Convolutional Layers
    #     elif len(shape) == 4:
    #         return MergeMethods.LayerType.CONV2D
    #
    #     # Default to matrix transformations
    #     return MergeMethods.LayerType.MATMUL
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
    #     R = MergeMethods.orthogonal_procrustes_ml(matrix_a, matrix_b)
    #
    #     # Cache the result
    #     if cache is not None:
    #         cache[cache_key] = R.cpu()
    #
    #     return R
    #
    # # @staticmethod
    # # @merge_method
    # # def s_ties_sum_extended(
    # #         *models: Parameter(Tensor, "delta"),
    # #         k: Parameter(Tensor) =0.218,
    # #         vote_sgn: Parameter(Tensor) =1.0,
    # #         apply_stock: Parameter(Tensor) =0.0,
    # #         cos_eps: Parameter(Tensor) =1e-6,
    # #         apply_median: Parameter(Tensor) =1.0,
    # #         eps: Parameter(Tensor) =1e-6,
    # #         maxiter: Parameter(Tensor) =150,
    # #         ftol: Parameter(Tensor) =1e-22,
    # #         weight_decay: Parameter(Tensor) =0.0218,
    # #         min_agreement: Parameter(Tensor) =0.3,
    # #         chunk_size: int = 4,  # Will be adjusted based on available memory
    # #         memory_safety_margin: float = 0.8,  # Fraction of available memory to use
    # #         **kwargs,
    # # ) -> Return(Tensor, "delta"):
    # #     """
    # #     Memory-efficient TIES implementation with dynamic chunking based on available GPU memory.
    # #     """
    # #     if not models:
    # #         raise ValueError("At least one model must be provided")
    # #
    # #     device = models[0].device
    # #     dtype = models[0].dtype
    # #     total_models = len(models)
    # #
    # #     # Calculate adaptive chunk size based on available memory if using CUDA
    # #     def get_adaptive_chunk_size(sample_model, total_models):
    # #         if device.type == 'cuda':
    # #             # Get available memory
    # #             available_memory = torch.cuda.get_device_properties(device).total_memory
    # #             free_memory = torch.cuda.memory_allocated(device)
    # #             usable_memory = (available_memory - free_memory) * memory_safety_margin
    # #
    # #             # Estimate memory needed per model
    # #             sample_size = sample_model.nelement() * sample_model.element_size()
    # #             # Account for additional tensors created during processing
    # #             estimated_overhead = sample_size * 3  # For filtered, signs, and temporary computations
    # #
    # #             # Calculate maximum models that can fit in memory
    # #             max_chunk_size = int(usable_memory / estimated_overhead)
    # #
    # #             # Ensure chunk size is at least 1 and no more than total models
    # #             return max(1, min(max_chunk_size, total_models))
    # #         return chunk_size  # Return default chunk size for CPU
    # #
    # #     # Get adaptive chunk size
    # #     adaptive_chunk_size = get_adaptive_chunk_size(models[0], total_models)
    # #
    # #     # Initialize accumulators
    # #     accumulated_filtered = []
    # #     accumulated_signs = []
    # #
    # #     # Process models in chunks with adaptive size
    # #     for chunk_start in range(0, total_models, adaptive_chunk_size):
    # #         chunk_end = min(chunk_start + adaptive_chunk_size, total_models)
    # #         chunk_models = models[chunk_start:chunk_end]
    # #
    # #         # Monitor memory before processing chunk
    # #         if device.type == 'cuda':
    # #             current_memory = torch.cuda.memory_allocated(device)
    # #             max_memory = torch.cuda.get_device_properties(device).total_memory
    # #
    # #             # If memory usage is too high, reduce chunk size
    # #             if current_memory > max_memory * 0.9:  # 90% memory threshold
    # #                 adaptive_chunk_size = max(1, adaptive_chunk_size // 2)
    # #                 print(f"Memory pressure detected. Reducing chunk size to {adaptive_chunk_size}")
    # #                 torch.cuda.empty_cache()
    # #
    # #         # Process current chunk
    # #         chunk_filtered, chunk_signs = MergeMethods._process_model_chunk(
    # #             chunk_models,
    # #             k=k,
    # #             device=device,
    # #             dtype=dtype
    # #         )
    # #
    # #         accumulated_filtered.append(chunk_filtered)
    # #         accumulated_signs.append(chunk_signs)
    # #
    # #         # Clear cache if memory pressure is high
    # #         if device.type == 'cuda' and torch.cuda.memory_allocated(device) > 0.8 * max_memory:
    # #             torch.cuda.empty_cache()
    # #
    # #     # Concatenate results
    # #     filtered_delta = torch.cat(accumulated_filtered, dim=0)
    # #     signs = torch.cat(accumulated_signs, dim=0)
    # #
    # #     # Update chunk size for downstream operations based on current memory state
    # #     if device.type == 'cuda':
    # #         adaptive_chunk_size = get_adaptive_chunk_size(filtered_delta, total_models)
    # #
    # #     # Compute final results with adaptive chunk size
    # #     final_results = MergeMethods._compute_final_results(
    # #         filtered_delta,
    # #         signs,
    # #         vote_sgn=vote_sgn,
    # #         min_agreement=min_agreement,
    # #         weight_decay=weight_decay
    # #     )
    # #
    # #     filtered_delta, param_counts = final_results
    # #
    # #     if apply_median <= 0.0:
    # #         # Model Stock pathway with adaptive chunking
    # #         if apply_stock > 0.0:
    # #             t = MergeMethods._compute_model_stock_chunked(
    # #                 filtered_delta,
    # #                 cos_eps=cos_eps,
    # #                 chunk_size=adaptive_chunk_size
    # #             )
    # #         else:
    # #             t = 1.0
    # #
    # #         filtered_delta = filtered_delta.sum(dim=0)
    # #         param_counts = torch.clamp(param_counts, min=eps)
    # #         result = filtered_delta * t / param_counts
    # #     else:
    # #         # Geometric median computation with adaptive chunks
    # #         result = MergeMethods._compute_geometric_median_chunked(
    # #             filtered_delta,
    # #             eps=eps,
    # #             maxiter=maxiter,
    # #             ftol=ftol,
    # #             chunk_size=adaptive_chunk_size
    # #         )
    # #
    # #     return torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    # #
    # # @staticmethod
    # # def _process_model_chunk(chunk_models, k, device, dtype):
    # #     """Process a chunk of models efficiently on GPU."""
    # #     chunk_filtered = []
    # #     chunk_signs = []
    # #
    # #     for model in chunk_models:
    # #         # Move single model to GPU, process, and free immediately
    # #         model_gpu = model.to(device=device, dtype=dtype)
    # #         filtered = MergeMethods.filter_top_k(model_gpu, k)
    # #         signs = torch.sign(torch.where(
    # #             torch.abs(filtered) > 1e-7,
    # #             filtered,
    # #             torch.zeros_like(filtered)
    # #         ))
    # #
    # #         chunk_filtered.append(filtered)
    # #         chunk_signs.append(signs)
    # #
    # #         # Clear GPU memory explicitly
    # #         del model_gpu
    # #         torch.cuda.empty_cache()
    # #
    # #     return (
    # #         torch.stack(chunk_filtered, dim=0),
    # #         torch.stack(chunk_signs, dim=0)
    # #     )
    # #
    # # @staticmethod
    # # def _compute_final_results(accumulated_filtered, accumulated_signs, vote_sgn, min_agreement, weight_decay):
    # #     """Compute final results efficiently on CPU."""
    # #     vote_tensor = accumulated_filtered if vote_sgn <= 0.0 else accumulated_signs
    # #     sign_sum = torch.sum(vote_tensor, dim=0)
    # #     agreement_ratio = torch.sum(accumulated_signs != 0, dim=0).float() / len(accumulated_signs)
    # #
    # #     final_sign = torch.where(
    # #         agreement_ratio >= min_agreement,
    # #         torch.sign(sign_sum),
    # #         torch.zeros_like(sign_sum)
    # #     )
    # #
    # #     delta_filters = (accumulated_signs == final_sign).float()
    # #     param_counts = torch.sum(delta_filters, dim=0)
    # #
    # #     if weight_decay > 0.0:
    # #         accumulated_filtered = accumulated_filtered * (1.0 - weight_decay)
    # #
    # #     filtered_delta = accumulated_filtered * delta_filters
    # #     return filtered_delta, param_counts
    # #
    # # @staticmethod
    # # def _compute_model_stock_chunked(filtered_delta, cos_eps, chunk_size):
    # #     """Compute model stock in memory-efficient chunks."""
    # #     n_models = filtered_delta.shape[0]
    # #     cos_sims = torch.zeros(n_models, n_models, device='cpu')
    # #
    # #     for i in range(0, n_models, chunk_size):
    # #         chunk_i = filtered_delta[i:i + chunk_size].flatten(1)
    # #         chunk_i_norm = torch.norm(chunk_i, dim=1, keepdim=True)
    # #
    # #         for j in range(0, n_models, chunk_size):
    # #             chunk_j = filtered_delta[j:j + chunk_size].flatten(1)
    # #             chunk_j_norm = torch.norm(chunk_j, dim=1, keepdim=True)
    # #
    # #             # Compute cosine similarity for the chunk
    # #             chunk_cos = torch.mm(chunk_i, chunk_j.t()) / (
    # #                     torch.mm(chunk_i_norm, chunk_j_norm.t()) + cos_eps
    # #             )
    # #
    # #             cos_sims[i:i + chunk_size, j:j + chunk_size] = chunk_cos.cpu()
    # #
    # #             del chunk_j, chunk_j_norm
    # #             torch.cuda.empty_cache()
    # #
    # #         del chunk_i, chunk_i_norm
    # #         torch.cuda.empty_cache()
    # #
    # #     # Compute final t score
    # #     t = torch.mean(cos_sims > 0).item()
    # #     return t
    # #
    # # @staticmethod
    # # def _compute_geometric_median_chunked(points, eps, maxiter, ftol, chunk_size):
    # #     """
    # #     Optimized memory-efficient geometric median computation for 3D tensors.
    # #     points shape: [n_points, d1, d2] where d1, d2 are the dimensions of each point
    # #     """
    # #     n_points = points.shape[0]
    # #     device = points.device
    # #     points_shape = points.shape
    # #
    # #     # Keep more data on GPU
    # #     points_flat = points.reshape(n_points, -1)  # Keep on GPU initially
    # #     weights = torch.ones(n_points, device=device)  # Keep weights on GPU
    # #
    # #     # Initialize median on GPU
    # #     median = torch.mean(points_flat, dim=0)
    # #     best_objective = float('inf')
    # #     best_median = median.clone()
    # #
    # #     # Process larger chunks on GPU
    # #     for iter_idx in range(maxiter):
    # #         prev_objective = 0.0
    # #         new_weights = torch.zeros_like(weights)
    # #
    # #         # Process chunks directly on GPU
    # #         for i in range(0, n_points, chunk_size):
    # #             chunk = points_flat[i:i + chunk_size]  # Already on GPU
    # #             chunk_weights = weights[i:i + chunk_size]
    # #
    # #             # Compute distances efficiently on GPU
    # #             diff = chunk - median.unsqueeze(0)
    # #             # Use efficient GPU operations
    # #             distances = torch.norm(diff, dim=1) + eps
    # #
    # #             # Update objective and weights on GPU
    # #             prev_objective += torch.sum(distances * chunk_weights)
    # #             new_weights[i:i + chunk_size] = chunk_weights / distances
    # #
    # #             # Optional: Only clear if memory pressure is high
    # #             if torch.cuda.memory_allocated() > 0.9 * torch.cuda.max_memory_allocated():
    # #                 del diff, distances
    # #                 torch.cuda.empty_cache()
    # #
    # #         # Update median efficiently on GPU
    # #         weighted_sum = torch.zeros_like(median)
    # #         weight_sum = new_weights.sum()
    # #
    # #         # Compute new median in chunks but stay on GPU
    # #         for i in range(0, n_points, chunk_size):
    # #             chunk = points_flat[i:i + chunk_size]
    # #             chunk_weights = new_weights[i:i + chunk_size]
    # #             weighted_sum += torch.sum(chunk * chunk_weights.unsqueeze(1), dim=0)
    # #
    # #         median = weighted_sum / (weight_sum + eps)
    # #
    # #         # Check convergence
    # #         if abs(prev_objective - best_objective) <= ftol * best_objective:
    # #             break
    # #
    # #         if prev_objective < best_objective:
    # #             best_objective = prev_objective
    # #             best_median = median.clone()
    # #
    # #         weights = new_weights
    # #
    # #         # Optional: Print progress every few iterations
    # #         if iter_idx % 10 == 0:
    # #             print(f"Iteration {iter_idx}, Objective: {prev_objective:.4e}")
    # #
    # #     # Reshape final result
    # #     final_median = best_median.reshape(points_shape[1:])
    # #     return final_median
    # #
    # # @staticmethod
    # # def filter_top_k(a: Tensor, k: float) -> torch.Tensor:
    # #     """Improved implementation using kthvalue with chunking."""
    # #     total_params = torch.numel(a)
    # #     k_params = max(int((1 - k) * total_params), 1)
    # #
    # #     if k_params >= total_params:
    # #         return torch.zeros_like(a)
    # #
    # #     # Process in chunks for memory efficiency
    # #     chunk_size = 1_000_000
    # #     abs_values = []
    # #
    # #     for i in range(0, total_params, chunk_size):
    # #         chunk = a.flatten()[i:i + chunk_size]
    # #         abs_values.append(torch.abs(chunk))
    # #
    # #     # Concatenate chunks and find kth value
    # #     abs_cat = torch.cat(abs_values)
    # #     k_value = torch.kthvalue(abs_cat, k_params).values
    # #
    # #     # Apply threshold with memory efficiency
    # #     mask = torch.abs(a) >= k_value
    # #     return a * mask.float()
    #

    @merge_method
    def svd_ties_sum_extended(
            *models: Parameter(Tensor, "delta"),
            k: Parameter(float) = 1.0,
            max_singular_values: Parameter(int) = 64,
            energy_threshold: Parameter(float) = 0.9,
            power_iterations: Parameter(int) = 1,
            vote_sgn: Parameter(float) = 1.0,
            apply_stock: Parameter(float) = 0.0,
            cos_eps: Parameter(float) = 1e-6,
            apply_median: Parameter(float) = 1.0,
            eps: Parameter(float) = 1e-6,
            maxiter: Parameter(int) = 150,
            ftol: Parameter(float) = 1e-22,
            weight_decay: Parameter(float) = 0.0218,  # .0218,
            min_agreement: Parameter(float) = 0.3,
            chunk_size: Parameter(int) = 4,
            memory_safety_margin: Parameter(float) = 0.9,  # Default to 90% usage
            tensor_chunk_size: Parameter(float) = -1.0,
            **kwargs,
    ) -> Return(Tensor, "delta"):
        """
        Memory-efficient TIES with dual-level (model + tensor) chunking.
        Dynamically adapts to use up to 90% of available VRAM by default.
        """
        if not models:
            raise ValueError("At least one model must be provided")

        # Enable faster math modes if available
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        device = models[0].device
        dtype = models[0].dtype
        total_models = len(models)
        tensor_shape = models[0].shape

        tensor_chunk_size = int(tensor_chunk_size) if tensor_chunk_size > 0 else None

        def get_optimized_chunks():
            if device.type != 'cuda':
                return chunk_size, tensor_chunk_size or 1024

            total_mem = torch.cuda.get_device_properties(device).total_memory
            free_mem = total_mem - torch.cuda.memory_allocated(device)
            usable_mem = free_mem * memory_safety_margin

            # Base memory calculation with LoRA-style approximation
            model_size = models[0].nelement() * models[0].element_size()

            # LoRA memory (A and B matrices)
            max_dim = max(tensor_shape)
            lora_rank = min(max_singular_values, 64)
            lora_mem = 2 * (max_dim * lora_rank) * models[0].element_size()  # A and B matrices

            # Batch-friendly calculation
            elements_per_batch = (usable_mem * 0.9) // (model_size + lora_mem)  # Use 90% VRAM
            safe_model_chunk = max(4, min(  # Allow larger batches
                int(elements_per_batch),
                total_models
            ))

            # Tensor chunk sizing
            if tensor_chunk_size is None or tensor_chunk_size <= 0:
                elements_per_chunk = (usable_mem * 0.8) // (safe_model_chunk * models[0].element_size())
                tensor_chunk = max(512, int(elements_per_chunk ** 0.5))  # Minimum 512 elements
            else:
                tensor_chunk = tensor_chunk_size

            return (
                safe_model_chunk,
                min(tensor_chunk, max_dim)
            )

        model_chunk_size, tensor_chunk_size = get_optimized_chunks()

        def batched_svd(matrices: Tensor) -> Tensor:
            """Handle various parameter types safely"""
            # Add dimensionality check
            if matrices.ndim not in [2, 3]:
                # Return original tensor for non-matrix params
                return matrices.mean(dim=0) if matrices.ndim > 3 else matrices

            # Ensure 3D shape even for single matrices
            if matrices.ndim == 2:
                matrices = matrices.unsqueeze(0)

            batch_size, m, n = matrices.shape
            max_rank = min(m, n, max_singular_values)

            # Power iteration initialized orthogonal basis
            A = torch.empty((batch_size, m, max_rank), device=device, dtype=dtype)
            torch.nn.init.orthogonal_(A)

            # Initial B computation (guaranteed to happen)
            B = torch.linalg.lstsq(A, matrices).solution

            # Power iteration refinement loop
            for _ in range(power_iterations):
                B = torch.linalg.lstsq(A, matrices).solution
                A = torch.linalg.lstsq(B.mT, matrices.mT).solution.mT

            # Approximate SVs via column norms
            sv = torch.linalg.norm(B, dim=2)  # (batch, rank)
            sv_sq_cumsum = torch.cumsum(sv ** 2, dim=-1)
            total_energy = sv_sq_cumsum[:, -1].unsqueeze(1)

            # Find first index meeting energy threshold per batch
            effective_rank = torch.argmax(
                (sv_sq_cumsum >= energy_threshold * total_energy).float(),  # Convert bool to float
                dim=-1
            ).clamp_min(1)

            # Use median rank across current batch for consistency
            final_rank = torch.median(effective_rank).int().clamp(1, max_rank)

            # Truncate to effective rank
            A_trunc = A[..., :final_rank]
            B_trunc = B[..., :final_rank, :]

            # Reconstruct and align signs
            recon = A_trunc @ B_trunc
            sign_match = torch.sign(recon) * torch.sign(matrices)

            return recon * sign_match.mean(dim=0, keepdim=True)

        # Initialize output tensor with page-locked memory
        final_result = torch.zeros_like(models[0], device='cpu', pin_memory=True)
        chunk_dim = 0 if tensor_shape[0] >= tensor_shape[1] else 1
        tensor_len = tensor_shape[chunk_dim]

        # Main processing loop with memory optimization
        for tensor_start in range(0, tensor_len, tensor_chunk_size):
            tensor_end = min(tensor_start + tensor_chunk_size, tensor_len)

            # Prepare sliced tensor chunk with async transfer
            model_slices = []
            for model in models:
                slice_args = tuple(
                    slice(tensor_start, tensor_end) if d == chunk_dim else slice(None)
                    for d in range(len(tensor_shape))
                )
                model_slices.append(model[slice_args].to(device, dtype, non_blocking=True))

            # Process in optimized batches
            chunk_filtered, chunk_signs = [], []
            for batch_start in range(0, total_models, model_chunk_size):
                batch = model_slices[batch_start:batch_start + model_chunk_size]

                # Original k-based filtering
                filtered, signs = MergeMethods._process_model_chunk(
                    batch,
                    k=k,
                    device=device,
                    dtype=dtype
                )

                # # Batch-optimized SVD with mixed precision
                # with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                #     svd_batch = batched_svd(filtered)
                #     svd_signs = torch.sign(svd_batch)

                svd_batch = batched_svd(filtered)
                if svd_batch.ndim > 3:  # For non-matrix params
                    svd_batch = svd_batch.mean(dim=0)
                svd_signs = torch.sign(svd_batch)

                chunk_filtered.append(svd_batch)
                chunk_signs.append(svd_signs)

            # Aggregate and process results
            filtered_delta = torch.cat(chunk_filtered)
            signs = torch.cat(chunk_signs)

            # Compute final chunk results
            result_chunk = MergeMethods._compute_final_chunk(
                filtered_delta, signs, vote_sgn, min_agreement, weight_decay,
                apply_stock, cos_eps, apply_median, eps, maxiter, ftol,
                model_chunk_size, tensor_chunk_size, device
            )

            # Update final tensor with page-locked memory copy
            slice_args = tuple(
                slice(tensor_start, tensor_end) if d == chunk_dim else slice(None)
                for d in range(len(tensor_shape))
            )
            final_result[slice_args] = result_chunk.to("cpu", non_blocking=True)

            # Managed memory cleanup
            del model_slices, chunk_filtered, chunk_signs, filtered_delta, signs
            if device.type == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        return final_result.to(device).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _process_model_chunk(chunk_models, k, device, dtype):
        """Process a chunk of models with tensor chunking."""
        filtered_chunks = []
        sign_chunks = []

        for model in chunk_models:
            # Process in tensor chunks
            filtered = MergeMethods.filter_top_k(model, k)
            signs = torch.sign(filtered)
            filtered_chunks.append(filtered)
            sign_chunks.append(signs)

        return torch.stack(filtered_chunks), torch.stack(sign_chunks)

    @staticmethod
    def _compute_final_chunk(filtered_delta, signs, vote_sgn, min_agreement, weight_decay,
                             apply_stock, cos_eps, apply_median, eps, maxiter, ftol,
                             model_chunk_size, tensor_chunk_size, device):
        """Compute final merged values for a tensor chunk."""
        # Compute agreement and filtering
        vote_tensor = filtered_delta if vote_sgn <= 0.0 else signs
        sign_sum = torch.sum(vote_tensor, dim=0)
        agreement_ratio = torch.sum(signs != 0, dim=0).float() / len(signs)

        final_sign = torch.where(
            agreement_ratio >= min_agreement,
            torch.sign(sign_sum),
            torch.zeros_like(sign_sum)
        )

        delta_filters = (signs == final_sign).float()
        param_counts = torch.sum(delta_filters, dim=0)

        if weight_decay > 0.0:
            filtered_delta = filtered_delta * (1.0 - weight_decay)

        filtered_delta *= delta_filters

        # Apply merge method
        if apply_median <= 0.0:
            if apply_stock > 0.0:
                t = MergeMethods._compute_model_stock_chunked(
                    filtered_delta,
                    cos_eps=cos_eps,
                    chunk_size=model_chunk_size
                )
            else:
                t = 1.0

            result = filtered_delta.sum(dim=0) * t / torch.clamp(param_counts, min=eps)
        else:
            result = MergeMethods._compute_geometric_median_chunked(
                filtered_delta,
                eps=eps,
                maxiter=maxiter,
                ftol=ftol,
                chunk_size=tensor_chunk_size
            )

        return result

    @staticmethod
    def _compute_model_stock_chunked(filtered_delta, cos_eps, chunk_size):
        """Memory-efficient cosine similarity calculation."""
        n_models = filtered_delta.shape[0]
        total = 0.0
        count = 0

        for i in range(0, n_models, chunk_size):
            chunk_i = filtered_delta[i:i + chunk_size].flatten(1)
            norm_i = torch.norm(chunk_i, dim=1, keepdim=True)

            for j in range(i, n_models, chunk_size):
                chunk_j = filtered_delta[j:j + chunk_size].flatten(1)
                norm_j = torch.norm(chunk_j, dim=1, keepdim=True)

                chunk_cos = torch.mm(chunk_i, chunk_j.T) / (torch.mm(norm_i, norm_j.T) + cos_eps)
                total += torch.sum(chunk_cos > 0).item()
                count += chunk_cos.numel()

                del chunk_j, norm_j
                torch.cuda.empty_cache()

            del chunk_i, norm_i
            torch.cuda.empty_cache()

        return total / count if count > 0 else 0.0

    @staticmethod
    def _compute_geometric_median_chunked(points, eps, maxiter, ftol, chunk_size):
        """Optimized geometric median with full chunking."""
        n_points, *dims = points.shape
        device = points.device
        median = torch.mean(points.view(n_points, -1), dim=0)

        for _ in range(maxiter):
            weighted_sum = torch.zeros_like(median)
            weight_sum = 0.0

            # Process distance calculations in chunks
            for i in range(0, n_points, chunk_size):
                chunk = points[i:i + chunk_size].view(-1, median.shape[0])
                chunk_dist = torch.norm(chunk - median, dim=1) + eps
                chunk_weights = 1 / chunk_dist

                weighted_sum += torch.sum(chunk * chunk_weights[:, None], dim=0)
                weight_sum += torch.sum(chunk_weights)

                # Prevent memory accumulation
                del chunk, chunk_dist, chunk_weights
                torch.cuda.empty_cache()

            new_median = weighted_sum / weight_sum.clamp(min=eps)

            if torch.norm(new_median - median) < ftol:
                break
            median = new_median.clone()

        return median.view(*dims)

    @staticmethod
    def filter_top_k(a: Tensor, k: float) -> torch.Tensor:
        """Memory-optimized top-k filtering with safe kthvalue handling."""
        total_elements = a.numel()
        k_val = max(int((1 - k) * total_elements), 1)

        if k_val >= total_elements:
            return torch.zeros_like(a)

        # Find threshold with chunked processing
        chunk_size = 1_000_000
        threshold = torch.tensor(float('inf'), device=a.device)
        remaining_k = k_val

        for i in range(0, total_elements, chunk_size):
            chunk = a.flatten()[i:i + chunk_size].abs()
            chunk_elements = chunk.numel()

            if remaining_k <= 0:
                break

            # Calculate how many elements we need from this chunk
            current_k = min(max(remaining_k, 1), chunk_elements)  # Clamp between 1 and chunk size
            chunk_thresh = torch.kthvalue(chunk, current_k).values

            # Update threshold and remaining elements to find
            threshold = torch.minimum(threshold, chunk_thresh)
            remaining_k -= current_k

        # Final safety check
        valid_threshold = threshold if not torch.isinf(threshold) else torch.tensor(0.0, device=a.device)
        return a * (a.abs() >= valid_threshold).to(a.dtype)

    @merge_method
    def svd_ties_sum_extended_v13(
            *models: Parameter(Tensor, "delta"),
            passthrough_index: Parameter(int) = 0,
            k: Parameter(float) = 1.0,
            max_singular_values: Parameter(int) = 64,
            energy_threshold: Parameter(float) = 0.9,
            power_iterations: Parameter(int) = 1,
            vote_sgn: Parameter(float) = 1.0,
            apply_stock: Parameter(float) = 0.0,
            cos_eps: Parameter(float) = 1e-6,
            apply_median: Parameter(float) = 1.0,
            eps: Parameter(float) = 1e-6,
            maxiter: Parameter(int) = 150,
            ftol: Parameter(float) = 1e-22,
            weight_decay: Parameter(float) = 0.0218,
            min_agreement: Parameter(float) = 0.3,
            memory_safety_margin: Parameter(float) = 0.8,
            **kwargs,
    ) -> Return(Tensor, "delta"):
        """
        Correctly implements hybrid chunking for massive tensors in concurrent environments.
        - Outer loop performs SPATIAL chunking (slicing large tensors).
        - Inner loop performs BATCH chunking (processing a few tensors at a time).
        - This robustly handles scenarios with many, very large input tensors.
        """
        if not models:
            raise ValueError("Onii-chan, you have to give me at least one model tensor!")

        if k == 0.0 and min_agreement == 0.0 and energy_threshold == 0.0:
            if 0 <= passthrough_index < len(models):
                return models[passthrough_index].clone()
            else:
                return torch.zeros_like(models[0])

        tensor_template = models[0]
        original_shape = tensor_template.shape
        original_ndim = tensor_template.ndim

        if original_ndim <= 1:
            with torch.no_grad():
                stacked = torch.stack([m.to(torch.float32) for m in models])
                average_tensor = torch.mean(stacked, dim=0)
                return average_tensor.to(dtype=tensor_template.dtype)

        if original_ndim == 4:
            # We must keep the output channels (dim 0) separate!
            # Reshape [out, in, h, w] -> [out, in*h*w]
            reshaped_models = [m.reshape(m.shape[0], -1) for m in models]
            models = tuple(reshaped_models)

        device = models[0].device
        dtype = models[0].dtype
        total_tensors = len(models)

        tensor_batch_size, spatial_chunk_size = MergeMethods._get_optimized_chunks_v12(
            models[0], total_tensors, memory_safety_margin, dtype, max_singular_values
        )

        final_result_2d = torch.zeros_like(models[0], device='cpu', pin_memory=True)

        with torch.no_grad():
            filtered_tensors = [MergeMethods.filter_top_k_v2(m, k) for m in models]

            tensor_shape = models[0].shape
            chunk_dim = 0 if tensor_shape[0] >= tensor_shape[1] else 1
            tensor_len = tensor_shape[chunk_dim]

            for tensor_start in range(0, tensor_len, spatial_chunk_size):
                tensor_end = min(tensor_start + spatial_chunk_size, tensor_len)
                slice_obj = tuple(
                    slice(tensor_start, tensor_end) if d == chunk_dim else slice(None)
                    for d in range(len(tensor_shape))
                )

                collected_deltas = []
                for i in range(0, total_tensors, tensor_batch_size):
                    batch_tensors_cpu = filtered_tensors[i:i + tensor_batch_size]
                    batch_slices_gpu = torch.stack(
                        [t[slice_obj].to(device, non_blocking=True) for t in batch_tensors_cpu]
                    )

                    reconstructed_batch = MergeMethods._approximate_svd_v2(
                        batch_slices_gpu,
                        max_rank=max_singular_values, power_iterations=power_iterations,
                        energy_threshold=energy_threshold
                    )

                    collected_deltas.append(reconstructed_batch)
                    del batch_slices_gpu
                    if device.type == 'cuda': torch.cuda.empty_cache()

                filtered_delta = torch.cat(collected_deltas)
                signs = torch.sign(filtered_delta)

                result_chunk = MergeMethods._compute_final_chunk_v2(
                    filtered_delta, signs,
                    vote_sgn, min_agreement, weight_decay, apply_stock, cos_eps,
                    apply_median, eps, maxiter, ftol, total_tensors
                )

                final_result_2d[slice_obj] = result_chunk.to('cpu', non_blocking=True)

                del collected_deltas, filtered_delta, signs, result_chunk
                if device.type == 'cuda': torch.cuda.empty_cache()

        final_result = final_result_2d.to(device)
        if original_ndim == 4:
            final_result = final_result.reshape(original_shape)

        return final_result.nan_to_num(0.0)

    @staticmethod
    def _get_optimized_chunks_v12(tensor_template: Tensor, total_tensors: int, margin: float, dtype: torch.dtype,
                                  max_rank: int) -> (int, int):
        """
        An aggressive and more precise memory calculator.
        - It correctly identifies the single largest memory allocation.
        - It uses more precise formulas for memory costs.
        - Goal: Maximize VRAM utilization without crashing.
        """
        if tensor_template.device.type != 'cuda':
            return total_tensors, max(tensor_template.shape)

        total_mem = torch.cuda.get_device_properties(tensor_template.device).total_memory
        free_mem = total_mem - torch.cuda.memory_allocated(tensor_template.device)
        usable_mem = free_mem * margin

        m, n = tensor_template.shape
        element_size = torch.tensor([], dtype=dtype).element_size()
        rank = min(m, n, max_rank)

        # --- Let's calculate the memory cost PER ROW for our two main operations ---

        # 1. Cost per row for the SVD step (for a single tensor in a batch)
        #    Cost is: (base_slice + A_matrix + B_matrix)
        #    For one row: (n_cols + rank_cols + n_cols)
        mem_per_row_svd = (n + rank + n) * element_size

        # 2. Cost per row for the final concatenated tensor (`filtered_delta` + `signs`)
        #    For one row: (n_cols * total_tensors * 2) because we hold both tensors.
        mem_per_row_final = (n * total_tensors * 2) * element_size

        # The LARGEST of these two determines our memory bottleneck.
        dominant_mem_per_row = max(mem_per_row_svd, mem_per_row_final)

        if dominant_mem_per_row == 0:
            safe_spatial_chunk = m
        else:
            # How many rows can we afford, based on the most expensive operation?
            num_rows_can_fit = math.floor(usable_mem / dominant_mem_per_row)
            safe_spatial_chunk = max(256, int(num_rows_can_fit))

        # With our new, aggressive spatial chunk, let's find the batch size.
        # We can now calculate the exact memory needed for SVD on one slice.
        current_m = min(safe_spatial_chunk, m)
        mem_for_one_svd_slice = (
                                        (current_m * n) +  # Base slice
                                        (current_m * rank) +  # Matrix A
                                        (n * rank)  # Matrix B
                                ) * element_size

        if mem_for_one_svd_slice == 0:
            safe_batch_size = total_tensors
        else:
            # How many SVD operations can we stack in our memory budget?
            num_batches_can_fit = math.floor(usable_mem / mem_for_one_svd_slice)
            safe_batch_size = max(1, int(num_batches_can_fit))

        return safe_batch_size, min(safe_spatial_chunk, m)

    # --- Helper methods from v2/v3 can be reused as they are clean ---
    @staticmethod
    def filter_top_k_v2(a: Tensor, k: float) -> Tensor:
        if k >= 1.0: return a
        if k <= 0.0: return torch.zeros_like(a)
        flat_abs = a.abs().flatten()
        k_val = max(1, int((1.0 - k) * flat_abs.numel()))
        if k_val >= flat_abs.numel(): return torch.zeros_like(a)
        threshold = torch.kthvalue(flat_abs, k_val).values
        return a * (a.abs() >= threshold)

    @staticmethod
    def _approximate_svd_v2(matrices: Tensor, max_rank: int, power_iterations: int,
                            energy_threshold: float) -> Tensor:
        if matrices.ndim < 2: return matrices
        if matrices.ndim == 2: matrices = matrices.unsqueeze(0)

        batch_size, m, n = matrices.shape
        rank = min(m, n, max_rank)

        A = torch.empty(batch_size, m, rank, device=matrices.device, dtype=matrices.dtype)
        torch.nn.init.orthogonal_(A)

        for _ in range(power_iterations):
            B = torch.linalg.lstsq(A, matrices).solution
            A = torch.linalg.lstsq(B.mT, matrices.mT).solution.mT

        singular_values_sq = torch.sum(B ** 2, dim=2)
        total_energy = torch.sum(singular_values_sq, dim=-1, keepdim=True)
        energy_cumsum = torch.cumsum(singular_values_sq, dim=-1)
        rank_indices = torch.argmax((energy_cumsum >= energy_threshold * total_energy).float(), dim=-1)
        final_rank = torch.median(rank_indices).int().clamp(min=1, max=rank).item()

        A_trunc, B_trunc = A[..., :final_rank], B[..., :final_rank, :]
        reconstructed = A_trunc @ B_trunc

        # --- THE CORRECTED SIGN ALIGNMENT LOGIC ---
        # We calculate the dot product for each matrix in the batch.
        # This is much more robust than my old, buggy method.
        dot_products = torch.sum(reconstructed * matrices, dim=(-1, -2), keepdim=True)
        signs = torch.sign(dot_products)

        # We use .detach() on the signs to ensure no weird gradients flow back, just in case.
        return reconstructed * signs.detach()

    @staticmethod
    def _compute_final_chunk_v2(
            filtered_delta, signs, vote_sgn, min_agreement, weight_decay,
            apply_stock, cos_eps, apply_median, eps, maxiter, ftol,
            processing_batch_size
    ):
        vote_tensor = signs if vote_sgn > 0.0 else filtered_delta
        sign_sum = torch.sum(vote_tensor, dim=0)
        agreement_mask = signs != 0
        agreement_ratio = agreement_mask.float().sum(dim=0) / signs.shape[0]
        final_sign = torch.sign(sign_sum)
        final_sign[agreement_ratio < min_agreement] = 0
        delta_filters = (signs == final_sign).float()
        param_counts = torch.sum(delta_filters, dim=0)
        if weight_decay > 0.0:
            filtered_delta = filtered_delta * (1.0 - weight_decay)
        filtered_delta *= delta_filters
        if apply_median > 0.0:
            return MergeMethods._compute_geometric_median_chunked_v2(
                filtered_delta, eps, maxiter, ftol, processing_batch_size
            )
        else:
            t = 1.0
            if apply_stock > 0.0:
                t = MergeMethods._compute_model_stock_chunked_v2(
                    filtered_delta, cos_eps, processing_batch_size
                )
            return (filtered_delta.sum(dim=0) * t) / param_counts.clamp(min=eps)

    # Note: The chunked median and stock methods are still useful for memory,
    # so I just cleaned them up and renamed them to _v2.
    @staticmethod
    def _compute_geometric_median_chunked_v2(points, eps, maxiter, ftol, chunk_size):
        """Optimized geometric median with chunking."""
        n_points, *dims = points.shape
        points_flat = points.view(n_points, -1)
        median = torch.mean(points_flat, dim=0)

        for _ in range(maxiter):
            prev_median = median.clone()

            weighted_sum = torch.zeros_like(median)
            weight_sum = torch.zeros(1, device=median.device)

            for i in range(0, n_points, chunk_size):
                chunk = points_flat[i:i + chunk_size]
                dist = torch.norm(chunk - median, dim=1)
                inv_dist = 1.0 / dist.clamp(min=eps)

                weighted_sum += torch.sum(chunk * inv_dist[:, None], dim=0)
                weight_sum += torch.sum(inv_dist)

            median = weighted_sum / weight_sum.clamp(min=eps)
            if torch.norm(median - prev_median) < ftol:
                break

        return median.view(*dims)

    @staticmethod
    def _compute_model_stock_chunked_v2(filtered_delta, cos_eps, chunk_size):
        """Memory-efficient cosine similarity calculation."""
        n_models = filtered_delta.shape[0]
        flat_delta = filtered_delta.flatten(1)
        total_sum = 0.0

        for i in range(0, n_models, chunk_size):
            chunk_i = flat_delta[i:i + chunk_size]
            norm_i = torch.norm(chunk_i, p=2, dim=1, keepdim=True)
            for j in range(i, n_models, chunk_size):
                chunk_j = flat_delta[j:j + chunk_size]
                norm_j = torch.norm(chunk_j, p=2, dim=1, keepdim=True)

                # Cosine similarity
                cos_sim = (chunk_i @ chunk_j.T) / ((norm_i @ norm_j.T) + cos_eps)

                # In the original, it seems you just wanted the positive ratio
                total_sum += torch.sum(cos_sim > 0).item()

        return total_sum / (n_models * n_models) if n_models > 0 else 0.0

    # @staticmethod
    # @merge_method
    # def ties_sum_with_dropout(
    #         *models: Parameter(Tensor, "delta"),
    #         probability: Parameter(Tensor) =0.9,
    #         della_eps: Parameter(Tensor) =0.0,
    #         rescale: Parameter(Tensor) =1.0,
    #         lambda_scale: Parameter(Tensor) =2.0,
    #         k: Parameter(Tensor) =0.218,
    #         vote_sgn: Parameter(Tensor) =0,
    #         apply_stock: Parameter(Tensor) =0.0,
    #         cos_eps: Parameter(Tensor) =1e-6,
    #         apply_median: Parameter(Tensor) =1.0,
    #         eps: Parameter(Tensor) =1e-5,
    #         maxiter: Parameter(Tensor) =150,
    #         ftol: Parameter(Tensor) =1e-11,
    #         seed: Parameter(Tensor) =218,
    #         **kwargs,
    # ) -> Return(Tensor, "delta"):
    #     """
    #     Applies TIES merging with dropout to a variable number of delta tensors.
    #
    #     Args:
    #         *models: The delta tensors to merge.
    #         probability: The dropout probability (0 <= probability <= 1).
    #         della_eps: The DELLA epsilon parameter, controlling magnitude-based dropout.
    #         rescale:  The rescaling factor for the merged delta.
    #         k: The TIES parameter trimming threshold.
    #         vote_sgn:  The TIES-SOUP mode activation parameter.
    #         apply_stock:  The Model Stock activation parameter.
    #         cos_eps: The cosine similarity epsilon for Model Stock.
    #         apply_median:  The Geometric Median activation parameter.
    #         eps: The epsilon for the Geometric Median calculation.
    #         maxiter: The maximum number of iterations for Geometric Median.
    #         ftol:  The tolerance for convergence for Geometric Median.
    #         seed: The random seed for dropout.
    #         **kwargs: Additional keyword arguments.
    #
    #     Returns:
    #         The merged delta tensor.
    #     """
    #     if not models or probability == 1:
    #         return torch.tensor(0.0, device=models[0].device if models else 'cpu')
    #
    #     device = models[0].device
    #     generator = torch.Generator(device)
    #     if seed is not None:
    #         generator.manual_seed(seed)
    #
    #     # Apply dropout to each delta tensor
    #     dropped_deltas = []
    #     for delta in models:
    #         dropout_mask = MergeMethods.create_dropout_mask(delta, probability, della_eps, generator)
    #         dropped_deltas.append(delta * dropout_mask)
    #
    #     # Apply TIES merging to the dropped deltas
    #     merged_delta = MergeMethods.streaming_ties_sum_extended.__wrapped__(
    #         *dropped_deltas,
    #         k=k,
    #         vote_sgn=vote_sgn,
    #         apply_stock=apply_stock,
    #         cos_eps=cos_eps,
    #         apply_median=apply_median,
    #         eps=eps,
    #         maxiter=maxiter,
    #         ftol=ftol
    #     )
    #
    #     active_ratio = 1.0 - probability
    #     rescalar = 1.0 / (active_ratio ** rescale + 1e-7)
    #     return merged_delta * rescalar * lambda_scale
    #
    # def create_dropout_mask(delta: Tensor, probability: float, della_eps: float, generator: torch.Generator) -> Tensor:
    #     """Paper-correct MAGPRUNE dropout mask."""
    #     # 1. Descending magnitude ranking
    #     flat_abs = delta.abs().flatten()
    #     ranks = torch.argsort(flat_abs, descending=True).argsort().float().reshape(delta.shape)
    #
    #     # 2. Paper's DELLA formula (Section 3.2)
    #     n = delta.numel()
    #     median_rank = n // 2
    #     delta_i = (ranks - median_rank) * della_eps / n
    #
    #     # 3. Clamp probabilities
    #     p_min = 1 - probability
    #     probabilities = torch.clamp(p_min + delta_i, 0.0, 1.0)
    #
    #     return torch.bernoulli(probabilities, generator=generator)
    #
    # @staticmethod
    # @merge_method
    # def model_aware_merge(
    #         a: Parameter(Tensor, "weight"),
    #         b: Parameter(Tensor, "weight"),
    #         *,
    #         alpha: Parameter(Tensor) =1.0,
    #         n_levels: int = 4,
    #         **kwargs,
    # ) -> Return(Tensor):
    #     # key = kwargs["key"]
    #     # layer_info = MergeMethods.identify_layer(key, a)
    #
    #     # if 'attn' in key and block_attention > 0.0:
    #     #     # Calculate attention-specific agreement with temperature parameter
    #     #     agreement = MergeMethods.calculate_attention_agreement(
    #     #         a,
    #     #         b,
    #     #         temperature=temperature,
    #     #         pattern_weight=pattern_weight,
    #     #         value_weight=value_weight
    #     #     )
    #     #     mask = agreement
    #     # elif layer_info.type in [MergeMethods.LayerType2.CONV_3X3, MergeMethods.LayerType2.CONV_1X1]:
    #     #     agreement = MergeMethods.calculate_spatial_agreement(
    #     #         a,
    #     #         b,
    #     #         kernel_size=window_size,
    #     #         local_weight=local_weight,
    #     #         channel_weight=channel_weight
    #     #     )
    #     #     mask = agreement
    #     # else:
    #     #     agreement = MergeMethods.calculate_advanced_agreement(
    #     #         a,
    #     #         b,
    #     #         window_size=window_size,
    #     #         cosine_weight=cosine_weight,
    #     #         structural_weight=structural_weight,
    #     #         frequency_weight=frequency_weight
    #     #     )
    #     #     mask = agreement
    #
    #     # Merge using laplacian_difference with custom parameters
    #     merged = MergeMethods.laplacian_difference(
    #         a,
    #         b,
    #         alpha=alpha,
    #         n_levels=n_levels,
    #     )
    #
    #     # print("\n=== Final Merged Tensor ===")
    #     # print(f"Shape: {merged.shape}")
    #     # print(f"Stats: min={merged.min().item():.3f}, max={merged.max().item():.3f}, mean={merged.mean().item():.3f}")
    #     # print(f"NaN/Inf Check: {torch.isnan(merged).any()} NaN, {torch.isinf(merged).any()} Inf")
    #
    #     return merged
    #
    # # def calculate_advanced_agreement(a: Tensor, b: Tensor, window_size: int = 5,
    # #                                  cosine_weight: float = 0.4, structural_weight: float = 0.4,
    # #                                  frequency_weight: float = 0.2) -> Tensor:
    # #     """
    # #     Calculate agreement with automatic shape handling
    # #     """
    # #
    # #     # For embedding/linear layers: scalar agreement + broadcast
    # #     if len(a.shape) == 2 and a.shape[0] > 1000:  # Embedding detection
    # #         agreement = F.cosine_similarity(a.flatten(), b.flatten(), dim=0)
    # #         return torch.clamp(agreement, 0, 1)
    # #
    # #     # Original implementation for other layers
    # #     min_size = min(a.numel(), b.numel())
    # #     a_flat = a.flatten()[:min_size]
    # #     b_flat = b.flatten()[:min_size]
    # #
    # #     # Cosine similarity
    # #     cosine_sim = F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0), dim=1)
    # #
    # #     # --- Structural Similarity (1D-safe) ---
    # #     def calculate_ssim_1d(x: Tensor, y: Tensor):
    # #         # Add channel and batch dimensions [B, C, L]
    # #         x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, L]
    # #         y = y.unsqueeze(0).unsqueeze(0)
    # #
    # #         padding = window_size // 2
    # #         x_padded = F.pad(x, (padding, padding), mode='replicate')  # Supported 3D padding
    # #         y_padded = F.pad(y, (padding, padding), mode='replicate')
    # #
    # #         # Extract windows [B, C, L, window_size]
    # #         x_windows = x_padded.unfold(-1, window_size, 1)
    # #         y_windows = y_padded.unfold(-1, window_size, 1)
    # #
    # #         # Calculate statistics
    # #         x_mean = x_windows.mean(-1)  # [1, 1, L]
    # #         y_mean = y_windows.mean(-1)
    # #         x_var = x_windows.var(-1, unbiased=False)
    # #         y_var = y_windows.var(-1, unbiased=False)
    # #         cov = (x_windows * y_windows).mean(-1) - x_mean * y_mean
    # #
    # #         # Stability constants
    # #         C1 = (0.01 * 1.0) ** 2  # Fixed for 0-1 normalized data
    # #         C2 = (0.03 * 1.0) ** 2
    # #
    # #         # SSIM map
    # #         ssim_map = ((2 * x_mean * y_mean + C1) * (2 * cov + C2)) / \
    # #                    ((x_mean ** 2 + y_mean ** 2 + C1) * (x_var + y_var + C2))
    # #         return ssim_map.mean()  # Scalar
    # #
    # #     structural_sim = calculate_ssim_1d(a_flat, b_flat)
    # #
    # #     # --- Frequency Agreement ---
    # #     fft_a = torch.fft.rfft(a_flat)
    # #     fft_b = torch.fft.rfft(b_flat)
    # #     freq_agreement = F.cosine_similarity(fft_a.abs().unsqueeze(0),
    # #                                          fft_b.abs().unsqueeze(0),
    # #                                          dim=1)
    # #
    # #     # Combine and reshape
    # #     agreement = (
    # #             cosine_weight * cosine_sim +
    # #             structural_weight * structural_sim +
    # #             frequency_weight * freq_agreement
    # #     )
    # #
    # #     print(f"\n=== Agreement Mask (advanced_agreement) ===")
    # #     print(f"Input Shapes: {a.shape} vs {b.shape}")
    # #     print(
    # #         f"Agreement Stats: min={agreement.min().item():.3f}, max={agreement.max().item():.3f}, mean={agreement.mean().item():.3f}")
    # #     print(f"NaN/Inf Check: {torch.isnan(agreement).any()} NaN, {torch.isinf(agreement).any()} Inf")
    # #
    # #     return torch.clamp(agreement, 0, 1)
    # #
    # # def calculate_attention_agreement(
    # #         a: Tensor,
    # #         b: Tensor,
    # #         temperature: float = 1.0,
    # #         pattern_weight: float = 0.7,
    # #         value_weight: float = 0.3
    # # ) -> Tensor:
    # #     """
    # #     Calculate agreement specifically for attention mechanisms using delta tensors.
    # #     Takes into account attention pattern similarity and head relationships.
    # #
    # #     Args:
    # #         a: Delta from base model for first model (a - c)
    # #         b: Delta from base model for second model (b - c)
    # #         temperature: Softmax temperature for attention pattern comparison
    # #
    # #     Returns:
    # #         Tensor: Agreement scores
    # #     """
    # #
    # #     # Calculate attention patterns from the deltas
    # #     def get_attention_pattern(x: Tensor) -> Tensor:
    # #         if x.numel() == 0:
    # #             return x
    # #
    # #         # Handle cases with fewer than 2 dimensions
    # #         if x.dim() < 2:
    # #             return torch.ones_like(x)
    # #
    # #         # Calculate attention scores from the delta directly
    # #         attn_pattern = torch.matmul(x, x.transpose(-2, -1)) / temperature
    # #
    # #         return F.softmax(attn_pattern, dim=-1)
    # #
    # #     # Get patterns for each delta
    # #     pattern_a = get_attention_pattern(a)
    # #     pattern_b = get_attention_pattern(b)
    # #
    # #     # Calculate pattern similarity directly between deltas
    # #     pattern_agreement = F.cosine_similarity(
    # #         pattern_a,
    # #         pattern_b,
    # #         dim=-1
    # #     )
    # #
    # #     # Calculate value space agreement directly between deltas
    # #     value_agreement = F.cosine_similarity(
    # #         a,
    # #         b,
    # #         dim=-1
    # #     )
    # #
    # #     # Combine agreements with emphasis on pattern agreement
    # #     combined_agreement = pattern_weight * pattern_agreement + value_weight * value_agreement
    # #
    # #     agreement = torch.clamp(combined_agreement, 0, 1).unsqueeze(-1)
    # #
    # #     print(
    # #         f"pattern_agreement Stats: min={pattern_agreement.min().item():.3f}, max={pattern_agreement.max().item():.3f}, mean={pattern_agreement.mean().item():.3f}")
    # #     print(
    # #         f"value_agreement Stats: min={value_agreement.min().item():.3f}, max={value_agreement.max().item():.3f}, mean={value_agreement.mean().item():.3f}")
    # #     print(
    # #         f"Combined Agreement Stats: min={combined_agreement.min().item():.3f}, max={combined_agreement.max().item():.3f}, mean={combined_agreement.mean().item():.3f}")
    # #     print(f"NaN/Inf Check: {torch.isnan(agreement).any()} NaN, {torch.isinf(agreement).any()} Inf")
    # #
    # #     return agreement
    # #
    # # def calculate_spatial_agreement(
    # #         a: Tensor,
    # #         b: Tensor,
    # #         kernel_size: int = 3,
    # #         local_weight: float = 0.6,
    # #         channel_weight: float = 0.4
    # # ) -> Tensor:
    # #     """Fixed spatial agreement calculation with dimension-aware processing"""
    # #
    # #     def get_local_features(x: Tensor) -> Tensor:
    # #         # Handle different dimensionalities
    # #         if x.dim() == 1:  # Bias vectors [C]
    # #             return x.unsqueeze(0)  # [1, C]
    # #         elif x.dim() == 2:  # Embedding layers [V, D]
    # #             x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, V, D]
    # #
    # #         # For all cases, pad before unfolding
    # #         padding = kernel_size // 2
    # #         x_padded = F.pad(x, (padding, padding, padding, padding))
    # #
    # #         # Unfold based on the original tensor's dimensions
    # #         if x.dim() == 3:
    # #             return F.unfold(x_padded, (1, kernel_size))  # [1, C*ks, L]
    # #         elif x.dim() == 4:
    # #             return F.unfold(x_padded, kernel_size)  # [B, C*ks*ks, L]
    # #
    # #         return x  # Fallback for other cases
    # #
    # #     # Get features and maintain original batch/channel dims
    # #     local_a = get_local_features(a)
    # #     local_b = get_local_features(b)
    # #
    # #     # Calculate spatial agreement per patch
    # #     spatial_agreement = F.cosine_similarity(local_a, local_b, dim=1)
    # #
    # #     # Calculate channel agreement
    # #     channel_agreement = F.cosine_similarity(a, b, dim=-1 if a.dim() > 1 else 0)
    # #
    # #     # Handle channel agreement shape mismatch
    # #     if channel_agreement.numel() == 1:
    # #         # Scalar case: broadcast directly
    # #         channel_agreement = channel_agreement.expand_as(spatial_agreement)
    # #     else:
    # #         # Get target spatial dimensions [B, *SPATIAL]
    # #         target_shape = spatial_agreement.shape
    # #
    # #         # Reshape channel agreement to match spatial dimensions via interpolation
    # #         if channel_agreement.dim() == 1:
    # #             # Conv1D/Linear: [C] -> [B, L] via unsqueeze + expand
    # #             channel_agreement = channel_agreement.unsqueeze(0).expand(target_shape)
    # #         elif channel_agreement.dim() == 2:
    # #             # Conv2D: [B, C] -> [B, L] via interpolation
    # #             channel_agreement = F.interpolate(
    # #                 channel_agreement.unsqueeze(1).float(),  # [B, 1, C]
    # #                 size=target_shape[1],
    # #                 mode='nearest'
    # #             ).squeeze(1).to(channel_agreement.dtype)
    # #         elif channel_agreement.dim() == 3:
    # #             # 3D case: [B, H, W] -> [B, L] via adaptive pooling
    # #             channel_agreement = F.adaptive_avg_pool1d(
    # #                 channel_agreement.flatten(1).float(),  # [B, H*W]
    # #                 output_size=target_shape[1]
    # #             ).to(channel_agreement.dtype)
    # #
    # #     combined = (local_weight * spatial_agreement + channel_weight * channel_agreement).clamp(0, 1)
    # #
    # #     print(
    # #         f"Spatial Agreement Stats: min={spatial_agreement.min().item():.3f}, max={spatial_agreement.max().item():.3f}, mean={spatial_agreement.mean().item():.3f}")
    # #     print(
    # #         f"Channel Agreement Stats: min={channel_agreement.min().item():.3f}, max={channel_agreement.max().item():.3f}, mean={channel_agreement.mean().item():.3f}")
    # #     print(
    # #         f"Combined Agreement Stats: min={combined.min().item():.3f}, max={combined.max().item():.3f}, mean={combined.mean().item():.3f}")
    # #     print(f"NaN/Inf Check: {torch.isnan(combined).any()} NaN, {torch.isinf(combined).any()} Inf")
    # #
    # #     return combined.mean()
    #
    # @staticmethod
    # @merge_method
    # def laplacian_difference(
    #         a: Parameter(Tensor, "weight"),
    #         b: Parameter(Tensor, "weight"),
    #         *,
    #         alpha: Parameter(Tensor) =0.5,
    #         n_levels: int = 4,
    #         **kwargs,
    # ) -> Return(Tensor):
    #     if a.numel() == 0:
    #         return MergeMethods.geometric_sum_full.__wrapped__(a, b, alpha=alpha)
    #
    #     def gaussian_downsample(x: Tensor) -> Tensor:
    #         """Downsampling using a learnable Conv2d layer"""
    #         if x.numel() == 0 or x.ndim < 3:
    #             return x
    #
    #         # Determine the number of channels (C) based on the tensor's shape
    #         if x.ndim == 4:  # [B, C, H, W]
    #             B, C, H, W = x.shape
    #         elif x.ndim == 3:  # [B, C, L]
    #             B, C, L = x.shape
    #         else:  # [N]
    #             return x  # Cannot downsample
    #
    #         # Create a Conv2d or Conv1d layer for downsampling
    #         if x.ndim == 4:  # [B, C, H, W]
    #             # Ensure the spatial dimensions are even for downsampling
    #             if x.shape[2] % 2 != 0:
    #                 x = F.pad(x, (0, 0, 0, 1), mode='replicate')
    #             if x.shape[3] % 2 != 0:
    #                 x = F.pad(x, (0, 1, 0, 0), mode='replicate')
    #
    #             conv = torch.nn.Conv2d(
    #                 in_channels=C,
    #                 out_channels=C,
    #                 kernel_size=3,
    #                 stride=2,
    #                 padding=1,
    #                 groups=C,
    #                 bias=False,
    #                 dtype=x.dtype
    #             ).to(x.device)
    #         else:  # [B, C, L] or [N]
    #             # Ensure the length is even for downsampling
    #             if x.ndim == 3 and x.shape[2] % 2 != 0:
    #                 x = F.pad(x, (0, 1), mode='replicate')
    #
    #             conv = torch.nn.Conv1d(
    #                 in_channels=C,
    #                 out_channels=C,
    #                 kernel_size=3,
    #                 stride=2,
    #                 padding=1,
    #                 groups=C,
    #                 bias=False,
    #                 dtype=x.dtype
    #             ).to(x.device)
    #
    #         # Initialize weights with a small value
    #         with torch.no_grad():
    #             conv.weight.data.normal_(0, 0.01)
    #
    #         return conv(x)
    #
    #     def gaussian_upsample(x: Tensor, target_shape: tuple) -> Tensor:
    #         """Upsampling using a learnable ConvTranspose2d layer"""
    #         if x.numel() == 0:
    #             return x
    #
    #         # Determine the number of channels (C) based on the tensor's shape
    #         if x.ndim == 4:  # [B, C, H, W]
    #             B, C, H, W = x.shape
    #         elif x.ndim == 3:  # [B, C, L]
    #             B, C, L = x.shape
    #         else:
    #             return x
    #
    #         # Create a ConvTranspose2d or ConvTranspose1d layer for upsampling
    #         if x.ndim == 4:  # [B, C, H, W]
    #             conv_transpose = torch.nn.ConvTranspose2d(
    #                 in_channels=C,
    #                 out_channels=C,
    #                 kernel_size=3,
    #                 stride=2,
    #                 padding=1,
    #                 output_padding=1,
    #                 groups=C,
    #                 bias=False,
    #                 dtype=x.dtype
    #             ).to(x.device)
    #         else:  # [B, C, L] or [N]
    #             conv_transpose = torch.nn.ConvTranspose1d(
    #                 in_channels=C,
    #                 out_channels=C,
    #                 kernel_size=3,
    #                 stride=2,
    #                 padding=1,
    #                 output_padding=1,
    #                 groups=C,
    #                 bias=False,
    #                 dtype=x.dtype
    #             ).to(x.device)
    #
    #         # Initialize weights with a small value
    #         with torch.no_grad():
    #             conv_transpose.weight.data.normal_(0, 0.01)
    #
    #         output = conv_transpose(x)
    #
    #         # crop to target shape
    #         if output.shape[-2:] != target_shape[-2:]:
    #             output = output[..., :target_shape[-2], :target_shape[-1]]
    #
    #         return output
    #
    #     def build_pyramid(x: Tensor) -> List[Tensor]:
    #         """Pyramid construction with robust spatial checks"""
    #         gaussian = [x]
    #         for _ in range(int(n_levels) - 1):
    #             next_level = gaussian_downsample(gaussian[-1])
    #
    #             # Check for invalid downsampling
    #             stop_condition = (
    #                     next_level.numel() == 0
    #                     or next_level.ndim == 0
    #                     or next_level.shape == gaussian[-1].shape
    #             )
    #
    #             if stop_condition:
    #                 break
    #             gaussian.append(next_level)
    #
    #         # Fill remaining levels with copies
    #         while len(gaussian) < n_levels:
    #             gaussian.append(gaussian[-1].clone())
    #
    #         return gaussian
    #
    #     # print(f"Merging with alpha: {alpha}")
    #
    #     # Build Laplacian pyramids
    #     a_pyr = build_pyramid(a)
    #     b_pyr = build_pyramid(b)
    #
    #     # Ensure the pyramids have the same number of levels
    #     min_levels = min(len(a_pyr), len(b_pyr))
    #     a_pyr = a_pyr[:min_levels]
    #     b_pyr = b_pyr[:min_levels]
    #
    #     # Merge pyramids
    #     merged_pyr = []
    #     for i in range(len(a_pyr)):
    #     #    print(f"  Level {i}: a_pyr shape: {a_pyr[i].shape}, b_pyr shape: {b_pyr[i].shape}")
    #         merged = a_pyr[i] * (1 - alpha) + b_pyr[i] * alpha
    #     #    print(
    #     #        f"  Merged level {i} shape: {merged.shape}, stats: min={merged.min().item():.4f}, max={merged.max().item():.4f}, mean={merged.mean().item():.4f}")
    #         merged_pyr.append(merged)
    #
    #     # Reconstruct the merged tensor from the pyramid
    #     merged = merged_pyr[-1]
    #     for level in range(len(merged_pyr) - 2, -1, -1):
    #     #    print(f"  Upsampling level {level} from shape: {merged.shape} to {merged_pyr[level].shape}")
    #         upsampled = gaussian_upsample(merged, merged_pyr[level].shape)
    #     #    print(f"  Upsampled shape: {upsampled.shape}")
    #
    #         # Scale the upsampled tensor before adding it to the next level
    #         scale_factor = 0.5  # You might want to make this a parameter or learn it adaptively
    #         merged = upsampled * scale_factor + merged_pyr[level]
    #
    #     return merged
    #
    # # class LayerType2(Enum):
    # #     ATTENTION_QKV = auto()
    # #     ATTENTION_OUTPUT = auto()
    # #     CONV_3X3 = auto()
    # #     CONV_1X1 = auto()
    # #     LINEAR = auto()
    # #     NORM = auto()
    # #     EMBEDDING = auto()
    # #     TIME_EMBEDDING = auto()
    # #     SCALAR = auto()
    # #
    # # @dataclass
    # # class LayerInfo:
    # #     type: 'MergeMethods.LayerType2'
    # #     shape: Tuple[int, ...]
    # #     head_dim: Optional[int] = None
    # #     num_heads: Optional[int] = None
    # #
    # # def identify_layer(key: str, tensor: Tensor) -> LayerInfo:
    # #     """Identify layer type with architecture-aware head calculations"""
    # #     shape = tuple(tensor.shape)
    # #
    # #     # 1. Scalar parameters first
    # #     if not shape or "logit_scale" in key:
    # #         return MergeMethods.LayerInfo(MergeMethods.LayerType2.SCALAR, (1,))
    # #
    # #     # 2. Biases
    # #     if key.endswith(".bias") or "bias" in key:
    # #         return MergeMethods.LayerInfo(MergeMethods.LayerType2.LINEAR, shape)
    # #
    # #     # 3. Normalization layers
    # #     if any(x in key for x in [".norm", "layer_norm", "ln_final", "ln_1", "ln_2", "layer_norm1", "layer_norm2",
    # #                               "final_layer_norm", "norm"]):
    # #         return MergeMethods.LayerInfo(MergeMethods.LayerType2.NORM, shape)
    # #
    # #     # 4. Embeddings
    # #     if "token_embedding" in key or "shared.weight" in key or "embed" in key:
    # #         return MergeMethods.LayerInfo(MergeMethods.LayerType2.EMBEDDING, shape)
    # #
    # #     # 5. Positional Embeddings
    # #     if "positional_embedding" in key:
    # #         return MergeMethods.LayerInfo(MergeMethods.LayerType2.LINEAR, shape)
    # #
    # #     # 6. Attention layer detection
    # #     is_attention = any(x in key for x in
    # #                        [".attn.", ".to_q.", ".to_k.", ".to_v.", "q_proj", "k_proj", "v_proj", "in_proj_"])
    # #     is_unet = "model.diffusion_model" in key
    # #     is_clip_te = "conditioner.embedders.0.transformer" in key
    # #     is_clip_te2 = "conditioner.embedders.1.model" in key
    # #
    # #     if is_attention:
    # #         num_heads, head_dim = None, None
    # #
    # #         if is_unet and ("input_blocks" in key or "output_blocks" in key):
    # #             num_heads = 8
    # #             head_dim = shape[-1] // num_heads if len(shape) >= 2 else 64
    # #         elif is_clip_te:
    # #             num_heads = 12
    # #             head_dim = shape[-1] // num_heads if len(shape) >= 2 else 64
    # #         elif is_clip_te2:
    # #             num_heads = 20
    # #             head_dim = shape[-1] // num_heads if len(shape) >= 2 else 64
    # #
    # #         # Fallback calculation
    # #         if num_heads is None and len(shape) >= 2:
    # #             head_dim = 64
    # #             num_heads = max(1, shape[-1] // head_dim)
    # #
    # #         return MergeMethods.LayerInfo(
    # #             MergeMethods.LayerType2.ATTENTION_QKV if "out_proj" not in key else MergeMethods.LayerType2.ATTENTION_OUTPUT,
    # #             shape,
    # #             head_dim,
    # #             num_heads
    # #         )
    # #
    # #     # 7. Convolution detection
    # #     if len(shape) == 4:
    # #         return MergeMethods.LayerInfo(
    # #             MergeMethods.LayerType2.CONV_1X1 if shape[-1] == 1
    # #             else MergeMethods.LayerType2.CONV_3X3,
    # #             shape
    # #         )
    # #
    # #     # 8. Time embeddings
    # #     if "time_embed" in key or "time_embedding" in key:
    # #         return MergeMethods.LayerInfo(MergeMethods.LayerType2.TIME_EMBEDDING, shape)
    # #
    # #     # 9. Linear/FFN layers
    # #     if any(x in key for x in ["fc1", "fc2", "c_fc", "c_proj", "proj", "mlp", "ff.net"]):
    # #         return MergeMethods.LayerInfo(MergeMethods.LayerType2.LINEAR, shape)
    # #
    # #     # Default to linear
    # #     return MergeMethods.LayerInfo(MergeMethods.LayerType2.LINEAR, shape)

    @merge_method
    def hswb_merge(
            *deltas: Parameter(Tensor, "delta"),
            hessian_curvature_threshold: Parameter(float) = 0.1,
            parallel_reinforcement: Parameter(float) = 1.0,
            orthogonal_contribution: Parameter(float) = 1.0,
            num_projections: Parameter(int) = 128,
            **kwargs,
    ) -> Return(Tensor, "delta"):
        """The final H-SWB Merge with landscape reconstruction and proper importance weighting."""
        if not deltas:
            raise ValueError("This function received no deltas.")

        key = kwargs["key"]
        core_indices = [1, 3, 4, 6, 7, 8, 9]  # Fixed: removed EPS model (index 10)

        # NO FILTERING - work with original deltas
        cleaned_deltas = list(deltas)

        num_deltas = len(cleaned_deltas)
        valid_core_indices = [i for i in core_indices if i < num_deltas]
        if not valid_core_indices:
            valid_core_indices = list(range(num_deltas))

        print(f"=== DELTA INDEX MAPPING ===")
        print(f"Total deltas: {num_deltas}")
        print(f"Core indices: {valid_core_indices}")
        outlier_indices = [i for i in range(num_deltas) if i not in valid_core_indices]
        print(f"Outlier indices: {outlier_indices}")

        core_deltas = [cleaned_deltas[i] for i in valid_core_indices]
        outlier_deltas = [d for i, d in enumerate(cleaned_deltas) if i not in valid_core_indices]

        print("\n=== OUTLIER MODELS ===")
        for i, idx in enumerate(outlier_indices):
            print(f"OUTLIER_{i} = Delta index {idx}")

        print(f"\nExpected shadowforge at index 10, got outlier indices: {outlier_indices}")

        if not core_deltas:
            return torch.mean(torch.stack(cleaned_deltas), dim=0)

        delta_core, _ = torch.median(torch.stack(core_deltas), dim=0)

        # Process outliers into parallel/perpendicular components
        parallel_components = []
        perpendicular_components = []

        MergeMethods.track_tensor_quality(delta_core, "DELTA_CORE", key)

        for i, outlier in enumerate(outlier_deltas):
            MergeMethods.track_tensor_quality(outlier, f"OUTLIER_{i}", key)

            parallel = MergeMethods._hswb_projection(outlier, delta_core)
            MergeMethods.track_tensor_quality(parallel, f"PARALLEL_{i}", key)

            if torch.isnan(parallel).any():
                print(f"NaN detected in parallel projection, using zero instead")
                parallel = torch.zeros_like(outlier)

            perpendicular = outlier - parallel
            MergeMethods.track_tensor_quality(perpendicular, f"PERPENDICULAR_{i}", key)

            if torch.isnan(perpendicular).any():
                print(f"NaN detected in perpendicular, using original outlier")
                perpendicular = outlier
                parallel = torch.zeros_like(outlier)

            parallel_components.append(parallel)
            perpendicular_components.append(perpendicular)

        # Merge components
        if parallel_components:
            merged_parallel = torch.mean(torch.stack(parallel_components), dim=0)
        else:
            merged_parallel = torch.zeros_like(delta_core)

        if not perpendicular_components:
            merged_perpendicular = torch.zeros_like(delta_core)
        else:
            merged_perpendicular = MergeMethods._hswb_swd_barycenter(
                tensors=perpendicular_components,
                reference_tensor=delta_core,
                num_projections=num_projections,
            )

        # LANDSCAPE RECONSTRUCTION: Get importance weights from Hessian approximation
        importance_weights = MergeMethods._hswb_reconstruct_hessian_diag(list(deltas))

        # Apply threshold to importance weights
        importance_mask = (importance_weights >= hessian_curvature_threshold).to(dtype=delta_core.dtype)
        final_importance = importance_weights * importance_mask

        # Add tracking after merging
        MergeMethods.track_tensor_quality(merged_parallel, "MERGED_PARALLEL", key)
        MergeMethods.track_tensor_quality(merged_perpendicular, "MERGED_PERPENDICULAR", key)
        MergeMethods.track_tensor_quality(final_importance, "FINAL_IMPORTANCE", key)

        # Print norm statistics
        core_norm = torch.norm(delta_core)
        parallel_norm = torch.norm(merged_parallel) if parallel_components else 0
        perp_norm = torch.norm(merged_perpendicular) if perpendicular_components else 0
        importance_norm = torch.norm(final_importance)

        print(f"Core norm: {core_norm:.6f}")
        print(f"Parallel norm: {parallel_norm:.6f}")
        print(f"Perpendicular norm: {perp_norm:.6f}")
        print(f"Importance norm: {importance_norm:.6f}")
        print(f"Parallel ratio: {parallel_norm / core_norm:.3f}")
        print(f"Perpendicular ratio: {perp_norm / core_norm:.3f}")
        print(f"Importance ratio: {importance_norm / core_norm:.3f}")

        # FINAL COMBINATION: Use importance weights for final combination
        final_delta = delta_core + final_importance * (orthogonal_contribution * merged_perpendicular)

        # Add final tracking
        MergeMethods.track_tensor_quality(final_delta, "FINAL_DELTA", key)

        return final_delta

    def _hswb_reconstruct_hessian_diag(deltas: List[Tensor]) -> Tensor:
        """
        Reconstructs Hessian diagonal by finding the quadratic bowl that best
        explains the geometric arrangement of the input deltas.
        """
        if len(deltas) < 2:
            return torch.ones_like(deltas[0])

        device = deltas[0].device
        num_params = deltas[0].numel()
        num_models = len(deltas)

        # Initialize: Hessian diagonal + individual energy levels per model
        hessian_diag_candidate = torch.ones(num_params, device=device, requires_grad=True)
        energy_levels = torch.ones(num_models, device=device, requires_grad=True)

        optimizer = torch.optim.LBFGS([hessian_diag_candidate, energy_levels],
                                      max_iter=20, history_size=10)

        # Pre-compute squared deltas for efficiency
        deltas_sq = torch.stack([d.flatten() ** 2 for d in deltas])

        def closure():
            optimizer.zero_grad()

            # Ensure positive values (curvature can't be negative)
            positive_hessian = torch.nn.functional.softplus(hessian_diag_candidate)
            positive_energies = torch.nn.functional.softplus(energy_levels)

            # Calculate predicted energy for each model: E = sum(H * d^2)
            predicted_energies = torch.sum(positive_hessian * deltas_sq, dim=1)

            # Loss: How well does our Hessian explain the model positions?
            energy_errors = predicted_energies - positive_energies

            # Use robust loss (less sensitive to outlier models)
            loss = torch.mean(torch.abs(energy_errors))  # L1 instead of L2

            loss.backward()
            return loss

        optimizer.step(closure)

        # Return normalized importance weights
        final_hessian = torch.nn.functional.softplus(hessian_diag_candidate).detach()
        max_importance = torch.max(final_hessian)
        if max_importance > 0:
            importance_weights = final_hessian / max_importance
        else:
            importance_weights = torch.ones_like(final_hessian)

        return importance_weights.reshape(deltas[0].shape)

    # --- Helper 2: Vector Projection ---
    def _hswb_projection(vector_to_project: Tensor, target_vector: Tensor) -> Tensor:
        """Calculates the projection of one vector onto another."""
        target_norm_sq = torch.sum(target_vector ** 2)
        if target_norm_sq < 1e-12:
            return torch.zeros_like(vector_to_project)
        dot_product = torch.sum(vector_to_project * target_vector)
        return target_vector * (dot_product / target_norm_sq)

    # --- Helper 3: The Reconstruction Engine (The Real, Iterative Version) ---
    def _hswb_reconstruct_from_projections(
            target_projections: Tensor,
            projection_dirs: Tensor,
            initial_guess: Tensor
    ) -> Tensor:
        """
        Reconstructs a high-dimensional tensor from its target 1D projections using LBFGS optimization.
        This is the "sculpting" process that finds the true optimal barycenter.
        """
        # The tensor we are optimizing, our "block of clay". It needs requires_grad=True.
        candidate = initial_guess.clone().requires_grad_(True)
        # The LBFGS optimizer is very effective for this kind of problem.
        optimizer = torch.optim.LBFGS([candidate], max_iter=20, history_size=10, line_search_fn="strong_wolfe")

        # Flatten the projection directions for efficient matrix multiplication.
        projection_dirs_flat = projection_dirs.view(projection_dirs.shape[0], -1)

        # The closure is a function that the optimizer calls repeatedly.
        def closure():
            optimizer.zero_grad()
            # Project our current guess to see what its shadows look like.
            current_projections = candidate.flatten() @ projection_dirs_flat.T

            # We compare the distribution of our current shadows to the target shadows.
            # Sorting is the key to comparing distributions in 1D.
            current_sorted, _ = torch.sort(current_projections)
            target_sorted, _ = torch.sort(target_projections)

            # The loss is the Mean Squared Error between the perfect shadows and our current ones.
            loss = torch.mean((current_sorted - target_sorted) ** 2)
            loss.backward()
            return loss

        # This is the magic line. It runs the optimization loop.
        optimizer.step(closure)

        # Return the final, sculpted statue, detached from the computation graph.
        return candidate.detach()

    # --- Helper 4: Sliced-Wasserstein Barycenter (Using the Real Engine) ---
    def _hswb_swd_barycenter(
            tensors: List[Tensor],
            reference_tensor: Tensor,
            num_projections: int = 128,  # Reduced from 128 to prevent memory issues
            max_iter: int = 20  # Iterations for the barycenter refinement
    ) -> Tensor:
        """Computes the proper SWB using iterative reconstruction."""
        MergeMethods.debug_memory_usage("SWB_START")
        num_tensors = len(tensors)
        if num_tensors == 0:
            return torch.zeros_like(reference_tensor)
        if num_tensors == 1:
            return tensors[0]

        device = tensors[0].device
        shape = tensors[0].shape

        # Start with a simple average as our initial guess.
        MergeMethods.debug_memory_usage("BEFORE_INITIAL_STACK")
        barycenter_guess = torch.mean(torch.stack(tensors), dim=0)  # SUSPECT #3
        MergeMethods.debug_memory_usage("AFTER_INITIAL_STACK")

        # Clear memory after initial stack operation
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        for iteration in range(max_iter):
            MergeMethods.debug_memory_usage(f"ITER_{iteration}_START")

            # Process projections in smaller batches to reduce memory pressure
            batch_size = 16  # Process projections in smaller batches
            batch_results = []  # Store reconstruction results per batch (FIXED: was all_projected)

            for batch_start in range(0, num_projections, batch_size):
                batch_end = min(batch_start + batch_size, num_projections)
                batch_size_actual = batch_end - batch_start

                # Create projection directions for this batch only
                projection_dirs = torch.randn(batch_size_actual, barycenter_guess.numel(), device=device)
                projection_dirs /= torch.linalg.norm(projection_dirs, dim=1, keepdim=True)

                # Project all input style vectors onto the random directions.
                MergeMethods.debug_memory_usage(f"ITER_{iteration}_BATCH_{batch_start // batch_size}_BEFORE_PROJECTION")

                # Project tensors in batch
                batch_projected = []
                for tensor in tensors:
                    proj = tensor.flatten() @ projection_dirs.T
                    batch_projected.append(proj)

                batch_tensor = torch.stack(batch_projected)

                # FIXED: Process this batch immediately instead of accumulating
                sorted_batch, _ = torch.sort(batch_tensor, dim=0)
                target_batch_1d = torch.mean(sorted_batch, dim=0)

                MergeMethods.debug_memory_usage(f"ITER_{iteration}_BATCH_{batch_start // batch_size}_AFTER_PROJECTION")

                # FIXED: Apply reconstruction TO THIS BATCH ONLY
                MergeMethods.debug_memory_usage(
                    f"ITER_{iteration}_BATCH_{batch_start // batch_size}_BEFORE_RECONSTRUCTION")
                batch_reconstruction = MergeMethods._hswb_reconstruct_from_projections(
                    target_batch_1d,  # Small batch target
                    projection_dirs,  # Small batch projection directions
                    barycenter_guess  # This is the only large tensor
                )
                MergeMethods.debug_memory_usage(
                    f"ITER_{iteration}_BATCH_{batch_start // batch_size}_AFTER_RECONSTRUCTION")

                batch_results.append(batch_reconstruction)

                # Explicit cleanup after each batch
                del projection_dirs, batch_projected, batch_tensor, sorted_batch, target_batch_1d
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            # FIXED: Combine batch reconstruction results (not the raw projections!)
            MergeMethods.debug_memory_usage(f"ITER_{iteration}_BEFORE_COMBINE_BATCH_RESULTS")
            if batch_results:
                # Average the reconstruction results from all batches
                new_barycenter = torch.mean(torch.stack(batch_results), dim=0)
            else:
                new_barycenter = barycenter_guess
            MergeMethods.debug_memory_usage(f"ITER_{iteration}_AFTER_COMBINE_BATCH_RESULTS")

            # Cleanup batch results
            del batch_results
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            # Check convergence
            MergeMethods.debug_memory_usage(f"ITER_{iteration}_BEFORE_CONVERGENCE_CHECK")
            if torch.norm(new_barycenter - barycenter_guess) < 1e-5:
                barycenter_guess = new_barycenter
                break

            barycenter_guess = new_barycenter

            # Cleanup after each iteration
            MergeMethods.debug_memory_usage(f"ITER_{iteration}_END_CLEANUP")
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        MergeMethods.debug_memory_usage("SWB_END")
        return barycenter_guess.reshape(shape)

    def debug_memory_usage(label):
        pass

        # if torch.cuda.is_available():
        #     allocated = torch.cuda.memory_allocated() / 1024**3
        #     reserved = torch.cuda.memory_reserved() / 1024**3
        #     print(f"{label}: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        # else:
        #     import psutil
        #     memory = psutil.virtual_memory().used / 1024**3
        #     print(f"{label}: System RAM: {memory:.2f}GB")

    def track_tensor_quality(tensor, label, key=None):
        """Track tensor quality metrics"""
        norm = torch.norm(tensor)
        mean_val = torch.mean(tensor)
        std_val = torch.std(tensor)
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        has_nan = torch.isnan(tensor).any()
        has_inf = torch.isinf(tensor).any()

        key_info = f"[{key}] " if key else ""
        print(
            f"{key_info}{label}: norm={norm:.6f}, mean={mean_val:.6f}, std={std_val:.6f}, min={min_val:.6f}, max={max_val:.6f}, nan={has_nan}, inf={has_inf}")
