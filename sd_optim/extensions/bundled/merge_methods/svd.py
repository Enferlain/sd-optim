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

def orthogonal_procrustes(a, b, cancel_reflection: bool = False):
    if a.shape != b.shape:
        raise ValueError(f"a {tuple(a.shape)} and b {tuple(b.shape)} must have the same shape")

    if not cancel_reflection and a.shape[0] + 10 < a.shape[1]:
        svd_driver = "gesvdj" if a.is_cuda else None
        u, _, v_t = torch_svd_lowrank(a.T @ b, q=a.shape[0] + 10, driver=svd_driver, full_matrices=cancel_reflection)
    else:
        svd_driver = "gesvd" if a.is_cuda else None
        u, _, v_t = torch.linalg.svd(a.T @ b, driver=svd_driver)
        if cancel_reflection:
            u[:, -1] /= torch.slogdet(u)[0] * torch.slogdet(v_t)[0]

    transform = u @ v_t
    return transform


def fractional_matrix_power(matrix: Tensor, power: float, cache: dict[str, Tensor] | None = None):
    if cache is not None and "eigenvalues" in cache:
        complex_dtype = torch_complex_dtype_map[matrix.dtype]
        eigenvalues = cache["eigenvalues"].to(matrix.device, complex_dtype)
        eigenvectors = cache["eigenvectors"].to(matrix.device, complex_dtype)
        eigenvectors_inv = cache["eigenvectors_inv"].to(matrix.device, complex_dtype)
    else:
        eigenvalues, eigenvectors = torch.linalg.eig(matrix)
        eigenvectors_inv = torch.linalg.inv(eigenvectors)
        if cache is not None:
            cache["eigenvalues"] = eigenvalues.to("cpu", torch.complex32)
            cache["eigenvectors"] = eigenvectors.to("cpu", torch.complex32)
            cache["eigenvectors_inv"] = eigenvectors_inv.to("cpu", torch.complex32)

    eigenvalues.pow_(power)
    result = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors_inv
    return result.real.to(dtype=matrix.dtype)


torch_complex_dtype_map = {
    torch.bfloat16: torch.complex32,
    torch.float16: torch.complex32,
    torch.float32: torch.complex64,
    torch.float64: torch.complex128,
}


# need to redefine torch.svd_lowrank to specify the svd driver and for full_matrices support
def torch_svd_lowrank(
    A: Tensor,
    q: int | None = 6,
    niter: int | None = 2,
    driver: str | None = None,
    full_matrices: bool | None = True,
) -> tuple[Tensor, Tensor, Tensor]:
    q = 6 if q is None else q
    m, n = A.shape[-2:]
    A_t = A.T

    # Algorithm 5.1 in Halko et al 2009, slightly modified to reduce
    # the number conjugate and transpose operations
    assert q <= min(m, n), f"Rank approximation q={q} should be <= min(m={m}, n={n})"
    # computing the SVD approximation of a transpose in
    # order to keep B shape minimal (the m < n case) or the V
    # shape small (the n > q case)
    Q = get_approximate_basis(A_t, q, niter=niter)
    Q_c = Q.conj()
    B_t = A @ Q_c
    assert B_t.shape[-2] == m, (B_t.shape, m)
    assert B_t.shape[-1] == q, (B_t.shape, q)
    assert B_t.shape[-1] <= B_t.shape[-2], B_t.shape
    U, S, Vh = torch.linalg.svd(B_t, driver=driver, full_matrices=full_matrices)
    V = Q @ Vh.mH
    if full_matrices:
        V = orthogonal_extend(V)

    return U, S, V.mH


def get_approximate_basis(A: Tensor, q: int, niter: int | None = 2) -> Tensor:
    niter = 2 if niter is None else niter
    m, n = A.shape[-2:]
    dtype = get_floating_dtype(A)

    R = torch.randn(n, q, dtype=dtype, device=A.device)

    A_H = A.mH
    Q = torch.linalg.qr(A @ R).Q
    for i in range(niter):
        Q = torch.linalg.qr(A_H @ Q).Q
        Q = torch.linalg.qr(A @ Q).Q

    return Q


def orthogonal_extend(A: torch.Tensor) -> torch.Tensor:
    m, n = A.shape
    if m <= n:
        return A

    proj = torch.eye(m, device=A.device, dtype=A.dtype) - A @ A.mH
    proj @= torch.randn(m, m - n, device=A.device, dtype=A.dtype)
    A_extension = torch.linalg.householder_product(*torch.linalg.qr(proj, mode="raw"))
    return torch.cat((A, A_extension), dim=1)


def get_floating_dtype(A):
    dtype = A.dtype
    if dtype.is_floating_point:
        return dtype
    return torch.float32
