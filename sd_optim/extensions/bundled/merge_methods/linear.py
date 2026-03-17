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
