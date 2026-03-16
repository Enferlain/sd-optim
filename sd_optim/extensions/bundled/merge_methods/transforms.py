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
#             vs.append(wavelet_merge.__wrapped__(k_a, k_b, **k_kwargs))
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