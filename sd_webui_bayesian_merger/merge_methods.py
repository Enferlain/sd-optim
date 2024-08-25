import sd_mecha
import functools
import operator
import torch

from torch import Tensor
from typing import Tuple, TypeVar, Dict, Optional

from sd_mecha.hypers import Hyper
from sd_mecha.merge_space import MergeSpace
from sd_mecha.extensions.merge_method import LiftFlag, convert_to_recipe

EPSILON = 1e-10
SameMergeSpace = TypeVar("SameMergeSpace", bound=LiftFlag[MergeSpace.BASE | MergeSpace.DELTA])


class MergeMethods:
    @staticmethod
    def weighted_sum(a, b, alpha: Hyper):
        return sd_mecha.weighted_sum(a, b, alpha=alpha)

    @staticmethod
    def rotate(a, b, alignment: Hyper = 1.0, alpha: Hyper = 0.0, cache=None):  # Use None as default
        if cache is None:
            cache = {}  # Create a new dictionary if cache is None

        return sd_mecha.rotate(a, b, alignment=alignment, alpha=alpha, cache=cache)

    @staticmethod
    def slerp(a, b, alpha: Hyper):
        return sd_mecha.slerp(a, b, alpha=alpha)

    @staticmethod
    def geometric_sum(a, b, alpha: Hyper):
        return sd_mecha.geometric_sum(a, b, alpha=alpha)

    @staticmethod
    def add_cosine_a(a, b, alpha: Hyper):
        return sd_mecha.merge_methods.add_cosine_a(a, b, alpha=alpha)

    @staticmethod
    def add_cosine_b(a, b, alpha: Hyper):
        return sd_mecha.merge_methods.add_cosine_b(a, b, alpha=alpha)

    @staticmethod
    def tensor_sum(a, b, width: Hyper, offset: Hyper):
        return sd_mecha.tensor_sum(a, b, width=width, offset=offset)

    @staticmethod
    def top_k_tensor_sum(a, b, width: Hyper, offset: Hyper):
        return sd_mecha.merge_methods.top_k_tensor_sum(a, b, width=width, offset=offset)

    @staticmethod
    def add_difference(a, b, alpha: Hyper):
        return sd_mecha.add_difference(a, b, alpha=alpha)

    @staticmethod
    def train_difference(a, b, c, alpha: Hyper):
        return sd_mecha.train_difference(a, b, c, alpha=alpha)

    @staticmethod
    def add_opposite(a, b, c, alpha: Hyper):
        return sd_mecha.merge_methods.add_opposite(a, b, c, alpha=alpha)

    @staticmethod
    def clamped_add_opposite(a, b, c, alpha: Hyper):
        return sd_mecha.merge_methods.clamped_add_opposite(a, b, c, alpha=alpha)

    @staticmethod
    def select_max_delta(a, b, alpha: Hyper):
        return sd_mecha.merge_methods.select_max_delta(a, b, alpha=alpha)

    @staticmethod
    def multiply_quotient(a, b, c, alpha: Hyper):
        return sd_mecha.merge_methods.multiply_quotient(a, b, c, alpha=alpha)

    @staticmethod
    def crossover(a, b, alpha: Hyper, tilt: Hyper):
        return sd_mecha.crossover(a, b, alpha=alpha, tilt=tilt)

    @staticmethod
    def distribution_crossover(a, b, c, alpha: Hyper, tilt: Hyper):
        return sd_mecha.distribution_crossover(a, b, c, alpha=alpha, tilt=tilt)

    # custom methods
    @staticmethod
    @convert_to_recipe
    def determinant_sum(
        a: Tensor | SameMergeSpace,
        b: Tensor | SameMergeSpace,
        *,
        alpha: Hyper = 0.5,
        **kwargs,
    ) -> Tensor | SameMergeSpace:
        key = kwargs.get("key", "")
        if key.endswith(("in_proj_weight", "in_proj_bias")):
            # workaround for concatenated attention projection layers
            vs = []
            for i, k in enumerate(("to_q", "to_k", "to_v")):
                k_kwargs = kwargs.copy()
                k_kwargs["key"] = key.replace("in_proj_", f"{k}.")
                dim = a.shape[0] // 3
                t_start = dim*i
                t_end = dim*(i+1)
                k_a = a[t_start:t_end]
                k_b = b[t_start:t_end]
                vs.append(MergeMethods.determinant_sum.__wrapped__(k_a, k_b, **k_kwargs))
            return torch.cat(vs)

        if key.endswith("bias"):
            return sd_mecha.merge_methods.weighted_sum.__wrapped__(a, b, alpha=alpha)

        is_conv_3x3 = len(a.shape) == 4 and a.shape[-1] != 1
        is_conv_1x1 = len(a.shape) == 4 and a.shape[-1] == 1
        original_shape = a.shape
        if is_conv_3x3:
            shape_2d = (-1, functools.reduce(operator.mul, a.shape[1:]))
        elif is_conv_1x1:
            shape_2d = (-1, functools.reduce(operator.mul, a.shape[1:]))
        elif not a.shape:
            shape_2d = (1, 1)
        else:
            shape_2d = (-1, a.shape[-1])

        a_neurons = a.reshape(*shape_2d)
        b_neurons = b.reshape(*shape_2d)
        ab_neurons = a_neurons * (1 - alpha) + b_neurons * alpha

        svd_driver = "gesvd" if a.is_cuda else None
        a_s = torch.linalg.svdvals(a_neurons, driver=svd_driver)
        b_s = torch.linalg.svdvals(b_neurons, driver=svd_driver)
        ab_s = torch.linalg.svdvals(ab_neurons, driver=svd_driver)

        def pdet(s):
            return (s.log().sum() / len(s)).exp()

        a_pdet = pdet(a_s)
        b_pdet = pdet(b_s)
        ab_pdet = pdet(ab_s)

        ab_rescale = torch.nan_to_num(a_pdet ** (1 - alpha) * b_pdet ** alpha / ab_pdet, nan=1, posinf=1)

        return (a * (1 - alpha) + b * alpha) * ab_rescale
