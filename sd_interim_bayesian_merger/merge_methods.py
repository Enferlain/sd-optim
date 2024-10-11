import sd_mecha
import functools
import operator
import torch
import math
import torch.nn.functional as F
import numpy as np
from scipy.linalg import sqrtm

from scipy.stats import binom, rankdata
from torch import Tensor
from typing import Optional, Callable, Dict, Tuple, TypeVar, Generic, get_type_hints, get_origin, Union, get_args, List, Set, Iterable
from pytorch_wavelets import DWTForward, DWTInverse

from sd_mecha.hypers import Hyper
from sd_mecha.merge_space import MergeSpace
from sd_mecha.merge_methods import SameMergeSpace
from sd_mecha.extensions.merge_method import LiftFlag, convert_to_recipe

EPSILON = 1e-10

DeltaMergeSpace = TypeVar("DeltaMergeSpace", bound=LiftFlag[MergeSpace.DELTA])


class MergeMethods:

    @staticmethod
    def weighted_sum(a, b, alpha: Hyper, device=None):
        return sd_mecha.weighted_sum(a, b, alpha=alpha, device=device)

    @staticmethod
    def rotate(a, b, alignment: Hyper, alpha: Hyper, device=None, cache=None):
        return sd_mecha.rotate(a, b, alignment=alignment, alpha=alpha, device=device, cache=cache)

    @staticmethod
    def slerp(a, b, alpha: Hyper, device=None):
        return sd_mecha.slerp(a, b, alpha=alpha, device=device)

    @staticmethod
    def geometric_sum(a, b, alpha: Hyper, device=None):
        return sd_mecha.geometric_sum(a, b, alpha=alpha, device=device)

    @staticmethod
    def add_cosine_a(a, b, alpha: Hyper, device=None):
        return sd_mecha.merge_methods.add_cosine_a(a, b, alpha=alpha, device=device)

    @staticmethod
    def add_cosine_b(a, b, alpha: Hyper, device=None):
        return sd_mecha.merge_methods.add_cosine_b(a, b, alpha=alpha, device=device)

    @staticmethod
    def tensor_sum(a, b, width: Hyper, offset: Hyper, device=None):
        return sd_mecha.tensor_sum(a, b, width=width, offset=offset, device=device)

    @staticmethod
    def top_k_tensor_sum(a, b, width: Hyper, offset: Hyper, device=None):
        return sd_mecha.merge_methods.top_k_tensor_sum(a, b, width=width, offset=offset, device=device)

    @staticmethod
    def add_difference(a, b, alpha: Hyper, device=None):
        return sd_mecha.add_difference(a, b, alpha=alpha, device=device)

    @staticmethod
    def train_difference(a, b, c, alpha: Hyper, device=None):
        return sd_mecha.train_difference(a, b, c, alpha=alpha, device=device)

    @staticmethod
    def add_opposite(a, b, c, alpha: Hyper, device=None):
        return sd_mecha.merge_methods.add_opposite(a, b, c, alpha=alpha, device=device)

    @staticmethod
    def clamped_add_opposite(a, b, c, alpha: Hyper, device=None):
        return sd_mecha.merge_methods.clamped_add_opposite(a, b, c, alpha=alpha, device=device)

    @staticmethod
    def select_max_delta(a, b, alpha: Hyper, device=None):
        return sd_mecha.merge_methods.select_max_delta(a, b, alpha=alpha, device=device)

    @staticmethod
    def multiply_quotient(a, b, c, alpha: Hyper, device=None):
        return sd_mecha.merge_methods.multiply_quotient(a, b, c, alpha=alpha, device=device)

    @staticmethod
    def crossover(a, b, alpha: Hyper, tilt: Hyper, device=None):
        return sd_mecha.crossover(a, b, alpha=alpha, tilt=tilt, device=device)

    @staticmethod
    def distribution_crossover(a, b, c, alpha: Hyper, tilt: Hyper, device=None):
        return sd_mecha.distribution_crossover(a, b, c, alpha=alpha, tilt=tilt, device=device)

    @staticmethod
    def ties_sum_extended(*models, k: Hyper, apply_stock: Hyper = 0.05, apply_median: Hyper = 0.1, eps: Hyper = 1e-5, ftol: Hyper = 1e-10, maxiter: Hyper = 150, **kwargs):
        return sd_mecha.ties_sum_extended(*models, k=k, apply_stock=apply_stock, apply_median=apply_median, eps=eps, ftol=ftol, maxiter=maxiter, **kwargs)

    @staticmethod
    @convert_to_recipe
    def ties_sum_with_dropout(
            *models: Tensor | LiftFlag[MergeSpace.DELTA],
            probability: Hyper = 0.9,
            della_eps: Hyper = 0.0,
            rescale: Hyper = 1.0,
            k: Hyper = 0.2,
            vote_sgn: Hyper = 0.0,
            apply_stock: Hyper = 0.0,
            cos_eps: Hyper = 1e-6,
            apply_median: Hyper = 1.0,
            eps: Hyper = 1e-6,
            maxiter: Hyper = 100,
            ftol: Hyper = 1e-20,
            seed: Optional[int] = 218,
            **kwargs,
    ) -> Tensor | LiftFlag[MergeSpace.DELTA]:
        """
        Applies TIES merging with dropout to a variable number of delta tensors.

        Args:
            *models: The delta tensors to merge.
            probability: The dropout probability (0 <= probability <= 1).
            della_eps: The DELLA epsilon parameter, controlling magnitude-based dropout.
            rescale:  The rescaling factor for the merged delta.
            k: The TIES parameter trimming threshold.
            vote_sgn:  The TIES-SOUP mode activation parameter.
            apply_stock:  The Model Stock activation parameter.
            cos_eps: The cosine similarity epsilon for Model Stock.
            apply_median:  The Geometric Median activation parameter.
            eps: The epsilon for the Geometric Median calculation.
            maxiter: The maximum number of iterations for Geometric Median.
            ftol:  The tolerance for convergence for Geometric Median.
            seed: The random seed for dropout.
            **kwargs: Additional keyword arguments.

        Returns:
            The merged delta tensor.
        """
        if not models or probability == 1:
            return torch.tensor(0.0, device=models[0].device if models else 'cpu')

        device = models[0].device
        generator = torch.Generator(device)
        if seed is not None:
            generator.manual_seed(seed)

        # Apply dropout to each delta tensor
        dropped_deltas = []
        for delta in models:
            dropout_mask = MergeMethods.create_dropout_mask(delta, probability, della_eps, generator)
            dropped_deltas.append(delta * dropout_mask)

        # Apply TIES merging to the dropped deltas
        merged_delta = sd_mecha.merge_methods.ties_sum_extended.__wrapped__(
            *dropped_deltas,
            k=k,
            vote_sgn=vote_sgn,
            apply_stock=apply_stock,
            cos_eps=cos_eps,
            apply_median=apply_median,
            eps=eps,
            maxiter=maxiter,
            ftol=ftol
        )

        # Rescale the merged delta based on the dropout probability and rescale factor
        rescalar = 1.0 if probability == 1.0 else max(0, (1.0 - probability) ** rescale)
        return merged_delta / rescalar

    def create_dropout_mask(delta: Tensor, probability: float, della_eps: float, generator: torch.Generator) -> Tensor:
        """Creates a dropout mask based on the DELLA variation.

        Args:
            delta: The delta tensor.
            probability: The base dropout probability.
            della_eps: The DELLA epsilon parameter.
            generator: The random number generator.

        Returns:
            The dropout mask as a tensor.
        """
        p_min = torch.full(delta.shape, 1 - probability, device=delta.device, dtype=delta.dtype)
        if della_eps != 0.0:
            # Calculate rank-based adjustments
            ranks = torch.argsort(torch.argsort(delta.abs().flatten())).reshape(delta.shape).float()
            delta_i = ((ranks / delta.numel()) - 0.5) * della_eps
            # Ensure probabilities are within the valid range
            probabilities = torch.clamp(p_min + delta_i, 0.0, 1.0)
        else:
            probabilities = p_min

        return torch.bernoulli(probabilities, generator=generator)


    ### CUSTOM METHODS ###


    @staticmethod
    @convert_to_recipe
    def slerp_norm(
            a: Tensor | SameMergeSpace,
            b: Tensor | SameMergeSpace,
            *,
            alpha: Hyper = 0.5,
            **kwargs,
    ) -> Tensor | SameMergeSpace:
        a_norm = a.norm()
        b_norm = b.norm()

        a_normalized = a / a_norm  # Normalize 'a'
        b_normalized = b / b_norm  # Normalize 'b'

        ab_dot = (a_normalized * b_normalized).sum().clamp(-1 + EPSILON, 1 - EPSILON)  # Add epsilon for stability

        omega = torch.arccos(ab_dot)

        # Check for near-zero omega to avoid division by zero
        if torch.abs(omega) < EPSILON:
            # If omega is near zero, tensors are almost parallel so simple weighted sum is sufficient
            merged_tensor = a * (1 - alpha) + b * alpha
            merged_norm = a_norm * (1 - alpha) + b_norm * alpha  # Weighted sum of norms for omega ~0

        else:
            a_contrib = a_normalized * torch.sin((1 - alpha) * omega)
            b_contrib = b_normalized * torch.sin(alpha * omega)

            merged_tensor = (a_contrib + b_contrib) / torch.sin(omega)
            merged_norm = a_norm ** (1 - alpha) * b_norm ** alpha  # Geometric interpolation of norms

        # Scale the merged tensor by the merged norm
        merged_tensor = merged_tensor * merged_norm

        if torch.isnan(merged_tensor).any():
            return sd_mecha.weighted_sum.__wrapped__(a, b, alpha=alpha)

        return merged_tensor

    @staticmethod
    @convert_to_recipe
    def weighted_sum_norm(
            a: Tensor | SameMergeSpace,
            b: Tensor | SameMergeSpace,
            *,
            alpha: Hyper = 0.5,
            **kwargs,
    ) -> Tensor | SameMergeSpace:

        key = kwargs.get("key", "")

        if key.endswith((".weight", ".bias")):  # Detect layernorm keys
            if key.endswith(".weight"):
                # Geometric interpolation for weights
                merged_tensor = a ** (1 - alpha) * b ** alpha
            else:
                # Linear interpolation for biases
                merged_tensor = a * (1 - alpha) + b * alpha

            return merged_tensor
        else:
            return (1 - alpha) * a + alpha * b

    @staticmethod
    @convert_to_recipe
    def slerp_norm_sign(
            a: Tensor | SameMergeSpace,
            b: Tensor | SameMergeSpace,
            *,
            alpha: Hyper = 0.5,
            **kwargs,
    ) -> Tensor | SameMergeSpace:
        key = kwargs.get("key", "")

        # Handle Layer Normalization layers
        if a.dim() == 1 and key.endswith((".weight", ".scale")):
            # Geometric interpolation for weight (scale)
            weight = a ** (1 - alpha) * b ** alpha
            # Linear interpolation for bias (if present)
            bias = a * (1 - alpha) + b * alpha if "bias" in key else None
            return torch.cat([weight, bias]) if bias is not None else weight

        # Handle convolutional layers in Fourier space
        if len(a.shape) == 4:  # 4D tensor (convolutional kernel)
            return MergeMethods.merge_conv_layers(a, b, alpha)

            # Original shape for reshaping later
        original_shape = a.shape

        if not original_shape:
            shape_2d = (1, 1)
        elif len(a.shape) == 4:
            shape_2d = (-1, functools.reduce(operator.mul, a.shape[1:]))
        else:
            shape_2d = (-1, a.shape[-1])
        a = a.reshape(*shape_2d)
        b = b.reshape(*shape_2d)

        # Geometric interpolation of norms using complex numbers
        a_complex = a.to(a.dtype.to_complex())
        b_complex = b.to(b.dtype.to_complex())
        merged_tensor = a_complex ** (1 - alpha) * b_complex ** alpha
        merged_tensor = merged_tensor.real + merged_tensor.imag  # Add imaginary component

        return merged_tensor.reshape(original_shape)

    def merge_conv_layers(a: Tensor, b: Tensor, alpha: float) -> Tensor:
        """Merges convolutional layer weights using Fourier transform and geometric interpolation."""
        a_fft = torch.fft.fftn(a, dim=tuple(range(1, a.dim())))  # Apply FFT along spatial dimensions
        b_fft = torch.fft.fftn(b, dim=tuple(range(1, b.dim())))
        merged_fft = a_fft ** (1 - alpha) * b_fft ** alpha
        merged_tensor = torch.fft.ifftn(merged_fft, dim=tuple(range(1, a.dim()))).real  # Inverse FFT and take real part

        # Handle 1x1 convolutions using polar interpolation
        if a.shape[-1] == 1 and a.shape[-2] == 1:
            U, S, V = torch.linalg.svd(merged_tensor.reshape(-1, a.shape[0]))
            merged_tensor = U @ torch.diag(S ** alpha) @ V.t()  # Polar interpolation

        return merged_tensor.reshape(a.shape)

    @staticmethod
    @convert_to_recipe
    def distribution_merge(
        a: Tensor | SameMergeSpace,
        b: Tensor | SameMergeSpace,
        *,
        alpha: Hyper = 0.5,
        **kwargs,
    ) -> Tensor | SameMergeSpace:
        """Merges tensors using distribution interpolation for specific layers."""
        key = kwargs.get("key", "")

        if key.endswith((".token_embedding.weight", ".shared.weight")):
            return MergeMethods.distribution_interpolate(a, b, alpha, kwargs)
        else:
            # Apply a default merging method for other layers (e.g., weighted_sum)
            return sd_mecha.weighted_sum.__wrapped__(a, b, alpha=alpha)

    def distribution_interpolate(a, b, alpha, kwargs):
        a = a.mH
        b = b.mH

        a_mean = a.mean(dim=1, keepdim=True)
        b_mean = b.mean(dim=1, keepdim=True)

        a_cov = torch.cov(a - a_mean)
        b_cov = torch.cov(b - b_mean)

        a_d_neghalf = torch.diag_embed(1 / a_cov.diag().sqrt())
        b_d_neghalf = torch.diag_embed(1 / b_cov.diag().sqrt())

        a_corr = a_d_neghalf @ a_cov @ a_d_neghalf
        b_corr = b_d_neghalf @ b_cov @ b_d_neghalf

        a_corr_neghalf = MergeMethods.fractional_pd(a_corr, -1 / 2)
        b_corr_neghalf = MergeMethods.fractional_pd(b_corr, -1 / 2)

        a_w_transform = a_corr_neghalf @ a_d_neghalf
        b_w_transform = b_corr_neghalf @ b_d_neghalf

        a_w = a_w_transform @ (a - a_mean)
        b_w = b_w_transform @ (b - b_mean)

        res_w = a_w*(1-alpha) + b_w*alpha
        res_centered = MergeMethods.positive_definite_interpolate(
            *torch.linalg.eigh(MergeMethods.fractional_pd(a_w_transform, -1)),
            *torch.linalg.eigh(MergeMethods.fractional_pd(b_w_transform, -1)),
            alpha=alpha,
            power=1,
            kwargs=kwargs,
        ) @ res_w
        res = res_centered + a_mean * (1 - alpha) + b_mean * alpha
        res = res.mH.contiguous()

        return res

    def positive_definite_interpolate(a_v, a_vs, b_v, b_vs, alpha, power, kwargs):
        original_dtype = a_vs.dtype
        complex_dtype = original_dtype.to_complex()

        a_v, a_vs = a_v.to(dtype=complex_dtype), a_vs.to(dtype=complex_dtype)

        a_half = a_vs @ torch.diag_embed(a_v ** (1 / 2)) @ a_vs.mH
        a_neghalf = a_vs @ torch.diag_embed((a_v ** (-1 / 2)).nan_to_num(nan=0)) @ a_vs.mH
        b = (b_vs @ torch.diag_embed(b_v) @ b_vs.mH).to(complex_dtype)

        delta_v, delta_vs = torch.linalg.eigh(a_neghalf @ b @ a_neghalf)
        delta = delta_vs @ torch.diag_embed((delta_v.to(complex_dtype) ** alpha).nan_to_num(nan=0)) @ delta_vs.mH
        res_v, res_vs = torch.linalg.eigh(a_half @ delta @ a_half)
        res_v_pow = (res_v ** power).to(complex_dtype)
        res = res_vs @ torch.diag_embed(res_v_pow) @ res_vs.mH

        if not original_dtype.is_complex and res.imag.abs().max() > 1e-6:
            print(
                f"fix your signs! max(|imag|): {res.imag.abs().max().item()}, det: {res.det().item()}, kwargs: {kwargs}")

        return res.to(dtype=original_dtype)

    def fractional_pd(a, power):
        v, vs = torch.linalg.eigh(a)
        v_pow = v ** power
        res = vs @ torch.diag_embed(v_pow) @ vs.mH
        return res.to(dtype=a.dtype)

    @staticmethod
    @convert_to_recipe(volatile_hypers=["cache"])
    def determinant_sum(
            a: Tensor | SameMergeSpace,
            b: Tensor | SameMergeSpace,
            *,
            alpha: Hyper = 0.5,
            cache: Optional[Dict[str, Dict[str, Tensor]]] = None,
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
                t_start = dim * i
                t_end = dim * (i + 1)
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

        svd_driver = "gesvd" if a.is_cuda else None

        # Cache handling
        if cache is not None:
            key = kwargs["key"]
            if key not in cache:
                cache[key] = {}
            cache = cache[key]

        if cache is not None and "a_s" in cache and "b_s" in cache:
            a_s = cache["a_s"].to(a.device, a.dtype)
            b_s = cache["b_s"].to(a.device, a.dtype)
        else:
            a_s = torch.linalg.svdvals(a_neurons, driver=svd_driver)
            b_s = torch.linalg.svdvals(b_neurons, driver=svd_driver)

            if cache is not None:
                cache["a_s"] = a_s.to("cpu")
                cache["b_s"] = b_s.to("cpu")

        ab_neurons = a_neurons * (1 - alpha) + b_neurons * alpha
        ab_s = torch.linalg.svdvals(ab_neurons, driver=svd_driver)

        def pdet(s):
            return (s.log().sum() / len(s)).exp()

        a_pdet = pdet(a_s)
        b_pdet = pdet(b_s)
        ab_pdet = pdet(ab_s)

        ab_rescale = torch.nan_to_num(a_pdet ** (1 - alpha) * b_pdet ** alpha / ab_pdet, nan=1, posinf=1)

        return (a * (1 - alpha) + b * alpha) * ab_rescale

    @staticmethod
    @convert_to_recipe
    def wavelet_merge(
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
                t_start = dim * i
                t_end = dim * (i + 1)
                k_a = a[t_start:t_end]
                k_b = b[t_start:t_end]
                vs.append(MergeMethods.wavelet_merge.__wrapped__(k_a, k_b, **k_kwargs))
            return torch.cat(vs)

        # Reshape tensors to 2D
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

        a = a.reshape(*shape_2d)
        b = b.reshape(*shape_2d)
        shape_2d = a.shape

        dwt = DWTForward(J=4, wave='db4', mode='zero')
        idwt = DWTInverse(wave='db4', mode='zero')

        dwt.to(device=a.device, dtype=a.dtype)
        idwt.to(device=a.device, dtype=a.dtype)

        a_yl, a_yh = dwt(a.unsqueeze(0).unsqueeze(0))
        b_yl, b_yh = dwt(b.unsqueeze(0).unsqueeze(0))

        merged_detail = alpha * a_yl + (1 - alpha) * b_yl, [alpha * aa + (1 - alpha) * bb for aa, bb in zip(a_yh, b_yh)]

        merged_tensor = idwt(merged_detail).squeeze(0).squeeze(0)
        merged_tensor = merged_tensor[:shape_2d[0], :shape_2d[1]]

        return merged_tensor.reshape(original_shape)

    @staticmethod
    @convert_to_recipe
    def anchored_guided_alignment(
            a: Tensor | SameMergeSpace,
            b: Tensor | SameMergeSpace,
            c: Tensor | SameMergeSpace,
            *,
            alpha: Hyper = 0.5,
            beta: Hyper = 0.5,
            **kwargs,
    ) -> Tensor | SameMergeSpace:
        """Merges tensors A and B using anchored neuron train difference with simplified alignment and slerp interpolation.

        Args:
            a (Tensor): The first tensor (assumed to be fp16).
            b (Tensor): The second tensor (assumed to be fp16).
            c (Tensor): The anchor tensor (assumed to be fp16).
            alpha (float): The alpha parameter for slerp interpolation (0 <= alpha <= 1).
            beta (float): The beta parameter for dissimilarity adjustment (0 <= beta <= 1).

        Returns:
            Tensor: The merged tensor (in fp16).
        """
        key = kwargs.get("key", "")
        if key.endswith(("in_proj_weight", "in_proj_bias")):
            # workaround for concatenated attention projection layers
            vs = []
            for i, k in enumerate(("to_q", "to_k", "to_v")):
                k_kwargs = kwargs.copy()
                k_kwargs["key"] = key.replace("in_proj_", f"{k}.")
                dim = a.shape[0] // 3
                t_start = dim * i
                t_end = dim * (i + 1)
                k_a = a[t_start:t_end]
                k_b = b[t_start:t_end]
                k_c = c[t_start:t_end]
                vs.append(MergeMethods.anchored_guided_alignment.__wrapped__(k_a, k_b, k_c, **k_kwargs))
            return torch.cat(vs)

        original_shape = a.shape

        # Align to anchor using frequency domain alignment with anchor guidance
        aligned_a = a
        aligned_b = MergeMethods.frequency_selective_alignment(a, b, c)

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

        # Convert to float32 for calculations
        aligned_a = aligned_a.reshape(*shape_2d)
        aligned_b = aligned_b.reshape(*shape_2d)
        c = c.reshape(*shape_2d)

        # --- Refinement (Neuron Train Difference) with Smoothing ---
        threshold = torch.max(
            (aligned_a - c).norm(dim=1, keepdim=True), (aligned_b - c).norm(dim=1, keepdim=True)
        )
        dissimilarity = (1 - torch.nan_to_num(
            ((aligned_a - c) * (aligned_b - c)).sum(dim=1, keepdim=True) / threshold ** 2, nan=0)) / 2

        # Apply Gaussian smoothing to the dissimilarity
        dissimilarity = MergeMethods.gaussian_blur(dissimilarity, kernel_size=3)

        # --- Merging with Slerp/Nlerp (with clamping) ---
        aligned_a_norm = aligned_a / (aligned_a.norm(dim=1, keepdim=True) + EPSILON)
        aligned_b_norm = aligned_b / (aligned_b.norm(dim=1, keepdim=True) + EPSILON)

        ab_dot = (aligned_a_norm * aligned_b_norm).sum(dim=1, keepdim=True).clip(-1 + EPSILON, 1 - EPSILON)
        omega = torch.acos(ab_dot).clip(EPSILON, math.pi - EPSILON)

        sin_omega = torch.sin(omega) + EPSILON

        a_contrib = aligned_a_norm * torch.sin((1 - alpha) * omega) / sin_omega
        b_contrib = aligned_b_norm * torch.sin(alpha * omega) / sin_omega

        # Use nlerp if vectors are close to parallel or anti-parallel
        if torch.all(1 - torch.abs(ab_dot) < EPSILON):
            merged_tensor = MergeMethods.nlerp(aligned_a_norm, aligned_b_norm, alpha=alpha) * (
                    torch.norm(aligned_a, dim=1, keepdim=True) * (1 - alpha)
                    + torch.norm(aligned_b, dim=1, keepdim=True) * alpha
            )
        else:
            merged_tensor = (a_contrib + b_contrib) * (
                    torch.norm(aligned_a, dim=1, keepdim=True) * (1 - alpha)
                    + torch.norm(aligned_b, dim=1, keepdim=True) * alpha
            )

        merged_tensor += (aligned_b - c) * beta * dissimilarity

        return merged_tensor.reshape(original_shape)

    def nlerp(a: Tensor, b: Tensor, *, alpha: float) -> Tensor:
        return (a * alpha + b * (1 - alpha)).div(torch.norm(a * alpha + b * (1 - alpha)) + EPSILON)

    def gaussian_blur(tensor: Tensor, kernel_size: int) -> Tensor:
        """Applies 1D Gaussian blur to a tensor along the feature dimension.

        Args:
            tensor (torch.Tensor): Input tensor.
            kernel_size (int): Size of the Gaussian kernel (must be odd).

        Returns:
            torch.Tensor: Blurred tensor.
        """
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd.")

        # Create a Gaussian kernel
        kernel = torch.exp(
            -(torch.arange(kernel_size, device=tensor.device) - kernel_size // 2) ** 2 / (2 * (kernel_size / 3) ** 2))
        kernel /= kernel.sum()

        # Reshape the kernel for convolution
        kernel = kernel.view(1, 1, -1)

        # Apply convolution along the feature dimension (dim=2)
        blurred_tensor = F.conv1d(tensor.unsqueeze(1).double(), kernel.double(), padding=kernel_size // 2).squeeze(1)

        return blurred_tensor

    def frequency_selective_alignment(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
        """Aligns tensor 'b' to 'a' in specific frequency bands, guided by 'c', by directly adjusting magnitudes.

        Args:
            a (Tensor): The first tensor (2D).
            b (Tensor): The tensor to be aligned (2D).
            c (Tensor): The anchor tensor (2D).

        Returns:
            Tensor: The aligned tensor 'b'.
        """

        # Reshape to 1D
        a_flat = a.reshape(-1).float()
        b_flat = b.reshape(-1).float()
        c_flat = c.reshape(-1).float()

        # Apply FFT along the feature dimension (dim=1)
        a_dft = torch.fft.rfft(a_flat)
        b_dft = torch.fft.rfft(b_flat)
        c_dft = torch.fft.rfft(c_flat)

        # Calculate spectral centroids (with epsilon for stability)
        a_centroid = MergeMethods.calculate_spectral_centroid(a_dft)
        b_centroid = MergeMethods.calculate_spectral_centroid(b_dft)
        c_centroid = MergeMethods.calculate_spectral_centroid(c_dft)

        # Dynamic beta based on spectral centroid distance (normalized)
        dissimilarity = abs(a_centroid - b_centroid)
        max_centroid = a_dft.shape[0] / 2  # Maximum possible spectral centroid
        normalized_dissimilarity = dissimilarity / max_centroid
        dynamic_beta = 1 - normalized_dissimilarity

        # Use spectral centroids to define passband and stopband (with overlap)
        centroid_margin = 0.1 * a_dft.shape[0]  # Overlap margin
        passband = (0, int(min(a_centroid, c_centroid) * a_dft.shape[0] - centroid_margin))
        stopband = (int(max(a_centroid, c_centroid) * a_dft.shape[0] + centroid_margin), a_dft.shape[0])

        # Define transition start and end
        transition_start = passband[1]
        transition_end = stopband[0]

        # --- Apply Magnitude Scaling Using dynamic_beta ---
        b_dft_magnitude = torch.abs(b_dft)
        scaled_magnitude = b_dft_magnitude * dynamic_beta
        b_dft = torch.polar(scaled_magnitude, torch.angle(b_dft))

        # Calculate magnitude difference between 'a' and 'c'
        a_dft_magnitude = torch.abs(a_dft[passband[0]:passband[1]])
        b_dft_magnitude = torch.abs(b_dft[passband[0]:passband[1]])
        c_dft_magnitude = torch.abs(c_dft[passband[0]:passband[1]])

        # Weighted average of 'a' and 'b' magnitudes using dynamic_beta
        weighted_magnitude = (1 - dynamic_beta) * a_dft_magnitude + dynamic_beta * b_dft_magnitude

        magnitude_difference = weighted_magnitude - c_dft_magnitude

        # Apply smooth magnitude adjustment using a Gaussian function
        transition_width = transition_end - transition_start
        if transition_width > 0:
            transition_slope = 1.0 / (transition_width + 1e-8)
            smooth_adjustment = torch.sigmoid(
                transition_slope * (torch.arange(magnitude_difference.shape[0]) - transition_start))
            b_dft[passband[0]:passband[1]] += magnitude_difference * smooth_adjustment

        # Apply inverse FFT to get the aligned tensor in the time domain
        aligned_b = torch.fft.irfft(b_dft, a_flat.shape[0])

        return aligned_b.reshape(a.shape)

    def calculate_spectral_centroid(dft: Tensor) -> float:
        """
        Calculates the spectral centroid of a tensor in the frequency domain.

        Args:
            dft (torch.Tensor): The tensor's Fourier Transform.

        Returns:
            float: The spectral centroid.
        """
        frequencies = torch.arange(dft.shape[0])
        magnitudes = torch.abs(dft)
        centroid = (frequencies * magnitudes).sum() / (magnitudes.sum() + EPSILON)
        return centroid.item()

    @staticmethod
    @convert_to_recipe
    def add_difference_var_clip(
            a: Tensor | SameMergeSpace,
            b: Tensor | SameMergeSpace,
            c: Tensor | SameMergeSpace,
            *,
            alpha: Hyper = 1.0,
            **kwargs,
    ) -> Tensor | SameMergeSpace:
        bc_corr = torch.corrcoef(torch.stack([
            (b - b.mean()).flatten(),
            (c - c.mean()).flatten()
        ], dim=0))[0, 1]

        b_var = b.var(correction=0)
        c_var = c.var(correction=0)

        bc_cov = bc_corr * torch.sqrt(b_var * c_var)

        min_corr = 0.9999
        if bc_corr < min_corr:
            bc_scale = torch.sqrt(b_var + c_var - 2 * min_corr * torch.sqrt(b_var * c_var)) / torch.sqrt(
                b_var + c_var - 2 * bc_cov)
        else:
            bc_scale = 1.0

        bc = b - c
        bc = (bc - bc.mean()) * bc_scale + bc.mean()
        res = a + alpha * bc
        return (res - res.mean()) * a.std(correction=0) / res.std(correction=0) + a.mean()

    @staticmethod
    @convert_to_recipe
    def gram_schmidt_ortho(
            a: Tensor | SameMergeSpace,
            b: Tensor | SameMergeSpace,
            c: Tensor | SameMergeSpace,
            *,
            alpha: Hyper = 1.0,
            **kwargs,
    ) -> Tensor | SameMergeSpace:
        # Calculate the vectors
        vector_a = a - c
        vector_b = b - c

        # Calculate the projection of B onto A
        projection_b_on_a = (torch.dot(vector_b.flatten(), vector_a.flatten()) / torch.dot(vector_a.flatten(),
                                                                                           vector_a.flatten())) * vector_a

        # Magnitude adjustment based on the difference between A and C
        magnitude_ratio = torch.norm(projection_b_on_a) / torch.norm(vector_a)
        adjusted_projection = projection_b_on_a * (1 + alpha * (magnitude_ratio - 1))

        # Add the adjusted projection to the base model
        return a + adjusted_projection

    @staticmethod
    @convert_to_recipe
    def orth_pro(
            a: Tensor | SameMergeSpace,
            b: Tensor | SameMergeSpace,
            c: Tensor | SameMergeSpace,
            *,
            alpha: Hyper = 1.0,
            use_perp: Hyper = 0,
            ab_only: Hyper = 0,
            noisy_c: Hyper = 0,
            noisy_c_sgn_flt: Hyper = 0,
            **kwargs,
    ) -> Tensor | SameMergeSpace:
        """
        Merges tensors 'a' and 'b' using Orthogonal Procrustes alignment with options for perpendicular
        component projection, noise injection, and control over alignment scope.

        Args:
            a (Tensor): The first tensor.
            b (Tensor): The second tensor.
            c (Tensor): The anchor tensor.
            alpha (float): The interpolation factor between the original tensor 'b' and the mapped
                           tensor (0 <= alpha <= 1).
            use_perp (bool): If True, projects 'a' onto the perpendicular component of 'b' before alignment.
            ab_only (bool): If True, performs alignment only between 'a' and 'b', ignoring 'c'.
            noisy_c (float): The standard deviation of Gaussian noise added to 'c' (0 for no noise).
            noisy_c_sgn_flt (bool): If True, filters the noise added to 'c' to match the sign of 'c'.
            **kwargs: Additional keyword arguments.

        Returns:
            Tensor: The merged tensor.
        """
        # Reshape tensors to 2D
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

        a = a.reshape(shape_2d)
        b = b.reshape(shape_2d)
        c = c.reshape(shape_2d) if not noisy_c else MergeMethods.create_noisy_tensor(c.reshape(shape_2d),
                                                                                     sign_filter=noisy_c_sgn_flt,
                                                                                     seed=0)
        ac = a if ab_only else (a - c)
        bc = b if ab_only else (b - c)

        if use_perp:
            norm_bc = torch.linalg.norm(bc) + 1e-20
            ac = ac - bc * (bc / norm_bc * (ac / norm_bc)).sum()

        res = MergeMethods.orthogonal_procrustes(ac, bc)
        if ab_only:
            return torch.lerp(b.reshape(original_shape), res.reshape(original_shape), alpha)
        else:
            return torch.lerp(b.reshape(original_shape), (c + res).reshape(original_shape), alpha)

    def orthogonal_procrustes(a: Tensor, b: Tensor):
        # Compute the QR decomposition of (a - c)
        Q, R = torch.qr(a)

        # Compute the mapping matrix
        mapping_matrix = torch.mm(Q.t(), b)

        # Map (a - c) to (b - c)
        mapped_tensor = torch.mm(Q, mapping_matrix)

        return mapped_tensor

    def create_noisy_tensor(
            a: Tensor,
            seed=218,
            sign_filter=False,
    ) -> Tensor:
        torch.manual_seed(seed)

        dist = torch.normal(a.mean(), a.std(correction=0, keepdim=True))

        if sign_filter:
            signs = torch.sign(dist)

            final_sign = torch.sign(a)

            delta_filters = (signs == final_sign).float()

            param_counts = torch.sum(delta_filters, dim=0)

            filtered_delta = (dist * delta_filters)

            filtered_delta = filtered_delta.sum(dim=0)

            dist = torch.nan_to_num(filtered_delta / param_counts)

        return dist

    @staticmethod
    @convert_to_recipe
    def lu_merge(
            a: Tensor | SameMergeSpace,
            b: Tensor | SameMergeSpace,
            c: Tensor | SameMergeSpace,
            *,
            alpha: Hyper = 1.0,
            theta: Hyper = 1.0,
            use_perp: Hyper = 0,
            ab_only: Hyper = 0,
            **kwargs,
    ) -> Tensor | SameMergeSpace:
        """
        Merges tensors 'a' and 'b' using LU decomposition interpolation with optional alignment adjustments.

        Args:
            a (Tensor): The first tensor.
            b (Tensor): The second tensor.
            c (Tensor): The anchor tensor.
            alpha (float): The interpolation factor between the original tensor 'a' and the merged
                           tensor (0 <= alpha <= 1).
            theta (float): The interpolation factor for the LU decomposition components of 'a' and 'b'
                           (0 <= theta <= 1).
            use_perp (bool): If True, projects 'a' onto the perpendicular component of 'b' before merging.
            ab_only (bool): If True, performs merging only between 'a' and 'b', ignoring 'c'.
            **kwargs: Additional keyword arguments.

        Returns:
            Tensor: The merged tensor.
        """
        # Reshape tensors to 2D
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

        a = a.reshape(shape_2d)
        b = b.reshape(shape_2d)
        c = c.reshape(shape_2d)

        # Calculate difference tensors if not ab_only
        ac = a if ab_only else (a - c)
        bc = b if ab_only else (b - c)

        # Project 'ac' onto the perpendicular component of 'bc' if use_perp is True
        if use_perp:
            norm_bc = torch.linalg.norm(bc) + 1e-20
            ac = ac - bc * (bc / norm_bc * (ac / norm_bc)).sum()

        # Perform LU decomposition-based merging
        res = MergeMethods.lu_decompose(ac, bc, theta)

        # Interpolate between original tensor 'A' and the merged result based on 'alpha'
        if ab_only:
            return torch.lerp(a.reshape(original_shape), res.reshape(original_shape), alpha)
        else:
            return torch.lerp(a.reshape(original_shape), (c + res).reshape(original_shape), alpha)

    def lu_decompose(a, b, t=1.0):
        """
        Performs LU decomposition-based interpolation between tensors a and b.

        Args:
            a (Tensor): The first tensor (2D).
            b (Tensor): The second tensor (2D).
            t (float): Interpolation factor (0 <= t <= 1).

        Returns:
            Tensor: Interpolated tensor based on LU decomposition.
        """
        # Compute LU decomposition for tensors a and b
        P_A, L_A, U_A = torch.linalg.lu(a)
        P_B, L_B, U_B = torch.linalg.lu(b)

        # Interpolate L and U matrices
        L_interpolated = (1 - t) * L_A + t * L_B
        U_interpolated = (1 - t) * U_A + t * U_B

        # Combine interpolated matrices
        A_interpolated = P_A @ L_interpolated @ U_interpolated

        return A_interpolated

    @staticmethod
    @convert_to_recipe(volatile_hypers=["cache"])
    def clyb_merge(
            a: Tensor | SameMergeSpace,
            b: Tensor | SameMergeSpace,
            c: Tensor | SameMergeSpace,
            *,
            alpha: Hyper = 1.0,
            use_perp: Hyper = 1.0,
            ab_only: Hyper = 0,
            cache: Optional[Dict[str, Dict[str, Tensor]]] = None,
            **kwargs,
    ) -> Tensor | SameMergeSpace:
        """
        Merges tensors 'a' and 'b' using a combination of low-rank approximation, orthogonal projection,
        and optional perpendicular component projection and alignment adjustments.

        Args:
            a (Tensor): The source tensor.
            b (Tensor): The target tensor.
            c (Tensor): The reference tensor.
            alpha (float): The interpolation factor between the original tensor 'b' and the merged
                           tensor (0 <= alpha <= 1).
            use_perp (bool): If True, projects 'a' onto the perpendicular component of 'b' before merging.
            ab_only (bool): If True, performs merging only between 'a' and 'b', ignoring 'c'.
            cache (dict): Cache svd and qr for performance
            **kwargs: Additional keyword arguments.

        Returns:
            Tensor: The merged tensor (in fp16).
        """
        # Reshape tensors to 2D
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

        a = a.reshape(shape_2d)
        b = b.reshape(shape_2d)
        c = c.reshape(shape_2d)

        # Calculate difference tensors if not ab_only
        ac = a if ab_only else (a - c)
        bc = b if ab_only else (b - c)

        # Project 'ac' onto the perpendicular component of 'bc' if use_perp is True
        if use_perp:
            norm_bc = torch.linalg.norm(bc) + 1e-20
            ac = ac - bc * (bc / norm_bc * (ac / norm_bc)).sum()

        # Perform the core merging operation
        res = MergeMethods.clyb_align(ac, bc, cache=cache, **kwargs)

        # Interpolate between original tensor 'b' and the merged result based on 'alpha'
        if ab_only:
            return torch.lerp(b.reshape(original_shape), res.reshape(original_shape), alpha)
        else:
            return torch.lerp(b.reshape(original_shape), (c + res).reshape(original_shape), alpha)

    def clyb_align(a, b, cache: Optional[Dict[str, Dict[str, Tensor]]] = None, **kwargs):
        """
        Performs the core merging operation using QR decomposition, low-rank approximation, and orthogonal projection.

        Args:
            a (Tensor): The source tensor (2D).
            b (Tensor): The target tensor (2D).
            cache (dict): A dictionary for caching intermediate results.

        Returns:
            Tensor: The merged tensor (2D).
        """
        if cache is not None:
            key = kwargs["key"]
            if key not in cache:
                cache[key] = {}
            cache = cache[key]

            if "Qb" in cache:
                Qb = cache["Qb"].to(b.device, b.dtype)  # Reuse cached Qb
            else:
                Qb, _ = torch.qr(b)  # Calculate and cache Qb
                cache["Qb"] = Qb.to("cpu")

            if "Ua" in cache and "Sa" in cache and "Va" in cache:
                Ua = cache["Ua"].to(a.device, a.dtype)  # Reuse cached Ua
                Sa = cache["Sa"].to(a.device, a.dtype)  # Reuse cached Sa
                Va = cache["Va"].to(a.device, a.dtype)  # Reuse cached Va
            else:
                compression = 16
                q_size = max(int(torch.linalg.matrix_rank(Qb)) // compression, 1)
                iters = min(max(int(math.exp(math.log(640 / q_size))), 2), 64)
                Ua, Sa, Va = torch.svd_lowrank(a, q=q_size, niter=iters)  # Calculate and cache SVD components
                cache["Ua"] = Ua.to("cpu")
                cache["Sa"] = Sa.to("cpu")
                cache["Va"] = Va.to("cpu")
        else:  # No caching, calculate everything
            compression = 16
            Qb, _ = torch.qr(b)
            q_size = max(int(torch.linalg.matrix_rank(Qb)) // compression, 1)
            iters = min(max(int(math.exp(math.log(640 / q_size))), 2), 64)
            Ua, Sa, Va = torch.svd_lowrank(a, q=q_size, niter=iters)

        a_lowrank = torch.mm(Ua, torch.mm(torch.diag(Sa), Va.t()))
        a_diff = a - a_lowrank

        # Project the difference onto the space spanned by Qb (orthogonal basis of B)
        a_diff_projected = torch.mm(Qb, torch.mm(Qb.t(), a_diff))

        return b + a_diff_projected

    @staticmethod
    @convert_to_recipe
    def decompose_merge(
            a: Tensor | SameMergeSpace,
            b: Tensor | SameMergeSpace,
            *,
            alpha: Hyper = 1.0,
            **kwargs,
    ) -> Tensor | SameMergeSpace:
        """
        Merges tensors 'a' and 'b' by decomposing 'b' using SVD, aligning the difference
        between 'b' and its low-rank approximation to 'a', and adding the scaled result to 'a'.

        Args:
            a (Tensor): The first tensor, serving as the base model.
            b (Tensor): The second tensor, whose components will be blended into 'a'.
            alpha (float): The scaling factor for the projected difference, controlling the strength of 'b's
                           influence on the merged result (0 <= alpha <= 1).
            **kwargs: Additional keyword arguments.

        Returns:
            Tensor: The merged tensor.
        """
        # Reshape tensors to 2D
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

        a = a.reshape(shape_2d)
        b = b.reshape(shape_2d)

        # Perform the core merging operation
        res = MergeMethods.decompose_align(a, b, alpha)

        return res.reshape(original_shape)

    def decompose_align(a, b, alpha):
        """
        Performs the core merging operation using SVD and orthogonal projection.

        Args:
            a (Tensor): The base tensor (2D).
            b (Tensor): The tensor whose components will be blended into 'a' (2D).
            alpha (float): The scaling factor for the projected difference.

        Returns:
            Tensor: The merged tensor (2D).
        """
        Ua, Sa, Va = torch.linalg.svd(a, full_matrices=False, driver="gesvd")
        Ub, Sb, Vb = torch.linalg.svd(b, full_matrices=False, driver="gesvd")

        # Reconstruct a low-rank approximation of B using the singular values from A
        b_lowrank = torch.mm(Ub, torch.mm(torch.diag(Sa), Vb))
        b_diff = b - b_lowrank

        # Project the difference (B - B_lowrank) onto the space spanned by Ua (orthogonal basis of A)
        b_diff_projected = torch.mm(Ua, torch.mm(Ua.t(), b_diff))

        # Add the scaled projected difference to A to create the merged tensor
        return a + b_diff_projected * alpha

    @staticmethod
    @convert_to_recipe(volatile_hypers=["cache"])
    def svd_replace_merge(
            a: Tensor | SameMergeSpace,
            b: Tensor | SameMergeSpace,
            *,
            alpha: Hyper = 1.0,
            cache: Optional[Dict[str, Dict[str, Tensor]]] = None,
            **kwargs,
    ) -> Tensor | SameMergeSpace:
        """
        Merges tensors 'a' and 'b' using Singular Value Decomposition (SVD) by replacing the singular values
        of 'b' with those from 'a' and reconstructing 'b' using the modified singular values.

        Args:
            a (Tensor): The first tensor, whose singular values will be used to modify 'b'.
            b (Tensor): The second tensor, whose structure will be retained but modified with 'a's singular values .
            alpha (float): The interpolation factor between the original tensor 'b' and the merged tensor (0 <= alpha <= 1).
            cache: Cache svd results to reuse and skip computation in subsequent iterations.
            **kwargs: Additional keyword arguments.

        Returns:
            Tensor: The merged tensor.
        """
        # Reshape tensors to 2D
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

        a = a.reshape(shape_2d)
        b = b.reshape(shape_2d)

        res = MergeMethods.SVD_Replace(a, b, alpha, cache=cache, **kwargs)

        return res.reshape(original_shape)

    def SVD_Replace(a, b, alpha, cache: Optional[Dict[str, Dict[str, Tensor]]] = None, **kwargs):
        """
        Performs the core merging operation using SVD, with caching for optimization.

        Args:
            a (Tensor): The source tensor (2D), providing the singular values.
            b (Tensor): The target tensor (2D), whose structure is retained.
            alpha (float): The interpolation factor.
            cache:  A dictionary for caching SVD results.

        Returns:
            Tensor: The merged tensor (2D).
        """

        # Check if merged_tensor is already cached BEFORE performing SVD
        if cache is not None:
            key = kwargs["key"]
            if key not in cache:
                cache[key] = {}
            cache = cache[key]

        if cache is not None and "merged_tensor" in cache:
            merged_tensor = cache["merged_tensor"].to(a.device, a.dtype)
        else:
            # Determine the SVD driver based on CUDA availability
            svd_driver = "gesvdj" if a.is_cuda else "gesvd"
            Ua, Sa, Va = torch.linalg.svd(a, full_matrices=False, driver=svd_driver)
            Ub, Sb, Vb = torch.linalg.svd(b, full_matrices=False, driver=svd_driver)

            # Reconstruct 'b' using the singular values from 'a' (Vb is already transposed)
            merged_tensor = torch.mm(Ub, torch.mm(torch.diag(Sa), Vb))
            if cache is not None:
                cache["merged_tensor"] = merged_tensor.to("cpu")

        # Interpolate between the original tensor 'b' and the merged tensor
        return torch.lerp(b, merged_tensor, alpha)

    @staticmethod
    @convert_to_recipe
    def weighted_sum_projection_v2(
            a: Tensor | SameMergeSpace,
            b: Tensor | SameMergeSpace,
            c: Tensor | SameMergeSpace,
            *,
            perplexity: Hyper = 0.0,
            **kwargs,
    ) -> Tensor | SameMergeSpace:

        key = kwargs.get("key", "")
        if key.endswith(("in_proj_weight", "in_proj_bias")):
            vs = []
            for i, k in enumerate(("to_q", "to_k", "to_v")):
                k_kwargs = kwargs.copy()
                k_kwargs["key"] = key.replace("in_proj_", f"{k}.")
                dim = a.shape[0] // 3
                t_start = dim * i
                t_end = dim * (i + 1)
                k_a = a[t_start:t_end]
                k_b = b[t_start:t_end]
                k_c = c[t_start:t_end]
                vs.append(MergeMethods.weighted_sum_projection_v2.__wrapped__(k_a, k_b, k_c, **k_kwargs))
            return torch.cat(vs)

        # Reshape tensors to 2D
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

        a = a.reshape(shape_2d)
        b = b.reshape(shape_2d)
        c = c.reshape(shape_2d)

        ba = b - a
        ca = c - a

        # Calculate alpha values at different levels of granularity
        key_alpha = torch.nan_to_num((ba * ca).sum() / (ba ** 2).sum(), nan=0, posinf=0, neginf=0)
        neuron_alpha = torch.nan_to_num((ba * ca).sum(dim=1, keepdim=True) / (ba ** 2).sum(dim=1, keepdim=True), nan=0,
                                        posinf=0, neginf=0)
        param_alpha = torch.nan_to_num((ba * ca) / (ba ** 2), nan=0, posinf=0, neginf=0)

        # Interpolate between alpha values based on perplexity
        alpha = torch.lerp(torch.lerp(key_alpha, neuron_alpha, 2 * perplexity),
                           torch.lerp(neuron_alpha, param_alpha, 2 * perplexity - 1), perplexity)

        # Perform weighted sum using the interpolated alpha
        return ((1 - alpha) * a + alpha * b).reshape(original_shape)

    @staticmethod
    @convert_to_recipe
    def neuron_train_difference(
        a: Tensor | SameMergeSpace,
        b: Tensor | SameMergeSpace,
        c: Tensor | SameMergeSpace,
        *,
        alpha: Hyper = 1.0,
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
                t_start = dim * i
                t_end = dim * (i + 1)
                k_a = a[t_start:t_end]
                k_b = b[t_start:t_end]
                k_c = c[t_start:t_end]
                vs.append(MergeMethods.neuron_train_difference.__wrapped__(k_a, k_b, k_c, **k_kwargs))
            return torch.cat(vs)

        if key.endswith("bias"):
            return sd_mecha.merge_methods.weighted_sum.__wrapped__(a, b, alpha=alpha)

        # Reshape tensors to 2D
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

        a = a.reshape(*shape_2d)
        b = b.reshape(*shape_2d)
        c = c.reshape(*shape_2d)

        threshold = torch.maximum((a - c).norm(dim=1, keepdim=True), (b - c).norm(dim=1, keepdim=True))
        dissimilarity = (1 - torch.nan_to_num(((a - c) * (b - c)).sum(dim=1, keepdim=True) / threshold**2, nan=0)) / 2

        res = a + (b - c) * alpha * dissimilarity
        return res.reshape(original_shape)

