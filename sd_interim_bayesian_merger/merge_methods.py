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
from scipy.linalg import sqrtm
from scipy.stats import binom, rankdata
from torch import Tensor, polar
from typing import Optional, Callable, Dict, Tuple, TypeVar, Generic, get_type_hints, get_origin, Union, get_args, List, Set, Iterable
from pytorch_wavelets import DWTForward, DWTInverse
from sd_mecha import Hyper, MergeSpace
from sd_mecha.merge_methods import SameMergeSpace
from sd_mecha.extensions.merge_method import LiftFlag, convert_to_recipe
from sd_interim_bayesian_merger.mm_docs import MERGE_METHOD_DOCS
from torchvision.models.video import S3D

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
    def ties_sum_extended(*models, k: Hyper, apply_stock: Hyper = 0.0, apply_median: Hyper = 1.0, eps: Hyper = 1e-5, ftol: Hyper = 1e-11, maxiter: Hyper = 150, **kwargs):
        return sd_mecha.ties_sum_extended(*models, k=k, apply_stock=apply_stock, apply_median=apply_median, eps=eps, ftol=ftol, maxiter=maxiter, **kwargs)

    @staticmethod
    @convert_to_recipe
    def ties_sum_with_dropout(
            *models: Tensor | LiftFlag[MergeSpace.DELTA],
            probability: Hyper = 0.9,
            della_eps: Hyper = 0.0,
            rescale: Hyper = 1.0,
            k: Hyper = 0.218,
            vote_sgn: Hyper = 0,
            apply_stock: Hyper = 0.0,
            cos_eps: Hyper = 1e-6,
            apply_median: Hyper = 1.0,
            eps: Hyper = 1e-5,
            maxiter: Hyper = 150,
            ftol: Hyper = 1e-11,
            seed: Hyper = 218,
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
        """Creates a dropout mask using smooth magnitude-based probabilities, following DELLA intuition."""
        p_min = torch.full(delta.shape, 1 - probability, device=delta.device, dtype=delta.dtype)
        if della_eps != 0.0:
            magnitudes = delta.abs()
            # Invert magnitudes so larger values get lower dropout probabilities
            inv_magnitudes = 1.0 / (magnitudes + 1e-8)
            # Use softmax for smooth distribution
            norm_probs = torch.softmax(inv_magnitudes.flatten() / della_eps, dim=0).reshape(delta.shape)
            delta_i = (norm_probs - norm_probs.mean()) * della_eps
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
    def multi_domain_alignment(
            a: Tensor | SameMergeSpace,
            b: Tensor | SameMergeSpace,
            c: Tensor | SameMergeSpace,
            *,
            alpha: float = 0.5,
            beta: float = 0.5,
            kernel_size: int = 5,
            frequency_weight: float = 0.218,
            use_cross_attention: bool = True,
            **kwargs,
    ) -> Tensor | SameMergeSpace:
        """
        Enhanced multi-domain model merging that better balances frequency and spatial information.

        Key improvements:
        1. More stable frequency domain processing
        2. Adaptive interpolation based on feature importance
        3. Better handling of different tensor types (attention, conv, linear)
        """
        try:
            with torch.no_grad():
                # Special handling for attention projections
                key = kwargs.get("key", "")
                if key.endswith(("in_proj_weight", "in_proj_bias")):
                    return MergeMethods.handle_attention_projection(a, b, c, alpha, beta, kwargs)

                original_shape = a.shape

                # Step 1: Process in frequency domain with stability checks
                freq_aligned = torch.utils.checkpoint.checkpoint(
                    MergeMethods.frequency_alignment,
                    a, b, c,
                    frequency_weight
                )

                # Step 2: Reshape for spatial processing
                shape_2d = MergeMethods.determine_reshape_dimensions(a)
                a_2d = a.reshape(*shape_2d)
                b_2d = b.reshape(*shape_2d)
                c_2d = c.reshape(*shape_2d)
                freq_2d = freq_aligned.reshape(*shape_2d)

                # Step 3: Calculate importance weights
                if use_cross_attention and min(shape_2d) > 1:
                    importance = MergeMethods.calculate_feature_importance(
                        a_2d.detach(), b_2d.detach(), c_2d.detach()
                    )
                else:
                    importance = torch.ones_like(a_2d)

                # Step 4: Calculate adaptive dissimilarity
                dissimilarity = MergeMethods.calculate_adaptive_dissimilarity(
                    a_2d.detach(), b_2d.detach(), c_2d.detach(),
                    kernel_size
                )

                # Step 5: Combine frequency and spatial information adaptively
                combined_weights = freq_2d * frequency_weight + b_2d * (1 - frequency_weight)

                # Step 6: Enhanced SLERP interpolation
                effective_alpha = alpha * importance
                dot_product = (F.normalize(a_2d, dim=-1) * F.normalize(combined_weights, dim=-1)).sum(dim=-1,
                                                                                                      keepdim=True).clamp(
                    -1.0, 1.0)
                omega = torch.acos(dot_product)

                # Improved stability for small angles
                small_angle_mask = omega < 1e-4
                sin_omega = torch.sin(omega).clamp_min(1e-6)

                # Compute interpolation coefficients
                coeff_a = torch.where(small_angle_mask,
                                      1.0 - effective_alpha,
                                      torch.sin((1.0 - effective_alpha) * omega) / sin_omega)
                coeff_b = torch.where(small_angle_mask,
                                      effective_alpha,
                                      torch.sin(effective_alpha * omega) / sin_omega)

                # Merge with magnitude preservation
                a_magnitude = a_2d.norm(dim=-1, keepdim=True)
                b_magnitude = combined_weights.norm(dim=-1, keepdim=True)
                target_magnitude = a_magnitude * (1 - effective_alpha) + b_magnitude * effective_alpha

                merged = coeff_a * a_2d + coeff_b * combined_weights
                merged = merged * (target_magnitude / (merged.norm(dim=-1, keepdim=True) + 1e-8))

                # Apply anchor adjustment with adaptive scaling
                if beta > 0:
                    anchor_scale = beta * torch.sigmoid(dissimilarity)  # Smooth scaling
                    anchor_delta = (combined_weights - c_2d) * anchor_scale
                    merged = merged + anchor_delta * importance

                return merged.reshape(original_shape)

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def handle_attention_projection(
            a: Tensor,
            b: Tensor,
            c: Tensor,
            alpha: float,
            beta: float,
            kwargs: dict
    ) -> Tensor:
        """Handle the special case of attention projection layers."""
        vs = []
        for i, k in enumerate(("to_q", "to_k", "to_v")):
            k_kwargs = kwargs.copy()
            k_kwargs["key"] = kwargs["key"].replace("in_proj_", f"{k}.")
            dim = a.shape[0] // 3
            t_start = dim * i
            t_end = dim * (i + 1)
            vs.append(
                MergeMethods.multi_domain_alignment.__wrapped__(
                    a[t_start:t_end],
                    b[t_start:t_end],
                    c[t_start:t_end],
                    alpha=alpha,
                    beta=beta,
                    **k_kwargs
                )
            )
        return torch.cat(vs)

    def determine_reshape_dimensions(tensor: Tensor) -> tuple:
        """Determine the appropriate reshape dimensions based on tensor type."""
        if not tensor.shape:
            return (1, 1)

        is_conv = len(tensor.shape) == 4
        if is_conv:
            return (-1, functools.reduce(operator.mul, tensor.shape[1:]))
        return (-1, tensor.shape[-1])

    def calculate_cross_attention(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
        """Calculate feature importance using cross-attention mechanism."""
        # Normalize inputs
        a_norm = F.normalize(a, dim=-1)
        b_norm = F.normalize(b, dim=-1)
        c_norm = F.normalize(c, dim=-1)

        # Calculate attention scores
        attn_ab = torch.matmul(a_norm, b_norm.transpose(-2, -1))
        attn_ac = torch.matmul(a_norm, c_norm.transpose(-2, -1))
        attn_bc = torch.matmul(b_norm, c_norm.transpose(-2, -1))

        # Softmax for probability distribution
        attn_ab = F.softmax(attn_ab / math.sqrt(a.size(-1)), dim=-1)
        attn_ac = F.softmax(attn_ac / math.sqrt(a.size(-1)), dim=-1)
        attn_bc = F.softmax(attn_bc / math.sqrt(a.size(-1)), dim=-1)

        # Calculate feature importance based on attention patterns
        importance = (
                             torch.sum(attn_ab, dim=-1, keepdim=True) +
                             torch.sum(attn_ac, dim=-1, keepdim=True) +
                             torch.sum(attn_bc, dim=-1, keepdim=True)
                     ) / 3.0

        # Normalize importance scores
        importance = F.normalize(importance, dim=0)
        return importance

    def calculate_dissimilarity(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
        """Calculate dissimilarity between tensors with anchor guidance."""
        diff_a_c = a - c
        diff_b_c = b - c

        norm_a = diff_a_c.norm(dim=1, keepdim=True)
        norm_b = diff_b_c.norm(dim=1, keepdim=True)

        # Use maximum of norms for normalization
        threshold = torch.max(norm_a, norm_b)

        # Calculate cosine similarity with improved numerical stability
        cos_sim = torch.nan_to_num(
            (diff_a_c * diff_b_c).sum(dim=1, keepdim=True) / (threshold ** 2 + EPSILON),
            nan=0
        )

        return (1 - cos_sim) / 2

    def frequency_alignment(a: Tensor, b: Tensor, c: Tensor, weight: float) -> Tensor:
        """Enhanced frequency domain alignment with better stability."""
        # Convert to frequency domain
        a_freq = torch.fft.rfft(a.reshape(-1))
        b_freq = torch.fft.rfft(b.reshape(-1))

        # Calculate magnitudes and phases
        a_mag = torch.abs(a_freq)
        b_mag = torch.abs(b_freq)
        b_phase = torch.angle(b_freq)

        # Adaptive magnitude mixing
        mixed_mag = a_mag * (1 - weight) + b_mag * weight

        # Reconstruct with mixed magnitudes and original phase
        result_freq = torch.polar(mixed_mag, b_phase)
        return torch.fft.irfft(result_freq, a.reshape(-1).shape[0]).reshape(a.shape)

    def calculate_adaptive_dissimilarity(a: Tensor, b: Tensor, c: Tensor, kernel_size: int) -> Tensor:
        """Calculate dissimilarity with adaptive kernel size."""
        # Normalize inputs for stable distance calculation
        a_norm = F.normalize(a, dim=-1)
        b_norm = F.normalize(b, dim=-1)
        c_norm = F.normalize(c, dim=-1)

        # Calculate pairwise distances
        dist_ab = 1 - (a_norm * b_norm).sum(dim=-1, keepdim=True)
        dist_ac = 1 - (a_norm * c_norm).sum(dim=-1, keepdim=True)
        dist_bc = 1 - (b_norm * c_norm).sum(dim=-1, keepdim=True)

        # Combine distances with smooth min/max
        dissimilarity = (dist_ab + dist_ac + dist_bc) / 3

        # Apply adaptive smoothing
        return MergeMethods.gaussian_smooth(dissimilarity, kernel_size)

    def gaussian_smooth(a: Tensor, kernel_size: int) -> Tensor:
        """
        Apply 1D Gaussian blur to tensor with handling for small tensors.
        Automatically adjusts kernel size for small inputs to prevent padding errors.
        """
        # Ensure input is at least 2D
        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        # Adjust kernel size if it's too large for the input
        min_dim = min(a.shape)
        if kernel_size > min_dim:
            # Use the largest odd number that's smaller than the minimum dimension
            kernel_size = max(3, min_dim - (min_dim % 2 == 0))

        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size -= 1

        # Skip blur for very small tensors
        if kernel_size < 3:
            return a.squeeze() if len(a.shape) > 1 else a

        sigma = kernel_size / 3
        x = torch.arange(kernel_size, device=a.device) - (kernel_size - 1) / 2
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, -1)

        pad_size = kernel_size // 2
        # Use replication padding for very small tensors where reflection wouldn't work
        padding_mode = 'replicate' if min_dim <= pad_size * 2 else 'reflect'

        padded = F.pad(a.unsqueeze(1), (pad_size, pad_size), mode=padding_mode)
        blurred = F.conv1d(padded.double(), kernel.double()).squeeze(1)

        # Return to original shape
        return blurred.squeeze() if len(a.shape) == 1 else blurred

    def calculate_feature_importance(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
        """Calculate feature importance using attention and gradient information."""
        # Normalize inputs
        a_norm = F.normalize(a, dim=-1)
        b_norm = F.normalize(b, dim=-1)
        c_norm = F.normalize(c, dim=-1)

        # Calculate attention scores
        attn_ab = torch.matmul(a_norm, b_norm.transpose(-2, -1))
        attn_ac = torch.matmul(a_norm, c_norm.transpose(-2, -1))

        # Calculate feature importance
        importance = (
                             torch.sum(torch.abs(attn_ab), dim=-1) +
                             torch.sum(torch.abs(attn_ac), dim=-1)
                     ) / 2

        # Normalize importance scores
        return F.softmax(importance, dim=-1).unsqueeze(-1)

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
            Qb, _ = torch.linalg.qr(b)
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

    @staticmethod
    @convert_to_recipe
    def polar_interpolate(
        a: Tensor | SameMergeSpace,
        b: Tensor | SameMergeSpace,
        *,
        alpha: Hyper = 0.5,
        polar_skew: Hyper = 0,
        **kwargs,
    ) -> Tensor | SameMergeSpace:
        gc.collect()
        torch.cuda.empty_cache()
        key = kwargs.get("key", "")
        # if not key.startswith("model.diffusion_model.input_blocks") or (len(a.shape) >= 2 and a.shape[0] == a.shape[1:].numel()):
        #     return a # 'model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_v.weight'

        if key.endswith(("in_proj_weight", "in_proj_bias")):
            # workaround for concatenated attention projection layers
            vs = []
            for i, k in enumerate(("to_q", "to_k", "to_v")):
                k_kwargs = kwargs.copy()
                k_kwargs["key"] = key.replace("in_proj_", f"{k}.")
                dim = a.shape[0] // 3
                t_start = dim*i
                t_end = dim*(i+1)
                k_models = tuple(m[t_start:t_end] for m in (a, b))
                vs.append(MergeMethods.polar_interpolate.__wrapped__(*k_models, alpha=alpha, **k_kwargs))
            return torch.cat(vs)

        if torch.allclose(a.half(), b.half()):
            return a*(1-alpha) + b*alpha

        layer_type = MergeMethods.get_layer_type(a.shape, kwargs)

        if layer_type is MergeMethods.LayerType.SCALAR:
            print(f"geometric interpolation of norm: {key}")
            res = a.to(a.dtype.to_complex())**(1-alpha) * b.to(a.dtype.to_complex())**alpha
            res = res.real + res.imag * torch.where(a.abs() >= b.abs(), a.sign(), b.sign())
        elif layer_type is MergeMethods.LayerType.CONV2D:
            print(f"fourrier interpolation of conv: {key}")
            res = MergeMethods.interpolate_conv_kernels(a, b, alpha, polar_skew, kwargs)
        elif layer_type is MergeMethods.LayerType.EMBEDD:
            # print(f"distribution interpolation of embeds: {key}")
            # res = distribution_interpolate(a, b, alpha, kwargs)
            # print(f"matmul interpolation of embeds: {key}")
            # res = matmul_interpolate(a.mH, b.mH, alpha, polar_skew, kwargs).mH
            print(f"linear interpolation of embeds: {key}")
            res = a*(1-alpha) + b*alpha
        elif layer_type is MergeMethods.LayerType.MATMUL:
            print(f"polar interpolation of matmul: {key}")
            res = MergeMethods.matmul_interpolate(a, b, alpha, polar_skew, kwargs)
        elif layer_type is MergeMethods.LayerType.OFFSET:
            print(f"linear interpolation of bias: {key}")
            res = a*(1-alpha) + b*alpha
        else:
            print(f"linear interpolation of unknown: {key}")
            res = a*(1-alpha) + b*alpha

        assert res.isfinite().all(), "bad merge: the model will contain nans or infs"
        return res

    class LayerType(enum.Enum):
        SCALAR = enum.auto()
        OFFSET = enum.auto()
        CONV2D = enum.auto()
        EMBEDD = enum.auto()
        MATMUL = enum.auto()

    def get_layer_type(shape, kwargs):
        key = kwargs["key"]

        if len(shape) == 1 and key.endswith(("weight", "scale")):
            return MergeMethods.LayerType.SCALAR

        if len(shape) < 2 or key.endswith("bias") or "position" in key:
            return MergeMethods.LayerType.OFFSET

        if len(shape) == 4:
            return MergeMethods.LayerType.CONV2D

        if "token_embedding" in key or "shared.weight" in key:
            return MergeMethods.LayerType.EMBEDD

        return MergeMethods.LayerType.MATMUL

    def matmul_interpolate(a_2d, b_2d, alpha, polar_skew, kwargs):
        original_dtype = a_2d.dtype

        m, n = a_2d.shape[-2:]

        svd_driver = "gesvd" if a_2d.is_cuda else None
        a_u, a_s, a_vt = torch.linalg.svd(a_2d, driver=svd_driver, full_matrices=False)
        b_u, b_s, b_vt = torch.linalg.svd(b_2d, driver=svd_driver, full_matrices=False)
        if not original_dtype.is_complex:
            MergeMethods.align_v4(a_u, b_u, a_s, b_s, a_vt, b_vt)

        a_v, b_v, v_proj = MergeMethods.close_ortho_columns_full(a_vt.mH, b_vt.mH)
        a_s1, b_s1 = a_s, b_s
        v_is_tall = m < n
        if v_is_tall:
            a_s1, b_s1 = torch.cat((a_s, b_s), dim=-1)[..., :min(2*m, n)], torch.cat((b_s, a_s), dim=-1)[..., :min(2*m, n)]

        p_right = MergeMethods.positive_definite_interpolate(a_s1, a_v, b_s1, b_v, alpha, polar_skew, kwargs)

        a_u2, b_u2, u2_proj = MergeMethods.close_ortho_columns_full(a_u, b_u)
        a_s2, b_s2 = a_s, b_s
        u_is_tall = n < m
        if u_is_tall:
            a_s2, b_s2 = torch.cat((a_s, b_s), dim=-1)[..., :min(2*n, m)], torch.cat((b_s, a_s), dim=-1)[..., :min(2*n, m)]

        p_left = MergeMethods.positive_definite_interpolate(a_s2, a_u2, b_s2, b_u2, alpha, 1-polar_skew, kwargs)

        a_w, b_w = a_u @ a_vt, b_u @ b_vt

        w_is_wide = m < n
        if w_is_wide:
            a_w = a_w.mH
            b_w = b_w.mH

        a_w, b_w, u_proj = MergeMethods.close_ortho_columns_full(a_w, b_w)
        w = MergeMethods.orthogonal_interpolate(a_w, b_w, alpha, kwargs)

        if w_is_wide:
            a_w = a_w.mH
            b_w = b_w.mH
            w = (u_proj @ w).mH

        res = w[..., :m, :n] @ v_proj @ p_right

        if not w_is_wide:
            res = (u_proj @ res)[..., :m, :]

        res = p_left @ u2_proj.mH @ res
        res = u2_proj @ res
        res = res @ v_proj.mH
        return res

    def align_v4(a_u, b_u, a_s, b_s, a_vt, b_vt):
        assert a_u.dim() == 2
        m, n = len(a_u), len(a_vt.mH)
        if m != n:
            return

        a_det_is_negative = MergeMethods.ortho_det(a_u @ a_vt)
        b_det_is_negative = MergeMethods.ortho_det(b_u @ b_vt)
        if a_det_is_negative != b_det_is_negative:
            if a_s[-1] < b_s[-1]:
                a_vt[-1] *= -1
            else:
                b_vt[-1] *= -1

    def orthogonal_interpolate(a, b, alpha, kwargs):
        return MergeMethods.fractional_matrix_power(b @ a.mH, alpha, kwargs) @ a

    def fractional_matrix_power(a, alpha, kwargs):
        original_dtype = a.dtype

        v, vs = torch.linalg.eig(a)
#        MergeMethods.fix_eig_reflections(v, vs)

        v_pow = v**alpha
        if not original_dtype.is_complex:
            assert v_pow.prod().real.sign() == 1

        res = torch.linalg.solve_ex(vs, vs @ torch.diag_embed(v_pow), left=False, check_errors=True)[0]
        if not original_dtype.is_complex and res.imag.abs().max() > 1e-6:
            print(f"fix your signs! max(|imag|): {res.imag.abs().max().item()}, det: {MergeMethods.ortho_det(res).item()}, kwargs: {kwargs}")

        res = res.to(a.dtype)
        return res

    def positive_definite_interpolate(a_v, a_vs, b_v, b_vs, alpha, power, kwargs):
        original_dtype = a_vs.dtype
        complex_dtype = original_dtype.to_complex()

        a_v, a_vs = a_v.to(dtype=complex_dtype), a_vs.to(dtype=complex_dtype)
        b_v, b_vs = b_v.to(dtype=complex_dtype), b_vs.to(dtype=complex_dtype)

        a_half = a_vs @ torch.diag_embed(a_v**(1/2)) @ a_vs.mH
        a_neghalf = a_vs @ torch.diag_embed((a_v**(-1/2)).nan_to_num(nan=0)) @ a_vs.mH
        b = (b_vs @ torch.diag_embed(b_v) @ b_vs.mH).to(complex_dtype)

        delta_v, delta_vs = torch.linalg.eigh(a_neghalf @ b @ a_neghalf)
        delta = delta_vs @ torch.diag_embed((delta_v.to(complex_dtype)**alpha).nan_to_num(nan=0)) @ delta_vs.mH
        res_v, res_vs = torch.linalg.eigh(a_half @ delta @ a_half)
        res_v_pow = (res_v**power).to(complex_dtype)
        res = res_vs @ torch.diag_embed(res_v_pow) @ res_vs.mH

        if not original_dtype.is_complex and res.imag.abs().max() > 1e-6:
            print(f"fix your signs! max(|imag|): {res.imag.abs().max().item()}, det: {res.det().item()}, kwargs: {kwargs}")

        return res.to(dtype=original_dtype)

    def close_ortho_columns_full(a, b):
        original_dtype = a.dtype
        m, n = a.shape[-2:]
        assert a.shape == b.shape

        if n == m:
            return a, b, MergeMethods.matmul_identity()

        if 2*n < m:
            def complement_proj_fn(a, b, x):
                x = b.mH @ x
                x = b @ x
                c = a.mH @ x
                c = a @ c
                return x - c

            def complement_proj_fn_t(a, b, y):
                y1 = a.mH @ y
                y1 = a @ y1
                y = y - y1
                y = b.mH @ y
                y = b @ y
                return y

            a_n = MergeMethods.extend_ortho(
                a,
                functools.partial(complement_proj_fn, a, b),
                functools.partial(complement_proj_fn_t, a, b),
                m, 2*n,
            )
            b_n = MergeMethods.extend_ortho(
                b,
                functools.partial(complement_proj_fn, b, a),
                functools.partial(complement_proj_fn_t, b, a),
                m, 2*n,
            )
            assert a_n.shape == b_n.shape

            # appending columns aligned in a criss-cross fashion might not be the best approach?
            # the idea is to close the column space of A and B so that both lie in the same vector space.
            # maybe this type of alignment prevents natural rotations?
            to_align = torch.stack([
                a_n[..., n:].mH @ b_n[..., :n],
                b_n[..., n:].mH @ a_n[..., :n],
            ], dim=0)
            svd_driver = "gesvd" if a.is_cuda else None
            u, _, vt = torch.linalg.svd(to_align, driver=svd_driver, full_matrices=False)
            r = u @ vt
            a_n[..., n:] @= r[0]
            b_n[..., n:] @= r[1]

            proj = MergeMethods.get_shared_basis(
                a_n, b_n,
                2*n,
                device=a.device, dtype=a.dtype,
            )
            a_p = proj.mH @ a_n
            b_p = proj.mH @ b_n
            if not original_dtype.is_complex and MergeMethods.ortho_det(b_p @ a_p.mH) == -1:
                proj[-1] *= -1
                a_p = proj.mH @ a_n
                b_p = proj.mH @ b_n
        else:
            def complement_proj_fn(a, x):
                c = a.mH @ x
                c = a @ c
                return x - c

            def complement_proj_fn_t(a, y):
                c = a.mH @ y
                c = a @ c
                return y - c

            a_n = MergeMethods.extend_ortho(
                a,
                functools.partial(complement_proj_fn, a),
                functools.partial(complement_proj_fn_t, a),
                m, m,
            )
            b_n = MergeMethods.extend_ortho(
                b,
                functools.partial(complement_proj_fn, b),
                functools.partial(complement_proj_fn_t, b),
                m, m,
            )
            assert a_n.shape == b_n.shape

            # for 2n >= m, a projection is not useful.
            # we simply add new orthogonal columns until both matrices are square
            #  and align those in A with those in B for minimal interference between
            #  the meaningful columns during interpolation
            to_align = a_n[..., n:].mH @ b_n[..., n:]
            svd_driver = "gesvd" if a.is_cuda else None
            u, _, vt = torch.linalg.svd(to_align, driver=svd_driver, full_matrices=False)
            a_n[..., n:] @= u @ vt

            proj = MergeMethods.matmul_identity()
            a_p = a_n
            b_p = b_n

            if not original_dtype.is_complex and MergeMethods.ortho_det(b_p @ a_p.mH) == -1:
                #a_p = a_n.clone()
                a_p[..., -1] *= -1

        return a_p, b_p, proj

    def ortho_det(a):
        return torch.linalg.eigvals(a).sgn().prod().sgn().to(a.dtype)

    def extend_ortho(x, r, rh, input_m, target_n):
        if target_n <= x.shape[-1]:
            return x

        k = target_n - x.shape[-1]
        k_frame = MergeMethods.get_approximate_basis(r, rh, input_m, k, dtype=x.dtype, device=x.device)
        return torch.cat((x, k_frame), dim=-1)

    def get_shared_basis(a, b, max_rank, niter=2, device=None, dtype=None):
        assert a.shape == b.shape
        m, n = a.shape[-2:]

        def outer_fn(x):
            x = a.mH @ x
            x = b @ x
            return x

        def outer_fn_t(y):
            y = b.mH @ y
            y = a @ y
            return y

        basis = last_basis = None
        rank = max_rank

        while basis is None or (
            MergeMethods.is_orthogonal((basis.mH @ b) @ (a.mH @ basis)) and
            MergeMethods.is_orthogonal((basis.mH @ a) @ (a.mH @ basis)) and
            MergeMethods.is_orthogonal((basis.mH @ b) @ (b.mH @ basis))
        ):
            last_basis = basis
            basis = MergeMethods.get_approximate_basis(
                outer_fn, outer_fn_t,
                m, rank, niter=niter,
                device=device, dtype=dtype,
            )
            rank -= 1

        if rank+2 < max_rank:
            print(f"optimized rank: {rank+2},\tmax case: {max_rank},\tbasis shape: {(basis if last_basis is None else last_basis).shape}")

        return basis if last_basis is None else last_basis

    def is_orthogonal(a, eye=None):
        if eye is None:
            eye = torch.eye(a.shape[-1], device=a.device, dtype=a.dtype)
        return torch.allclose(a.mH @ a, eye)

    def get_approximate_basis(f, fh, input_m: int, rank: int, niter=2, device=None, dtype=None):
        Q = torch.randn(input_m, rank, dtype=dtype, device=device)
        Q = torch.linalg.qr(f(Q)).Q
        for i in range(niter):
            Q = torch.linalg.qr(fh(Q)).Q
            Q = torch.linalg.qr(f(Q)).Q

        return Q

    def fix_eig_reflections(v, vs):
        reflections = torch.isclose(v, -torch.ones_like(v))
        idx = torch.arange(v.shape[-1], device=v.device)[reflections]
        num_reflections = idx.numel()
        if num_reflections % 2 == 1:
            return

        imag_sign_bits = MergeMethods.get_imag_sign_bit(v[..., idx])
        vs = vs[..., idx]

        half_mask = torch.zeros(num_reflections, device=v.device, dtype=torch.bool)
        half_mask[MergeMethods.get_imag_sign_bit(vs[..., 0, :]) == 0] = True

        cost_matrix = (vs[..., half_mask].mT @ vs[..., ~half_mask]).abs()  # do not use .mH since we are matching conjugate pairs
        if cost_matrix.numel():
            conj_idx = cost_matrix.argmax(dim=-1)
        else:
            conj_idx = torch.empty(0, device=v.device, dtype=torch.long)
        assert torch.allclose(vs[..., half_mask], vs[..., ~half_mask][..., conj_idx].conj())

        to_fix_idx = (imag_sign_bits[..., half_mask] == imag_sign_bits[..., ~half_mask][..., conj_idx]).nonzero().flatten()
        to_fix = v[..., to_fix_idx]
        to_fix.conj_physical_()

    def get_imag_sign_bit(tensor):
        int_dtype = MergeMethods.complex_int_dtype_map[tensor.dtype]
        int_tensor = tensor.imag.view(int_dtype)
        last_bit = int_dtype.itemsize * 8 - 1
        sign_bits = (int_tensor >> last_bit) & 1
        return sign_bits

    complex_int_dtype_map = {
        torch.complex32: torch.int16,
        torch.complex64: torch.int32,
        torch.complex128: torch.int64,
    }

    def vector_slerp(a, b, alpha):
        original_shape = a.shape
        a_2d = a.flatten(start_dim=1)
        b_2d = b.flatten(start_dim=1)

        m, n = a_2d.shape
        if m > n:
            # assume more vectors than features
            a_2d = a_2d.mH
            b_2d = b_2d.mH

        res = MergeMethods.batch_slerp(a_2d, b_2d, alpha)
        if m > n:
            res = res.mH

        return res.reshape(original_shape)

    def batch_slerp(a, b, t):
        a_norm = a.norm(dim=0, keepdim=True)
        b_norm = b.norm(dim=0, keepdim=True)
        a_normalized = a / a_norm
        b_normalized = b / b_norm

        dot_product = (a_normalized * b_normalized.conj()).sum(dim=0, keepdim=True)

        if not dot_product.is_complex():
            dot_product = torch.clip(dot_product, -1, 1)

        theta = torch.arccos(dot_product)

        sin_theta = torch.sin(theta)
        slerp_vector = (torch.sin((1 - t) * theta) / sin_theta) * a_normalized + (torch.sin(t * theta) / sin_theta) * b_normalized
        slerp_vector *= a_norm**(1-t) * b_norm**t

        return torch.where(slerp_vector.isfinite(), slerp_vector, a*(1-t) + b*t)

    def interpolate_conv_kernels(a, b, alpha, polar_skew, kwargs):
        original_shape = a.shape

        a_dft = torch.fft.rfft2(a, norm="ortho")
        b_dft = torch.fft.rfft2(b, norm="ortho")

        a_dft = a_dft.permute(2, 3, 0, 1)
        b_dft = b_dft.permute(2, 3, 0, 1)

        permuted_shape = a_dft.shape
        a_dft = a_dft.flatten(end_dim=1)
        b_dft = b_dft.flatten(end_dim=1)

        res_dft = []
        for i in range(len(a_dft)):
            res_dft.append(MergeMethods.matmul_interpolate(a_dft[i], b_dft[i], alpha, polar_skew, kwargs))

        res_dft = torch.stack(res_dft)
        res_dft = res_dft.reshape(permuted_shape)

        res_dft = res_dft.permute(2, 3, 0, 1)

        res = torch.fft.irfft2(res_dft, s=original_shape[-2:], norm="ortho")
        return res.to(a.dtype)

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

        a_corr_neghalf = MergeMethods.fractional_pd(a_corr, -1/2)
        b_corr_neghalf = MergeMethods.fractional_pd(b_corr, -1/2)

        a_w_transform = a_corr_neghalf @ a_d_neghalf
        b_w_transform = b_corr_neghalf @ b_d_neghalf

        a_w = a_w_transform @ (a - a_mean)
        b_w = b_w_transform @ (b - b_mean)

        res_w = a_w*(1 - alpha) + b_w*alpha
        res_centered = MergeMethods.positive_definite_interpolate(
            *torch.linalg.eigh(MergeMethods.fractional_pd(a_w_transform, -1)),
            *torch.linalg.eigh(MergeMethods.fractional_pd(b_w_transform, -1)),
            alpha=alpha,
            power=1,
            kwargs=kwargs,
        ) @ res_w
        res = res_centered + a_mean*(1-alpha) + b_mean*alpha
        res = res.mH.contiguous()

        return res

    def fractional_pd(a, power):
        v, vs = torch.linalg.eigh(a)
        v_pow = v**power
        res = vs @ torch.diag_embed(v_pow) @ vs.mH
        return res.to(dtype=a.dtype)

    class matmul_identity:
        def __matmul__(self, other):
            return other

        def __rmatmul__(self, other):
            return other

        @property
        def mH(self):
            return self

        @property
        def mT(self):
            return self

        @property
        def H(self):
            return self

        @property
        def T(self):
            return self

    def wavelet_packet_decomposition(x: Tensor, wavelet: str, level: int) -> List[Tensor]:
        # Ensure the input is 2D
        original_shape = x.shape
        x_2d = x.view(-1, x.shape[-1] if len(x.shape) > 1 else 1)

        # Convert to numpy for PyWavelets
        x_np = x_2d.cpu().numpy()

        coeffs = []
        for row in x_np:
            wp = pywt.WaveletPacket(data=row, wavelet=wavelet, mode='symmetric', maxlevel=level)
            coeffs.append([wp[node.path].data for node in wp.get_level(level, 'natural')])

        # Convert back to torch tensors
        return [torch.tensor(np.array(c), device=x.device, dtype=x.dtype) for c in zip(*coeffs)]

    def wavelet_packet_reconstruction(coeffs: List[Tensor], wavelet: str, original_shape: torch.Size) -> Tensor:
        # Reconstruct each row
        reconstructed_rows = []
        for i in range(coeffs[0].shape[0]):
            wp = pywt.WaveletPacket(data=None, wavelet=wavelet, mode='symmetric', maxlevel=len(coeffs))
            for j, coeff in enumerate(coeffs):
                wp[wp.get_leaf_nodes()[j].path] = coeff[i].cpu().numpy()
            reconstructed_rows.append(wp.reconstruct(update=False))

        # Combine reconstructed rows
        reconstructed = torch.tensor(np.array(reconstructed_rows), device=coeffs[0].device, dtype=coeffs[0].dtype)

        # Reshape to original shape
        return reconstructed.view(original_shape)

    def simple_embedding_merge(a: Tensor, b: Tensor, alpha: float = 0.5) -> Tensor:
        a_mean, a_std = a.mean(dim=1, keepdim=True), a.std(dim=1, keepdim=True)
        b_mean, b_std = b.mean(dim=1, keepdim=True), b.std(dim=1, keepdim=True)
        a_norm = (a - a_mean) / a_std
        b_norm = (b - b_mean) / b_std
        merged_norm = (1 - alpha) * a_norm + alpha * b_norm
        merged_mean = (1 - alpha) * a_mean + alpha * b_mean
        merged_std = (1 - alpha) * a_std + alpha * b_std
        return merged_norm * merged_std + merged_mean

    @staticmethod
    @convert_to_recipe
    def merge_layers(
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
                vs.append(MergeMethods.merge_layers.__wrapped__(k_a, k_b, **k_kwargs))
            return torch.cat(vs)

        layer_type = MergeMethods.get_layer_type(a.shape, kwargs)

        if layer_type == MergeMethods.LayerType.SCALAR:
            return MergeMethods.geometric_sum_full.__wrapped__(a, b, alpha=alpha)
        elif layer_type == MergeMethods.LayerType.OFFSET:
            return sd_mecha.weighted_sum.__wrapped__(a, b, alpha=alpha)
        elif layer_type == MergeMethods.LayerType.EMBEDD:
            return MergeMethods.advanced_embedding_merge(a, b, alpha=alpha)
        elif layer_type == MergeMethods.LayerType.MATMUL:
            if MergeMethods.is_positive_definite(a) and MergeMethods.is_positive_definite(b):
                return MergeMethods.log_euclidean(a, b, alpha=alpha)
            else:
                if (MergeMethods.matrix_is_large(a) and not MergeMethods.dominant_rotation(a)) or MergeMethods.matrix_is_ill_conditioned(a):
                    return MergeMethods.svd_interpolation(a, b, alpha=alpha)
                else:
                    return MergeMethods.polar_decomposition(a, b, alpha=alpha)
        elif layer_type == MergeMethods.LayerType.CONV2D:
            return MergeMethods.merge_wavelets(a, b, alpha=alpha)
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

    @convert_to_recipe
    def geometric_sum_full(
        a: Tensor | SameMergeSpace,
        b: Tensor | SameMergeSpace,
        *,
        alpha: Hyper = 0.5,
        **kwargs,
    ) -> Tensor | SameMergeSpace:
        a = torch.complex(a, torch.zeros_like(a))
        b = torch.complex(b, torch.zeros_like(b))
        res = a ** (1 - alpha) * b ** alpha
        return res.real

    def svdeez_low_rank(a: Tensor, b: Tensor, alpha: float, align: bool = False) -> Tensor:
        """
        Merges two matmul layers (represented by matrices a and b) using low-rank SVD
        and optional alignment.

        Args:
            a: The first matmul matrix.
            b: The second matmul matrix.
            alpha: The interpolation factor (0 <= alpha <= 1).
            align: Whether to perform singular vector alignment (default: False).

        Returns:
            The merged matmul matrix.
        """
        a_u, a_s, a_v = sd_mecha.merge_methods.svd.torch_svd_lowrank(a)
        b_u, b_s, b_v = sd_mecha.merge_methods.svd.torch_svd_lowrank(b)

        if align:
            transform = sd_mecha.merge_methods.svd.orthogonal_procrustes(a_u, b_u)
            b_u = b_u @ transform
            b_v = b_v @ transform

        merged_s = (1 - alpha) * a_s + alpha * b_s
        merged_u = (1 - alpha) * a_u + alpha * b_u
        merged_v = (1 - alpha) * a_v + alpha * b_v

        return merged_u @ torch.diag(merged_s) @ merged_v.t()

    def merge_wavelets(a: Tensor, b: Tensor, alpha: float, wave: str = 'db3',
                       levels: int = None) -> Tensor:
        """
        Merges two convolutional layers using a multi-level wavelet transform
        while attempting to preserve original sizes.

        Args:
        - a, b: Input tensors (convolutional kernels)
        - alpha: Blending factor (0 to 1)
        - wave: Wavelet to use (default: 'db3')
        - levels: Number of decomposition levels (default: None, which means adaptive)
        """
        original_size = a.shape

        # Pad the input tensors to the nearest power of 2
        max_size = max(a.shape[2], a.shape[3])
        target_size = 2 ** (max_size - 1).bit_length()
        pad_size = target_size - max_size

        a_padded = F.pad(a, (0, pad_size, 0, pad_size))
        b_padded = F.pad(b, (0, pad_size, 0, pad_size))

        # Determine the number of levels if not specified
        if levels is None:
            levels = min(4, (target_size - 1).bit_length() - 1)  # Adaptive J

        # Initialize wavelet transform
        dwt = DWTForward(J=levels, wave=wave, mode='zero')
        idwt = DWTInverse(wave=wave, mode='zero')
        dwt = dwt.to(device=a.device, dtype=a.dtype)
        idwt = idwt.to(device=a.device, dtype=a.dtype)

        # Perform forward DWT
        a_ll, a_h = dwt(a_padded)
        b_ll, b_h = dwt(b_padded)

        # Merge the low-frequency components
        merged_ll = alpha * a_ll + (1 - alpha) * b_ll

        # Merge the high-frequency components
        merged_h = []
        for a_h_level, b_h_level in zip(a_h, b_h):
            merged_h_level = alpha * a_h_level + (1 - alpha) * b_h_level
            merged_h.append(merged_h_level)

        # Perform inverse DWT
        merged = idwt((merged_ll, merged_h))

        # Crop back to original size
        merged = merged[:, :, :original_size[2], :original_size[3]]

        return merged

    def polar_decompose(A: Tensor, B: Tensor, alpha: float) -> Tensor:
        """
        Merges two matmul layers using polar decomposition.
        """
        # Apply torch.polar directly to the matrices
        polarA = torch.polar(torch.abs(A), torch.angle(A))
        polarB = torch.polar(torch.abs(B), torch.angle(B))

        # Extract absolute values and angles
        QA, PA = torch.abs(polarA), torch.angle(polarA)
        QB, PB = torch.abs(polarB), torch.angle(polarB)

        merged_Q = sd_mecha.slerp.__wrapped__(QA, QB, alpha=alpha)
        merged_P = (1 - alpha) * PA + alpha * PB

        return merged_Q @ merged_P

    def truncated_svd(a: torch.Tensor, b: torch.Tensor, alpha: float, rank: int) -> torch.Tensor:
        """
        Merges two matmul layers using truncated SVD.

        Args:
            a: The first matmul matrix.
            b: The second matmul matrix.
            alpha: The interpolation factor (0 <= alpha <= 1).
            rank: The truncation rank for SVD.

        Returns:
            The merged matmul matrix.
        """
        device = a.device  # Get the device of the input tensors

        # Compute truncated SVD for both matrices
        U_a, S_a, V_a = torch.linalg.svd(a, full_matrices=False)
        U_b, S_b, V_b = torch.linalg.svd(b, full_matrices=False)

        # Truncate the singular values and vectors
        U_a = U_a[:, :rank]
        S_a = S_a[:rank]
        V_a = V_a[:rank, :]
        U_b = U_b[:, :rank]
        S_b = S_b[:rank]
        V_b = V_b[:rank, :]

        # Interpolate the truncated components
        merged_S = (1 - alpha) * S_a + alpha * S_b
        merged_U = (1 - alpha) * U_a + alpha * U_b
        merged_V = (1 - alpha) * V_a + alpha * V_b

        # Reconstruct the merged matrix
        merged_matrix = merged_U @ torch.diag(merged_S).to(device) @ merged_V

        return merged_matrix

    def log_euclidean(a: torch.Tensor, b: torch.Tensor, alpha: float) -> torch.Tensor | None:
        """
        Merges two matmul layers using the Log-Euclidean mean if they are both
        positive-definite. Otherwise, returns None to signal a fallback method should be used.

        Args:
            a: The first matmul matrix.
            b: The second matmul matrix.
            alpha: The interpolation factor (0 <= alpha <= 1).

        Returns:
            The merged matmul matrix if both inputs are positive-definite, otherwise None.
        """
        # Check if both matrices are positive-definite
        if MergeMethods.is_positive_definite(a) and MergeMethods.is_positive_definite(b):
            # Compute the Log-Euclidean mean
            log_a = torch.linalg.logm(a)
            log_b = torch.linalg.logm(b)
            merged_log = (1 - alpha) * log_a + alpha * log_b
            merged_matrix = torch.linalg.expm(merged_log)

            return merged_matrix
        else:
            return None  # Signal to use a fallback method

    def is_positive_definite(A: torch.Tensor) -> bool:
        """
        Checks if a matrix is positive-definite.

        Args:
            A: The input matrix.

        Returns:
            True if the matrix is positive-definite, False otherwise.
        """
        try:
            torch.linalg.cholesky(A)  # Cholesky decomposition exists only for PD matrices
            return True
        except RuntimeError:
            return False

    def polar_decomposition(a: torch.Tensor, b: torch.Tensor, alpha: float,
                            regularization_eps: float = 1e-6) -> torch.Tensor:
        """
        Merges two matmul layers using an adaptive polar decomposition approach,
        handling matrices with potentially very different shapes.

        Args:
            a: The first matmul matrix (can be non-square).
            b: The second matmul matrix (can be non-square).
            alpha: The interpolation factor (0 <= alpha <= 1).
            regularization_eps: Small constant added to the singular values for numerical stability.

        Returns:
            The merged matmul matrix.
        """
        device = a.device

        # Perform SVD for matrices 'a' and 'b'
        u_a, s_a, vt_a = torch.linalg.svd(a, full_matrices=False)
        u_b, s_b, vt_b = torch.linalg.svd(b, full_matrices=False)

        # Regularize singular values
        s_a = s_a + regularization_eps
        s_b = s_b + regularization_eps

        # Determine the target shape (use the shape of 'a' as the target)
        target_shape = a.shape

        # Function to resize a matrix to the target shape
        def resize_to_target(matrix, target_shape):
            if matrix.shape != target_shape:
                matrix = F.interpolate(matrix.unsqueeze(0).unsqueeze(0), size=target_shape, mode='bilinear',
                                       align_corners=False)
                matrix = matrix.squeeze(0).squeeze(0)
            return matrix

        # Resize components to match the target shape
        u_a = resize_to_target(u_a, target_shape)
        vt_a = resize_to_target(vt_a, (target_shape[1], target_shape[1]))
        u_b = resize_to_target(u_b, target_shape)
        vt_b = resize_to_target(vt_b, (target_shape[1], target_shape[1]))

        # Recompute polar components
        u_a_polar = u_a @ vt_a
        p_a = vt_a.t() @ torch.diag(
            F.interpolate(s_a.unsqueeze(0).unsqueeze(0), size=target_shape[1], mode='linear').squeeze()) @ vt_a
        u_b_polar = u_b @ vt_b
        p_b = vt_b.t() @ torch.diag(
            F.interpolate(s_b.unsqueeze(0).unsqueeze(0), size=target_shape[1], mode='linear').squeeze()) @ vt_b

        # Interpolate the components
        merged_u = torch.lerp(u_a_polar, u_b_polar, alpha)
        merged_p = torch.lerp(p_a, p_b, alpha)

        # Reconstruct the merged matrix
        merged_matrix = merged_u @ merged_p

        return merged_matrix.to(device)

    def advanced_embedding_merge(a: torch.Tensor, b: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
        """
        Merges two embedding layers using a combination of alignment and interpolation techniques.

        Args:
        - a, b: Input embedding tensors (vocab_size x embedding_dim)
        - alpha: Blending factor (0 to 1)

        Returns:
        - Merged embedding tensor
        """
        # Step 1: Normalize embeddings
        a_norm = F.normalize(a, p=2, dim=1)
        b_norm = F.normalize(b, p=2, dim=1)

        # Step 2: Compute principal components
        _, _, V_a = torch.svd(a_norm, compute_uv=True)
        _, _, V_b = torch.svd(b_norm, compute_uv=True)

        # Step 3: Align embedding spaces
        R = torch.mm(V_b[:, :min(V_a.shape[1], V_b.shape[1])],
                     V_a[:, :min(V_a.shape[1], V_b.shape[1])].t())
        b_aligned = torch.mm(b, R)

        # Step 4: Interpolate in the aligned space
        merged = (1 - alpha) * a + alpha * b_aligned

        # Step 5: Preserve norms
        a_norms = torch.norm(a, dim=1, keepdim=True)
        b_norms = torch.norm(b, dim=1, keepdim=True)
        merged_norms = (1 - alpha) * a_norms + alpha * b_norms
        merged_normalized = F.normalize(merged, p=2, dim=1)
        merged_rescaled = merged_normalized * merged_norms

        return merged_rescaled

    def svd_interpolation(a: torch.Tensor, b: torch.Tensor, alpha: float,
                               regularization_eps: float = 1e-6) -> torch.Tensor:
        """
        Merges two matmul layers using SVD-based matrix logarithm interpolation, handling dimension mismatches.

        Args:
            a: The first matmul matrix (can be non-square).
            b: The second matmul matrix (can be non-square).
            alpha: The interpolation factor (0 <= alpha <= 1).
            regularization_eps: Small constant added to the singular values for numerical stability.

        Returns:
            The merged matmul matrix.
        """
        # Ensure the matrices are on the same device and have the same dtype
        device = a.device
        dtype = a.dtype

        # Get the shapes of a and b
        a_shape = a.shape
        b_shape = b.shape

        # Compute SVD of both matrices
        u_a, s_a, v_a = torch.linalg.svd(a, full_matrices=False)
        u_b, s_b, v_b = torch.linalg.svd(b, full_matrices=False)

        # Regularize singular values to avoid instability
        s_a = s_a + regularization_eps
        s_b = s_b + regularization_eps

        # Compute the logarithm of the singular values
        log_s_a = torch.log(s_a)
        log_s_b = torch.log(s_b)

        # Interpolate the singular values in log-space
        merged_log_s = (1 - alpha) * log_s_a + alpha * log_s_b

        # Exponentiate to return to the original space
        merged_s = torch.exp(merged_log_s)

        # Reconstruct the interpolated matrix based on shape compatibility
        u_a, v_a = u_a[:, :merged_s.shape[-1]], v_a[:, :merged_s.shape[-1]]
        u_b, v_b = u_b[:, :merged_s.shape[-1]], v_b[:, :merged_s.shape[-1]]

        # Handle different input/output dimensions
        min_dim = min(u_a.shape[-1], v_a.shape[-1], merged_s.shape[-1])

        u_a, u_b = u_a[:, :min_dim], u_b[:, :min_dim]
        v_a, v_b = v_a[:, :min_dim], v_b[:, :min_dim]

        # Interpolate the orthogonal components
        merged_u = (1 - alpha) * u_a + alpha * u_b
        merged_v = (1 - alpha) * v_a + alpha * v_b

        # Reconstruct the matrix using interpolated singular values and orthogonal components
        merged_matrix = merged_u @ torch.diag(merged_s) @ merged_v.T

        return merged_matrix.to(device=device, dtype=dtype)

    def polar_decomposition2(a: torch.Tensor, b: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        Merges two matmul layers using a simpler polar decomposition without matrix log/exp.

        Args:
            a: The first matmul matrix.
            b: The second matmul matrix.
            alpha: The interpolation factor (0 <= alpha <= 1).

        Returns:
            The merged matmul matrix.
        """
        device = a.device  # Get the device of the input tensors

        # Perform SVD for matrix 'a'
        u_a, s_a, v_a = torch.linalg.svd(a, full_matrices=False)
        u_a_polar = u_a @ v_a.T  # Unit matrix from 'a'
        p_a = v_a.T @ torch.diag(s_a) @ v_a  # Positive semi-definite matrix from 'a'

        # Perform SVD for matrix 'b'
        u_b, s_b, v_b = torch.linalg.svd(b, full_matrices=False)
        u_b_polar = u_b @ v_b.T  # Unit matrix from 'b'
        p_b = v_b.T @ torch.diag(s_b) @ v_b  # Positive semi-definite matrix from 'b'

        # Direct interpolation between the components without matrix_log/matrix_exp
        merged_u = (1 - alpha) * u_a_polar + alpha * u_b_polar
        merged_p = (1 - alpha) * p_a + alpha * p_b

        # Reconstruct the merged matrix
        merged_matrix = merged_u @ merged_p

        return merged_matrix.to(device)

    def matrix_is_large(A: torch.Tensor, threshold: int = 512) -> bool:
        """
        Determines if a matrix is considered "large" based on its dimensions.

        Args:
            A: The input matrix.
            threshold: The threshold for the minimum dimension size to be considered "large."

        Returns:
            True if the matrix is considered large, False otherwise.
        """
        m, n = A.shape  # Get the matrix dimensions
        return m >= threshold or n >= threshold  # Check if either dimension exceeds the threshold

    def dominant_rotation(A: torch.Tensor, threshold: float = 0.8) -> bool:
        """
        Estimates if a matrix primarily represents a rotation based on its singular values.

        Args:
            A: The input matrix.
            threshold: The threshold for the ratio of the largest singular value to the smallest
                       singular value to be considered "dominant rotation."

        Returns:
            True if the matrix is estimated to have a dominant rotation, False otherwise.
        """
        _, S, _ = torch.linalg.svd(A)  # Compute the singular values of the matrix
        largest_singular_value = S[0]
        smallest_singular_value = S[-1]
        return largest_singular_value / smallest_singular_value >= threshold

    def matrix_is_ill_conditioned(A: torch.Tensor, threshold: float = 100) -> bool:
        """
        Determines if a matrix is ill-conditioned based on its condition number.

        Args:
            A: The input matrix.
            threshold: The threshold for the condition number to be considered ill-conditioned.

        Returns:
            True if the matrix is ill-conditioned, False otherwise.
        """
        condition_number = torch.linalg.cond(A)  # Compute the condition number
        return condition_number >= threshold