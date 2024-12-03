import re
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

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from scipy.linalg import sqrtm
from scipy.stats import binom, rankdata
from sd_mecha.merge_methods.svd import torch_complex_dtype_map
from torch import Tensor, polar
from typing import Optional, Callable, Dict, Tuple, TypeVar, Generic, get_type_hints, get_origin, Union, get_args, List, Set, Iterable
from pytorch_wavelets import DWTForward, DWTInverse
from sd_mecha import Hyper, MergeSpace
from sd_mecha.merge_methods import SameMergeSpace
from sd_mecha.extensions.merge_method import LiftFlag, convert_to_recipe
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

    ### CUSTOM METHODS ###

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
            alpha: Hyper = 0.5,
            beta: Hyper = 0.5,
            kernel_size: int = 3,
            centroid_margin_factor: float = 0.08,
            frequency_weight: float = 0.4,
            use_cross_attention: bool = True,
            **kwargs,
    ) -> Tensor | SameMergeSpace:
        try:
            with torch.no_grad():  # Prevent gradient graph building for intermediate calculations
                if not (0 <= alpha <= 1 and 0 <= beta <= 1):
                    raise ValueError("Alpha and beta must be between 0 and 1")

                key = kwargs.get("key", "")
                if key.endswith(("in_proj_weight", "in_proj_bias")):
                    return MergeMethods.handle_attention_projection(a, b, c, alpha, beta, kwargs)

                original_shape = a.shape

                # Step 1: Frequency domain alignment in isolated context
                freq_aligned_b = torch.utils.checkpoint.checkpoint(
                    MergeMethods.frequency_selective_alignment,
                    a, b, c,
                    centroid_margin_factor
                )

                # Step 2: Spatial domain processing
                shape_2d = MergeMethods.determine_reshape_dimensions(a)
                a_2d = a.reshape(*shape_2d)
                b_2d = b.reshape(*shape_2d)
                c_2d = c.reshape(*shape_2d)
                freq_aligned_b_2d = freq_aligned_b.reshape(*shape_2d)

                # Calculate importance weights using cross-attention if enabled
                if use_cross_attention and min(shape_2d) > 1:
                    importance_weights = torch.utils.checkpoint.checkpoint(
                        MergeMethods.calculate_cross_attention,
                        a_2d.detach(), b_2d.detach(), c_2d.detach()
                    )
                else:
                    importance_weights = torch.ones_like(a_2d)

                # Calculate dissimilarity with anchor using checkpointing
                dissimilarity = torch.utils.checkpoint.checkpoint(
                    MergeMethods.calculate_dissimilarity,
                    a_2d.detach(), b_2d.detach(), c_2d.detach()
                )

                dissimilarity = MergeMethods.gaussian_blur(dissimilarity, kernel_size)

                # Combine frequency and spatial information
                b_combined = (
                        freq_aligned_b_2d * frequency_weight +
                        b_2d * (1 - frequency_weight)
                )

                # Vectorized SLERP implementation
                effective_alpha = alpha * importance_weights

                # Normalize vectors
                a_norm = F.normalize(a_2d, p=2, dim=-1)
                b_norm = F.normalize(b_combined, p=2, dim=-1)

                # Compute dot product
                dot_product = (a_norm * b_norm).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
                omega = torch.acos(dot_product)

                # Handle small angles to prevent numerical instability
                small_angle_mask = omega < 1e-4
                sin_omega = torch.sin(omega).clamp_min(1e-6)

                # Compute SLERP coefficients
                slerp_a = torch.where(small_angle_mask,
                                      1.0 - effective_alpha,
                                      torch.sin((1.0 - effective_alpha) * omega) / sin_omega)
                slerp_b = torch.where(small_angle_mask,
                                      effective_alpha,
                                      torch.sin(effective_alpha * omega) / sin_omega)

                # Compute merged result
                merged = slerp_a * a_2d + slerp_b * b_combined

                # Apply anchor-based adjustment
                anchor_adjustment = (b_combined - c_2d) * beta * dissimilarity
                merged = merged + anchor_adjustment * importance_weights

                result = merged.reshape(original_shape)

                # Ensure all intermediate tensors are cleared
                del (freq_aligned_b, a_2d, b_2d, c_2d, freq_aligned_b_2d, importance_weights,
                     dissimilarity, b_combined, a_norm, b_norm, dot_product, omega,
                     small_angle_mask, sin_omega, slerp_a, slerp_b, merged, anchor_adjustment)

                return result

        finally:
            # Clear any CUDA cache if using GPU
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

    @staticmethod
    def frequency_selective_alignment(
            a: Tensor,
            b: Tensor,
            c: Tensor,
            centroid_margin_factor: float = 0.1
    ) -> Tensor:
        """Frequency selective alignment with improved memory management."""
        with torch.no_grad():
            # Reshape tensors
            a_flat = a.reshape(-1).float()
            b_flat = b.reshape(-1).float()
            c_flat = c.reshape(-1).float()

            # Compute FFTs one at a time to reduce peak memory usage
            a_dft = torch.fft.rfft(a_flat)
            b_dft = torch.fft.rfft(b_flat)
            c_dft = torch.fft.rfft(c_flat)

            fft_size = a_dft.shape[0]

            # Calculate centroids sequentially
            centroids = {
                'a': MergeMethods.calculate_spectral_centroid(a_dft),
                'b': MergeMethods.calculate_spectral_centroid(b_dft),
                'c': MergeMethods.calculate_spectral_centroid(c_dft)
            }

            # Calculate phase coherence
            phase_coherence = MergeMethods.calculate_phase_coherence(a_dft, b_dft, c_dft)

            # Dynamic beta calculation
            freq_dissimilarity = abs(centroids['a'] - centroids['b'])
            dynamic_beta = torch.cos(torch.tensor(math.pi / 2) * freq_dissimilarity).item()
            dynamic_beta = dynamic_beta * phase_coherence

            # Define frequency bands
            margin = int(centroid_margin_factor * fft_size)
            passband_end = int(min(centroids['a'], centroids['c']) * fft_size - margin)
            stopband_start = int(max(centroids['a'], centroids['c']) * fft_size + margin)

            passband_end = max(0, min(passband_end, fft_size - margin))
            stopband_start = min(fft_size, max(stopband_start, margin))

            # Adjust frequency components
            result = MergeMethods.adjust_frequency_components(
                a_dft, b_dft, c_dft,
                passband_end, stopband_start,
                dynamic_beta
            )

            # Clean up FFT tensors explicitly
            del a_dft, b_dft, c_dft

            return torch.fft.irfft(result, a_flat.shape[0]).reshape(a.shape)

    def calculate_phase_coherence(a_dft: Tensor, b_dft: Tensor, c_dft: Tensor) -> float:
        """Calculate phase coherence between three signals."""
        phase_a = torch.angle(a_dft)
        phase_b = torch.angle(b_dft)
        phase_c = torch.angle(c_dft)

        # Calculate phase differences
        diff_ab = torch.abs(torch.angle(torch.exp(1j * (phase_a - phase_b))))
        diff_ac = torch.abs(torch.angle(torch.exp(1j * (phase_a - phase_c))))
        diff_bc = torch.abs(torch.angle(torch.exp(1j * (phase_b - phase_c))))

        # Average phase coherence
        coherence = torch.mean(torch.cos(diff_ab) + torch.cos(diff_ac) + torch.cos(diff_bc)) / 3
        return coherence.item()

    def adjust_frequency_components(
            a_dft: Tensor,
            b_dft: Tensor,
            c_dft: Tensor,  # Keep parameter for API consistency, but use minimally
            passband_end: int,
            stopband_start: int,
            dynamic_beta: float
    ) -> Tensor:
        """
        Adjust magnitude and phase of frequency components.
        The anchor tensor (c_dft) is used only for band definition in the caller,
        not for direct magnitude/phase adjustment.
        """
        # Separate magnitude and phase
        mag_b = torch.abs(b_dft)
        phase_b = torch.angle(b_dft)

        # Get reference magnitudes
        mag_a = torch.abs(a_dft)

        # Calculate weighted magnitude
        weighted_mag = torch.where(
            torch.arange(mag_b.shape[0], device=mag_b.device) < passband_end,
            (1 - dynamic_beta) * mag_a + dynamic_beta * mag_b,
            mag_b
        )

        # Apply smooth transition only if there's a valid transition range
        transition_range = stopband_start - passband_end
        if transition_range > 0:
            transition = torch.cos(
                torch.linspace(0, math.pi / 2, transition_range, device=mag_b.device)
            )
            weighted_mag[passband_end:stopband_start] *= transition

        return torch.polar(weighted_mag, phase_b)

    def calculate_spectral_centroid(dft: Tensor) -> float:
        """
        Calculates the spectral centroid of a tensor in the frequency domain.
        Returns a normalized centroid value between 0 and 1.
        """
        fft_size = dft.shape[0]
        frequencies = torch.arange(fft_size, device=dft.device) / fft_size  # Normalize frequencies to [0, 1]
        magnitudes = torch.abs(dft)
        centroid = (frequencies * magnitudes).sum() / (magnitudes.sum() + EPSILON)
        return centroid.item()

    def gaussian_blur(a: Tensor, kernel_size: int) -> Tensor:
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
    @convert_to_recipe(volatile_hypers=["cache"])
    def merge_layers(
            a: Tensor | SameMergeSpace,
            b: Tensor | SameMergeSpace,
            *,
            alpha: Hyper = 0.5,
            corr_threshold: Hyper = 0.5,
            cache: Optional[Dict[str, Dict[str, Tensor]]] = None,
            **kwargs,
    ) -> Tensor | SameMergeSpace:
        key = kwargs.get("key", "")

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
            return sd_mecha.weighted_sum.__wrapped__(a, b, alpha=alpha)
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
            return sd_mecha.weighted_sum.__wrapped__(a, b, alpha=alpha)

    def polar_decomposition(a: Tensor, b: Tensor, alpha: float,
                            regularization_eps: float = 1e-6,
                            cache: Optional[Dict] = None) -> Tensor:
        device = a.device
        dtype = a.dtype
        original_shape = a.shape

        if not original_shape:
            shape_2d = (1, 1)
        elif len(a.shape) == 4:
            shape_2d = (-1, functools.reduce(operator.mul, a.shape[1:]))
        else:
            shape_2d = (-1, a.shape[-1])
        a = a.reshape(*shape_2d)
        b = b.reshape(*shape_2d)

        def get_cached_svd(matrix: Tensor, prefix: str) -> Tuple[Tensor, Tensor, Tensor]:
            """Helper to handle SVD caching for either matrix."""
            if cache is not None and f"{prefix}_polar" in cache:
                # Cached polar decomposition available
                u_polar = cache[f"{prefix}_polar"].to(device, dtype)
                s = cache[f"{prefix}_s"].to(device, dtype)
                vt = cache[f"{prefix}_vt"].to(device, dtype)
            else:
                # Calculate and cache SVD components
                u, s, vt = torch.linalg.svd(matrix, full_matrices=False)
                u_polar = u @ vt  # Pre-compute polar component

                if cache is not None:
                    cache[f"{prefix}_polar"] = u_polar.to("cpu")
                    cache[f"{prefix}_s"] = s.to("cpu")
                    cache[f"{prefix}_vt"] = vt.to("cpu")

            return u_polar, s, vt

        # Get decompositions (from cache or compute)
        u_a_polar, s_a, vt_a = get_cached_svd(a, "a")
        u_b_polar, s_b, vt_b = get_cached_svd(b, "b")

        # Get or compute alignment transform
        if cache is not None and "transform" in cache:
            transform = cache["transform"].to(device, dtype)
        else:
            transform = MergeMethods.orthogonal_procrustes_ml(u_a_polar, u_b_polar)
            if cache is not None:
                cache["transform"] = transform.to("cpu")

        # Align polar decompositions
        u_b_polar_aligned = u_b_polar @ transform

        # Compute positive semidefinite parts
        p_a = vt_a.t() @ torch.diag(s_a + regularization_eps) @ vt_a
        p_b = vt_b.t() @ torch.diag(s_b + regularization_eps) @ vt_b

        # Merge components
        merged_u = MergeMethods.slerp_unitary_taylor(u_a_polar, u_b_polar_aligned, alpha)
        merged_p = torch.lerp(p_a, p_b, alpha)

        return (merged_u @ merged_p).reshape(original_shape)

    def slerp_unitary_taylor(A: Tensor, B: Tensor, alpha: float, num_terms: int = 5) -> Tensor:
        """
        Performs slerp between two unitary matrices using a Taylor series approximation
        of the matrix logarithm.

        Args:
            A: The first unitary matrix.
            B: The second unitary matrix.
            alpha: The interpolation factor (0 <= alpha <= 1).
            num_terms: The number of terms to include in the Taylor series approximation.

        Returns:
            The interpolated unitary matrix.
        """
        if torch.allclose(A, B, atol=1e-6):
            return A
        else:
            # Compute the relative rotation
            relative_rotation = B @ A.t()

            # Compute X for the Taylor series: X = relative_rotation - I
            X = relative_rotation - torch.eye(relative_rotation.size(-1), device=A.device)

            # Approximate the logarithm using the Taylor series
            log_rotation = torch.zeros_like(X)
            for i in range(1, num_terms + 1):
                log_rotation += ((-1) ** (i + 1) / i) * torch.linalg.matrix_power(X, i)

            # Interpolate in the tangent space
            interpolated_log = alpha * log_rotation

            # Map back to the space of unitary matrices
            interpolated_unitary = torch.linalg.matrix_exp(interpolated_log) @ A

            return interpolated_unitary

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
                return MergeMethods.polar_decomposition(a, b, adjusted_alpha, cache=cache)

            # For key/value projections (different dimensions), focus caching on SVD and transform
            def get_cached_svd(matrix: Tensor, prefix: str) -> Tuple[Tensor, Tensor, Tensor]:
                """Helper to handle SVD caching."""
                cache_key = f"{key}_{prefix}"
                if cache is not None and f"{cache_key}_u" in cache:
                    u = cache[f"{cache_key}_u"].to(device, dtype)
                    s = cache[f"{cache_key}_s"].to(device, dtype)
                    vh = cache[f"{cache_key}_vh"].to(device, dtype)
                else:
                    svd_driver = "gesvdj" if matrix.is_cuda else "gesvd"
                    u, s, vh = torch.linalg.svd(matrix, full_matrices=False, driver=svd_driver)

                    if cache is not None:
                        cache[f"{cache_key}_u"] = u.to('cpu')
                        cache[f"{cache_key}_s"] = s.to('cpu')
                        cache[f"{cache_key}_vh"] = vh.to('cpu')

                return u, s, vh

            # Get cached SVD components for matrices `a` and `b`
            u_a, s_a, vh_a = get_cached_svd(a, "a")
            u_b, s_b, vh_b = get_cached_svd(b, "b")

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
                attn_a = torch.softmax(x @ a.T / math.sqrt(a.shape[-1]), dim=-1)
                attn_b = torch.softmax(x @ b.T / math.sqrt(b.shape[-1]), dim=-1)

                kl_div = F.kl_div(attn_a.log(), attn_b, reduction='batchmean')
                adjusted_alpha = alpha * torch.sigmoid(1.0 - kl_div)

            # Call polar_decomposition without caching, due to dynamic adjusted_alpha
            return MergeMethods.polar_decomposition(a, b, adjusted_alpha)

    def merge_attention_output(a: Tensor, b: Tensor, alpha: float, key: str,
                               cache: Optional[Dict] = None) -> Tensor:
        """
        Merge attention output projections while preserving output distribution,
        without caching for dynamically adjusted alpha values.
        """
        with torch.no_grad():
            # Generate sample inputs
            x = torch.randn(min(512, a.shape[-1]), a.shape[-1], device=a.device)

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
        merged = MergeMethods.polar_decomposition(a, b, adjusted_alpha)

        # Scale to preserve activation magnitude
        scale_a = torch.norm(out_a) / torch.norm(x)
        scale_b = torch.norm(out_b) / torch.norm(x)
        target_scale = (1 - alpha) * scale_a + alpha * scale_b

        with torch.no_grad():
            current_scale = torch.norm(x @ merged.T) / torch.norm(x)

        return merged * (target_scale / (current_scale + 1e-6))

    def merge_ffn_proj(a: torch.Tensor, b: torch.Tensor, alpha: float, key: str) -> torch.Tensor:
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

    def merge_ffn_proj_standard(a: torch.Tensor, b: torch.Tensor, alpha: float,
                                expansion_factor: float) -> torch.Tensor:
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
        merged = torch.lerp(a, b, merge_weight)

        # Rescale to preserve activation magnitude
        scale_a = torch.norm(a_act) / torch.norm(test_input)
        scale_b = torch.norm(b_act) / torch.norm(test_input)
        target_scale = (1 - alpha) * scale_a + alpha * scale_b
        current_scale = torch.norm(torch.relu(test_input @ merged.T)) / torch.norm(test_input)

        return merged * (target_scale / (current_scale + 1e-6))

    def merge_ffn_out(a: torch.Tensor, b: torch.Tensor, alpha: float, corr_threshold: float,
                      cache: Optional[Dict[str, Dict[str, torch.Tensor]]] = None) -> torch.Tensor:
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
                groups_a.append(group_a[:actual_size])  # Only take the matching number of indices
                groups_b.append(group_b[:actual_size])
                used_indices.update(group_a[:actual_size].tolist())

        # Initialize merged tensor
        merged = torch.zeros_like(a)

        # Helper function for caching SVD components
        def get_cached_svd(matrix: torch.Tensor, prefix: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            cache_key = f"{prefix}_svd"
            if cache is not None and f"{cache_key}_u" in cache:
                u = cache[f"{cache_key}_u"].to(device, dtype)
                s = cache[f"{cache_key}_s"].to(device, dtype)
                vh = cache[f"{cache_key}_vh"].to(device, dtype)
            else:
                u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
                if cache is not None:
                    cache[f"{cache_key}_u"] = u.cpu()
                    cache[f"{cache_key}_s"] = s.cpu()
                    cache[f"{cache_key}_vh"] = vh.cpu()
            return u, s, vh

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

            # Get SVD components with caching
            u_a, s_a, v_a = get_cached_svd(slice_a_norm, f"{group_a}_a")
            u_b, s_b, v_b = get_cached_svd(slice_b_norm, f"{group_b}_b")

            # Use minimum number of components for alignment
            k = min(v_a.shape[1], v_b.shape[1])

            # Use orthogonal Procrustes for alignment with caching
            if k > 0:
                procrustes_key = f"procrustes_{len(group_a)}_{len(group_b)}"
                if cache is not None and procrustes_key in cache:
                    R = cache[procrustes_key].to(device, dtype)
                else:
                    R = MergeMethods.orthogonal_procrustes_ml(v_a[:, :k], v_b[:, :k])
                    if cache is not None:
                        cache[procrustes_key] = R.cpu()

                v_b_aligned = v_b[:, :k] @ R.T
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

    def slerp_interp(a: Tensor, b: Tensor, alpha: float) -> Tensor:
        """
        Spherical linear interpolation (slerp) between two tensors `a` and `b`.
        Args:
            a: The first tensor, normalized along the appropriate dimension.
            b: The second tensor, same shape as `a`.
            alpha: The interpolation factor (0 <= alpha <= 1).
        Returns:
            Interpolated tensor in the same shape as `a` and `b`.
        """
        # Normalize input tensors along the feature dimension
        a_norm = a / a.norm(dim=-1, keepdim=True)
        b_norm = b / b.norm(dim=-1, keepdim=True)

        # Dot product between the normalized tensors to calculate the angle
        dot_product = torch.clamp((a_norm * b_norm).sum(dim=-1, keepdim=True), -1.0, 1.0)
        theta = torch.acos(dot_product)

        # Spherical interpolation formula
        sin_theta = torch.sin(theta)
        slerp_factor_a = torch.sin((1 - alpha) * theta) / sin_theta
        slerp_factor_b = torch.sin(alpha * theta) / sin_theta

        # Calculate and return the interpolated tensor
        return slerp_factor_a * a + slerp_factor_b * b

    def get_layer_type(shape, kwargs):
        key = kwargs["key"]

        # Check for attention layers first
        if any(x in key for x in
               [".to_q.", ".to_k.", ".to_v.", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]):
            # Add cross-attention check
            if ".attn2." in key:
                return MergeMethods.LayerType.CROSS_ATTENTION_QKV
            return MergeMethods.LayerType.ATTENTION_QKV

        elif any(x in key for x in [".to_out.", "self_attn.out_proj"]) and ".bias" in key:
            return MergeMethods.LayerType.OFFSET

        # Feed Forward Network (FFN)
        elif ".ff.net." in key and ".proj." in key:
            return MergeMethods.LayerType.FFN_PROJ
        elif ".ff.net." in key and ".weight" in key:
            return MergeMethods.LayerType.FFN_OUT
        elif ".ff.net." in key and ".bias" in key:
            return MergeMethods.LayerType.OFFSET

        # Layer Norms
        elif ".norm" in key:
            return MergeMethods.LayerType.SCALAR

        # True embeddings (vocabulary mappings)
        elif "token_embedding" in key or "shared.weight" in key:
            return MergeMethods.LayerType.EMBEDD

        # Treat other embedding-like layers as matrix transformations
        elif any(x in key for x in ["positional_embedding", "text_projection", "label_emb"]):
            return MergeMethods.LayerType.MATMUL

        # Your existing fallbacks
        elif len(shape) == 4:
            return MergeMethods.LayerType.CONV2D

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

    def matrix_is_large(A: Tensor, threshold: int = 1280) -> bool:
        """
        Determines if a matrix is considered "large" based on its dimensions.

        Args:
            A: The input matrix.
            threshold: The threshold for the minimum dimension size to be considered "large."

        Returns:
            True if the matrix is considered large, False otherwise.
        """
        if A.ndim < 2:  # Check if tensor has fewer than 2 dimensions
            return False  # Treat non-2D tensors as "not large"
        m, n = A.shape  # Get the matrix dimensions
        return m >= threshold or n >= threshold  # Check if either dimension exceeds the threshold

    def dominant_rotation(A: Tensor, threshold: float = 0.8) -> bool:
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

    def matrix_is_ill_conditioned(A: Tensor, threshold: float = 100) -> bool:
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

    def orthogonal_procrustes_ml(a, b, cancel_reflection: bool = False):
        # Compute A^T @ B once since it's used in both branches
        atb = a.T @ b

        use_lowrank = not cancel_reflection and a.shape[0] + 10 < a.shape[1]
        if use_lowrank:
            svd_driver = "gesvdj" if a.is_cuda else None
            u, _, v = sd_mecha.merge_methods.svd.torch_svd_lowrank(atb, driver=svd_driver, q=a.shape[0] + 10)
            vt = v.T
            del v
        else:
            svd_driver = "gesvd" if a.is_cuda else None
            u, _, vt = torch.linalg.svd(atb, driver=svd_driver)
            if cancel_reflection:
                u[:, -1] *= torch.sign(torch.det(u) * torch.det(vt))  # More numerically stable

        transform = u @ vt

        if not torch.isfinite(transform).all():  # Check the transform instead of just u
            raise ValueError(
                f"determinant error: {torch.det(transform)}. "
                'This can happen when merging on the CPU with the "rotate" method. '
                "Consider merging on a cuda device, "
                "or try setting `alignment` to 1 for the problematic blocks. "
                "See this related discussion for more info: "
                "https://github.com/s1dlx/meh/pull/50#discussion_r1429469484"
            )

        return transform

    @staticmethod
    @convert_to_recipe
    def streaming_ties_sum_extended(
            *models: Tensor | LiftFlag[MergeSpace.DELTA],
            k: Hyper = 0.218,
            vote_sgn: Hyper = 0.0,
            apply_stock: Hyper = 0.0,
            cos_eps: Hyper = 1e-6,
            apply_median: Hyper = 1.0,
            eps: Hyper = 1e-6,
            maxiter: Hyper = 150,
            ftol: Hyper = 1e-22,
            weight_decay: Hyper = 0.0218,
            min_agreement: Hyper = 0.3,
            chunk_size: int = 4,  # Will be adjusted based on available memory
            memory_safety_margin: float = 0.8,  # Fraction of available memory to use
            **kwargs,
    ) -> Tensor | LiftFlag[MergeSpace.DELTA]:
        """
        Memory-efficient TIES implementation with dynamic chunking based on available GPU memory.
        """
        if not models:
            raise ValueError("At least one model must be provided")

        device = models[0].device
        dtype = models[0].dtype
        total_models = len(models)

        # Calculate adaptive chunk size based on available memory if using CUDA
        def get_adaptive_chunk_size(sample_model, total_models):
            if device.type == 'cuda':
                # Get available memory
                available_memory = torch.cuda.get_device_properties(device).total_memory
                free_memory = torch.cuda.memory_allocated(device)
                usable_memory = (available_memory - free_memory) * memory_safety_margin

                # Estimate memory needed per model
                sample_size = sample_model.nelement() * sample_model.element_size()
                # Account for additional tensors created during processing
                estimated_overhead = sample_size * 3  # For filtered, signs, and temporary computations

                # Calculate maximum models that can fit in memory
                max_chunk_size = int(usable_memory / estimated_overhead)

                # Ensure chunk size is at least 1 and no more than total models
                return max(1, min(max_chunk_size, total_models))
            return chunk_size  # Return default chunk size for CPU

        # Get adaptive chunk size
        adaptive_chunk_size = get_adaptive_chunk_size(models[0], total_models)

        # Initialize accumulators
        accumulated_filtered = []
        accumulated_signs = []

        # Process models in chunks with adaptive size
        for chunk_start in range(0, total_models, adaptive_chunk_size):
            chunk_end = min(chunk_start + adaptive_chunk_size, total_models)
            chunk_models = models[chunk_start:chunk_end]

            # Monitor memory before processing chunk
            if device.type == 'cuda':
                current_memory = torch.cuda.memory_allocated(device)
                max_memory = torch.cuda.get_device_properties(device).total_memory

                # If memory usage is too high, reduce chunk size
                if current_memory > max_memory * 0.9:  # 90% memory threshold
                    adaptive_chunk_size = max(1, adaptive_chunk_size // 2)
                    print(f"Memory pressure detected. Reducing chunk size to {adaptive_chunk_size}")
                    torch.cuda.empty_cache()

            # Process current chunk
            chunk_filtered, chunk_signs = MergeMethods._process_model_chunk(
                chunk_models,
                k=k,
                device=device,
                dtype=dtype
            )

            accumulated_filtered.append(chunk_filtered)
            accumulated_signs.append(chunk_signs)

            # Clear cache if memory pressure is high
            if device.type == 'cuda' and torch.cuda.memory_allocated(device) > 0.8 * max_memory:
                torch.cuda.empty_cache()

        # Concatenate results
        filtered_delta = torch.cat(accumulated_filtered, dim=0)
        signs = torch.cat(accumulated_signs, dim=0)

        # Update chunk size for downstream operations based on current memory state
        if device.type == 'cuda':
            adaptive_chunk_size = get_adaptive_chunk_size(filtered_delta, total_models)

        # Compute final results with adaptive chunk size
        final_results = MergeMethods._compute_final_results(
            filtered_delta,
            signs,
            vote_sgn=vote_sgn,
            min_agreement=min_agreement,
            weight_decay=weight_decay
        )

        filtered_delta, param_counts = final_results

        if apply_median <= 0.0:
            # Model Stock pathway with adaptive chunking
            if apply_stock > 0.0:
                t = MergeMethods._compute_model_stock_chunked(
                    filtered_delta,
                    cos_eps=cos_eps,
                    chunk_size=adaptive_chunk_size
                )
            else:
                t = 1.0

            filtered_delta = filtered_delta.sum(dim=0)
            param_counts = torch.clamp(param_counts, min=eps)
            result = filtered_delta * t / param_counts
        else:
            # Geometric median computation with adaptive chunks
            result = MergeMethods._compute_geometric_median_chunked(
                filtered_delta,
                eps=eps,
                maxiter=maxiter,
                ftol=ftol,
                chunk_size=adaptive_chunk_size
            )

        return torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _process_model_chunk(chunk_models, k, device, dtype):
        """Process a chunk of models efficiently on GPU."""
        chunk_filtered = []
        chunk_signs = []

        for model in chunk_models:
            # Move single model to GPU, process, and free immediately
            model_gpu = model.to(device=device, dtype=dtype)
            filtered = MergeMethods.filter_top_k(model_gpu, k)
            signs = torch.sign(torch.where(
                torch.abs(filtered) > 1e-7,
                filtered,
                torch.zeros_like(filtered)
            ))

            chunk_filtered.append(filtered)
            chunk_signs.append(signs)

            # Clear GPU memory explicitly
            del model_gpu
            torch.cuda.empty_cache()

        return (
            torch.stack(chunk_filtered, dim=0),
            torch.stack(chunk_signs, dim=0)
        )

    @staticmethod
    def _compute_final_results(accumulated_filtered, accumulated_signs, vote_sgn, min_agreement, weight_decay):
        """Compute final results efficiently on CPU."""
        vote_tensor = accumulated_filtered if vote_sgn <= 0.0 else accumulated_signs
        sign_sum = torch.sum(vote_tensor, dim=0)
        agreement_ratio = torch.sum(accumulated_signs != 0, dim=0).float() / len(accumulated_signs)

        final_sign = torch.where(
            agreement_ratio >= min_agreement,
            torch.sign(sign_sum),
            torch.zeros_like(sign_sum)
        )

        delta_filters = (accumulated_signs == final_sign).float()
        param_counts = torch.sum(delta_filters, dim=0)

        if weight_decay > 0.0:
            accumulated_filtered = accumulated_filtered * (1.0 - weight_decay)

        filtered_delta = accumulated_filtered * delta_filters
        return filtered_delta, param_counts

    @staticmethod
    def _compute_model_stock_chunked(filtered_delta, cos_eps, chunk_size):
        """Compute model stock in memory-efficient chunks."""
        n_models = filtered_delta.shape[0]
        cos_sims = torch.zeros(n_models, n_models, device='cpu')

        for i in range(0, n_models, chunk_size):
            chunk_i = filtered_delta[i:i + chunk_size].flatten(1)
            chunk_i_norm = torch.norm(chunk_i, dim=1, keepdim=True)

            for j in range(0, n_models, chunk_size):
                chunk_j = filtered_delta[j:j + chunk_size].flatten(1)
                chunk_j_norm = torch.norm(chunk_j, dim=1, keepdim=True)

                # Compute cosine similarity for the chunk
                chunk_cos = torch.mm(chunk_i, chunk_j.t()) / (
                        torch.mm(chunk_i_norm, chunk_j_norm.t()) + cos_eps
                )

                cos_sims[i:i + chunk_size, j:j + chunk_size] = chunk_cos.cpu()

                del chunk_j, chunk_j_norm
                torch.cuda.empty_cache()

            del chunk_i, chunk_i_norm
            torch.cuda.empty_cache()

        # Compute final t score
        t = torch.mean(cos_sims > 0).item()
        return t

    @staticmethod
    def _compute_geometric_median_chunked(points, eps, maxiter, ftol, chunk_size):
        """
        Optimized memory-efficient geometric median computation for 3D tensors.
        points shape: [n_points, d1, d2] where d1, d2 are the dimensions of each point
        """
        n_points = points.shape[0]
        device = points.device
        points_shape = points.shape

        # Keep more data on GPU
        points_flat = points.reshape(n_points, -1)  # Keep on GPU initially
        weights = torch.ones(n_points, device=device)  # Keep weights on GPU

        # Initialize median on GPU
        median = torch.mean(points_flat, dim=0)
        best_objective = float('inf')
        best_median = median.clone()

        # Process larger chunks on GPU
        for iter_idx in range(maxiter):
            prev_objective = 0.0
            new_weights = torch.zeros_like(weights)

            # Process chunks directly on GPU
            for i in range(0, n_points, chunk_size):
                chunk = points_flat[i:i + chunk_size]  # Already on GPU
                chunk_weights = weights[i:i + chunk_size]

                # Compute distances efficiently on GPU
                diff = chunk - median.unsqueeze(0)
                # Use efficient GPU operations
                distances = torch.norm(diff, dim=1) + eps

                # Update objective and weights on GPU
                prev_objective += torch.sum(distances * chunk_weights)
                new_weights[i:i + chunk_size] = chunk_weights / distances

                # Optional: Only clear if memory pressure is high
                if torch.cuda.memory_allocated() > 0.9 * torch.cuda.max_memory_allocated():
                    del diff, distances
                    torch.cuda.empty_cache()

            # Update median efficiently on GPU
            weighted_sum = torch.zeros_like(median)
            weight_sum = new_weights.sum()

            # Compute new median in chunks but stay on GPU
            for i in range(0, n_points, chunk_size):
                chunk = points_flat[i:i + chunk_size]
                chunk_weights = new_weights[i:i + chunk_size]
                weighted_sum += torch.sum(chunk * chunk_weights.unsqueeze(1), dim=0)

            median = weighted_sum / (weight_sum + eps)

            # Check convergence
            if abs(prev_objective - best_objective) <= ftol * best_objective:
                break

            if prev_objective < best_objective:
                best_objective = prev_objective
                best_median = median.clone()

            weights = new_weights

            # Optional: Print progress every few iterations
            if iter_idx % 10 == 0:
                print(f"Iteration {iter_idx}, Objective: {prev_objective:.4e}")

        # Reshape final result
        final_median = best_median.reshape(points_shape[1:])
        return final_median

    @staticmethod
    def filter_top_k(a: Tensor, k: float) -> torch.Tensor:
        """Improved implementation using kthvalue with chunking."""
        total_params = torch.numel(a)
        k_params = max(int((1 - k) * total_params), 1)

        if k_params >= total_params:
            return torch.zeros_like(a)

        # Process in chunks for memory efficiency
        chunk_size = 1_000_000
        abs_values = []

        for i in range(0, total_params, chunk_size):
            chunk = a.flatten()[i:i + chunk_size]
            abs_values.append(torch.abs(chunk))

        # Concatenate chunks and find kth value
        abs_cat = torch.cat(abs_values)
        k_value = torch.kthvalue(abs_cat, k_params).values

        # Apply threshold with memory efficiency
        mask = torch.abs(a) >= k_value
        return a * mask.float()

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
        merged_delta = MergeMethods.streaming_ties_sum_extended.__wrapped__(
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
            # Calculate rank-based adjustments
            ranks = torch.argsort(torch.argsort(delta.abs().flatten())).reshape(delta.shape).float()
            delta_i = ((ranks / delta.numel()) - 0.5) * della_eps
            # Ensure probabilities are within the valid range
            probabilities = torch.clamp(p_min + delta_i, 0.0, 1.0)
        else:
            probabilities = p_min

        return torch.bernoulli(probabilities, generator=generator)

    @staticmethod
    @convert_to_recipe
    def frequency_merge(
            a: Tensor | SameMergeSpace,
            b: Tensor | SameMergeSpace,
            c: Tensor | SameMergeSpace,
            *,
            alpha: Hyper = 1.0,
            **kwargs,
    ) -> Tensor | SameMergeSpace:
        # Get differences from base
        delta_a = a - c
        delta_b = b - c

        # Could use FFT for frequency decomposition
        # Or use something like wavelets
        # But a simpler approach might be to look at local vs global changes:

        # Global changes (low frequency)
        global_a = torch.mean(delta_a, dim=-1, keepdim=True)
        global_b = torch.mean(delta_b, dim=-1, keepdim=True)

        # Local changes (high frequency)
        local_a = delta_a - global_a
        local_b = delta_b - global_b

        # Merge differently at each scale
        # Maybe trust global changes more (higher alpha)
        # And be more conservative with local changes
        merged = c + (global_a + alpha * (global_b - global_a)) + \
                 (local_a + (alpha * 0.5) * (local_b - local_a))

        return merged

    @staticmethod
    @convert_to_recipe
    def laplacian_difference1(
            a: Tensor | SameMergeSpace,
            b: Tensor | SameMergeSpace,
            c: Tensor | SameMergeSpace,
            *,
            alpha: Hyper = 1.0,
            n_levels: int = 4,
            **kwargs,
    ) -> Tensor | SameMergeSpace:
        def gaussian_downsample(x: Tensor) -> Tensor:
            # Smooth and downsample
            # Using average pooling as simple approximation of Gaussian
            return torch.nn.functional.avg_pool1d(x, kernel_size=2, stride=2)

        def gaussian_upsample(x: Tensor) -> Tensor:
            # Upsample and smooth
            return torch.nn.functional.interpolate(x, scale_factor=2, mode='linear')

        def build_pyramid(x: Tensor) -> List[Tensor]:
            gaussian = [x]
            laplacian = []

            # Build Gaussian pyramid
            for _ in range(n_levels - 1):
                gaussian.append(gaussian_downsample(gaussian[-1]))

            # Build Laplacian pyramid
            for i in range(n_levels - 1):
                upsampled = gaussian_upsample(gaussian[i + 1])
                # Pad/trim upsampled to match current level size
                if upsampled.shape != gaussian[i].shape:
                    diff = gaussian[i].shape[-1] - upsampled.shape[-1]
                    if diff > 0:
                        upsampled = torch.nn.functional.pad(upsampled, (0, diff))
                    else:
                        upsampled = upsampled[..., :gaussian[i].shape[-1]]
                laplacian.append(gaussian[i] - upsampled)

            # Add smallest Gaussian level as last Laplacian level
            laplacian.append(gaussian[-1])

            return laplacian

        # Build pyramids for all three models
        a_pyr = build_pyramid(a)
        b_pyr = build_pyramid(b)
        c_pyr = build_pyramid(c)

        merged_pyr = []
        for level in range(n_levels):
            # Adjust alpha based on level
            # More aggressive at lower frequencies (higher levels)
            level_alpha = alpha * (1.0 + level / (n_levels - 1))

            # Get changes from base for both models
            a_delta = a_pyr[level] - c_pyr[level]
            b_delta = b_pyr[level] - c_pyr[level]

            # Calculate agreement mask
            # Higher when changes are similar, lower when divergent
            agreement = torch.cosine_similarity(a_delta, b_delta, dim=-1, eps=1e-8)
            agreement = torch.clamp(agreement, 0, 1).unsqueeze(-1)

            # Calculate magnitude mask
            # Favor stronger changes but prevent extreme differences
            mag_a = torch.norm(a_delta, dim=-1, keepdim=True)
            mag_b = torch.norm(b_delta, dim=-1, keepdim=True)
            mag_ratio = torch.minimum(mag_a, mag_b) / torch.maximum(mag_a, mag_b).clamp(min=1e-8)

            # Combined mask considers both agreement and relative magnitudes
            mask = agreement * mag_ratio

            # For highest frequencies (level 0), be more conservative
            if level == 0:
                mask *= 0.5

            # Merge this level
            # Start with a's changes, blend in b's changes where mask is high
            merged_delta = a_delta * (1 - mask * level_alpha) + b_delta * (mask * level_alpha)
            merged_pyr.append(c_pyr[level] + merged_delta)

        # Reconstruct from pyramid
        result = merged_pyr[-1]
        for level in range(n_levels - 2, -1, -1):
            upsampled = gaussian_upsample(result)
            # Handle size mismatch
            if upsampled.shape != merged_pyr[level].shape:
                diff = merged_pyr[level].shape[-1] - upsampled.shape[-1]
                if diff > 0:
                    upsampled = torch.nn.functional.pad(upsampled, (0, diff))
                else:
                    upsampled = upsampled[..., :merged_pyr[level].shape[-1]]
            result = upsampled + merged_pyr[level]

        return result

    @staticmethod
    def model_aware_merge(
            a: Tensor | LiftFlag[MergeSpace.DELTA],
            b: Tensor | LiftFlag[MergeSpace.DELTA],
            *,
            block_attention: bool = True,
            alpha: float = 1.0,
            temperature: float = 1.0,
            window_size: int = 5,
            n_levels: int = 4,
            sigma: float = 1.0,
            agreement_weights: Optional[Dict[str, float]] = None,
            **kwargs,
    ) -> Tensor | LiftFlag[MergeSpace.DELTA]:
        """
        Model-aware merge using enhanced agreement metrics and attention with layer-specific handling.
        All merges use laplacian_difference with appropriate layer-specific agreement calculations.

        Args:
            a: First model's tensor or state dict
            b: Second model's tensor or state dict
            block_attention: Whether to process attention blocks together (True) or separately (False)
            alpha: Base interpolation strength (0-1). Higher values favor model B more
            temperature: Temperature for attention pattern softmax
            window_size: Window size for calculating local agreement metrics
            n_levels: Number of levels in the Laplacian pyramid
            sigma: Gaussian blur parameter for Laplacian pyramid
            agreement_weights: Optional custom weights for different agreement metrics
                             Default: {'cosine': 0.4, 'structural': 0.4, 'frequency': 0.2}
        """
        merged = {}

        # Set default agreement weights if not provided
        if agreement_weights is None:
            agreement_weights = {
                'cosine': 0.4,
                'structural': 0.4,
                'frequency': 0.2
            }

        # Group parameters by layer type and position
        layer_groups = {
            'attention': [],
            'feed_forward': [],
            'conv': [],
            'embedding': [],
            'norm': []
        }

        for key in a.keys():
            if 'attn' in key:
                layer_groups['attention'].append(key)
            elif 'mlp' in key or 'ff' in key:
                layer_groups['feed_forward'].append(key)
            elif 'conv' in key:
                layer_groups['conv'].append(key)
            elif 'embed' in key:
                layer_groups['embedding'].append(key)
            elif 'norm' in key:
                layer_groups['norm'].append(key)

        def process_attention_blocks(keys: list[str]) -> None:
            """Special handling for attention blocks to maintain relationships"""
            if not block_attention:
                # Process each attention key individually if blocking is disabled
                for key in keys:
                    merged[key] = process_layer(key, a[key], b[key])
                return

            # Group Q,K,V matrices together
            qkv_groups = defaultdict(list)
            for key in keys:
                match = re.match(r'.*block_(\d+).*(?:query|key|value)', key)
                if match:
                    block_num = match.group(1)
                    qkv_groups[block_num].append(key)

            for block_num, qkv_keys in qkv_groups.items():
                if len(qkv_keys) != 3:  # Skip incomplete QKV triads
                    continue

                # Identify layer type and get proper reshape functions
                layer_info = identify_layer(qkv_keys[0], a[qkv_keys[0]])

                # Process each QKV set while preserving attention structure
                tensors_a = [a[k] for k in qkv_keys]
                tensors_b = [b[k] for k in qkv_keys]

                # Reshape tensors properly
                reshaped_a, restore_fn = zip(*[reshape_for_processing(t, layer_info) for t in tensors_a])
                reshaped_b, _ = zip(*[reshape_for_processing(t, layer_info) for t in tensors_b])

                # Stack for processing
                stacked_a = torch.stack(reshaped_a)
                stacked_b = torch.stack(reshaped_b)

                # Calculate attention-specific agreement with temperature parameter
                agreement = calculate_attention_agreement_fixed(stacked_a, stacked_b, temperature=temperature)

                # Merge using laplacian_difference with custom parameters
                merged_qkv = laplacian_difference(
                    stacked_a,
                    stacked_b,
                    alpha=alpha,
                    n_levels=n_levels,
                    sigma=sigma
                )

                # Unstack and restore original shapes
                for idx, key in enumerate(qkv_keys):
                    merged[key] = restore_fn[idx](merged_qkv[idx])

        def process_layer(key: str, a: Tensor, b: Tensor) -> Tensor:
            """Process individual layer using appropriate agreement calculation"""
            layer_info = identify_layer(key, a)

            # Reshape tensors for processing
            a_reshaped, restore_fn = reshape_for_processing(a, layer_info)
            b_reshaped, _ = reshape_for_processing(b, layer_info)

            # Calculate layer-specific agreement with custom parameters
            if layer_info.type == LayerType.ATTENTION_QKV:
                agreement = calculate_attention_agreement_fixed(
                    a_reshaped,
                    b_reshaped,
                    temperature=temperature
                )
            elif layer_info.type in [LayerType.CONV_3X3, LayerType.CONV_1X1]:
                agreement = calculate_spatial_agreement_fixed(
                    a_reshaped,
                    b_reshaped,
                    kernel_size=window_size
                )
            else:
                agreement = calculate_advanced_agreement(
                    a_reshaped,
                    b_reshaped,
                    window_size=window_size,
                    weights=agreement_weights
                )

            # Merge using laplacian_difference with custom parameters
            merged_tensor = laplacian_difference(
                a_reshaped,
                b_reshaped,
                alpha=alpha,
                n_levels=n_levels,
                sigma=sigma
            )

            return restore_fn(merged_tensor)

        # Process each layer group
        for group_name, keys in layer_groups.items():
            if group_name == 'attention' and block_attention:
                process_attention_blocks(keys)
            else:
                for key in keys:
                    merged[key] = process_layer(key, a[key], b[key])

        return merged

    def calculate_advanced_agreement(a: Tensor, b: Tensor, window_size: int = 5) -> Tensor:
        """
        Calculate enhanced agreement metrics between two delta tensors with layer-aware processing
        """
        # Reshape tensors for agreement calculation while preserving structure
        orig_shape = a.shape

        if len(orig_shape) == 4:  # Convolution layers
            a_flat = a.view(a.size(0), -1)
            b_flat = b.view(b.size(0), -1)
        else:
            a_flat = a.view(-1, a.shape[-1]) if len(orig_shape) > 2 else a
            b_flat = b.view(-1, b.shape[-1]) if len(orig_shape) > 2 else b

        # Basic cosine similarity
        cosine_sim = torch.cosine_similarity(a_flat, b_flat, dim=-1, eps=1e-8)

        # Structural similarity with proper dimensionality handling
        def local_stats(x: Tensor, kernel_size: int):
            if x.dim() <= 2:
                padding = kernel_size // 2
                padded = torch.nn.functional.pad(x, (padding, padding))
                windows = padded.unfold(-1, kernel_size, 1)
                return windows.mean(dim=-1), windows.var(dim=-1)
            else:
                # For higher dimensions, use spatial averaging
                pool = torch.nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size // 2)
                mean = pool(x)
                return mean, pool(x.pow(2)) - mean.pow(2)

        # Compute local statistics
        a_mean, a_var = local_stats(a_flat, window_size)
        b_mean, b_var = local_stats(b_flat, window_size)

        # Calculate structural similarity
        C1 = (0.01 * torch.max(torch.abs(a_flat))) ** 2
        C2 = (0.03 * torch.max(torch.abs(a_flat))) ** 2

        numerator = (2 * a_mean * b_mean + C1) * (2 * torch.sqrt(torch.clamp(a_var * b_var, min=1e-8)) + C2)
        denominator = (a_mean ** 2 + b_mean ** 2 + C1) * (a_var + b_var + C2)
        structural_sim = numerator / denominator

        # Frequency domain agreement with proper reshaping
        if a_flat.dim() <= 2:
            fft_a = torch.fft.rfft(a_flat, dim=-1)
            fft_b = torch.fft.rfft(b_flat, dim=-1)
            freq_agreement = torch.cosine_similarity(
                torch.abs(fft_a),
                torch.abs(fft_b),
                dim=-1
            )
        else:
            # For higher dimensions, use 2D FFT
            fft_a = torch.fft.rfft2(a)
            fft_b = torch.fft.rfft2(b)
            freq_agreement = torch.cosine_similarity(
                torch.abs(fft_a).view(fft_a.size(0), -1),
                torch.abs(fft_b).view(fft_b.size(0), -1),
                dim=-1
            )

        # Combine metrics with learned weights
        agreement = (
                0.4 * cosine_sim +
                0.4 * structural_sim +
                0.2 * freq_agreement
        )

        return torch.clamp(agreement, 0, 1).unsqueeze(-1)

    def calculate_attention_agreement_fixed(
            a: Tensor,
            b: Tensor,
            temperature: float = 1.0
    ) -> Tensor:
        """
        Calculate agreement specifically for attention mechanisms using delta tensors.
        Takes into account attention pattern similarity and head relationships.

        Args:
            a: Delta from base model for first model (a - c)
            b: Delta from base model for second model (b - c)
            temperature: Softmax temperature for attention pattern comparison

        Returns:
            Tensor: Agreement scores
        """

        # Calculate attention patterns from the deltas
        def get_attention_pattern(x: Tensor) -> Tensor:
            if x.dim() == 3:  # (num_heads, seq_len, head_dim)
                q, k, v = x.chunk(3, dim=0)
            else:
                q = k = v = x

            # Calculate attention scores from the delta directly
            attn_pattern = torch.matmul(q, k.transpose(-2, -1)) / temperature
            return F.softmax(attn_pattern, dim=-1)

        # Get patterns for each delta
        pattern_a = get_attention_pattern(a)
        pattern_b = get_attention_pattern(b)

        # Calculate pattern similarity directly between deltas
        pattern_agreement = F.cosine_similarity(
            pattern_a.view(pattern_a.size(0), -1),
            pattern_b.view(pattern_b.size(0), -1),
            dim=-1
        )

        # Calculate value space agreement directly between deltas
        value_agreement = F.cosine_similarity(
            a.view(a.size(0), -1),
            b.view(b.size(0), -1),
            dim=-1
        )

        # Combine agreements with emphasis on pattern agreement
        combined_agreement = 0.7 * pattern_agreement + 0.3 * value_agreement
        return torch.clamp(combined_agreement, 0, 1).unsqueeze(-1)

    def calculate_spatial_agreement_fixed(
            a: Tensor,
            b: Tensor,
            kernel_size: int = 3
    ) -> Tensor:
        """
        Calculate agreement for convolutional layers considering spatial relationships.

        Args:
            a: Delta from base model for first model (a - c)
            b: Delta from base model for second model (b - c)
            kernel_size: Size of the local neighborhood to consider

        Returns:
            Tensor: Agreement scores
        """

        # Unfold for local neighborhood analysis
        def get_local_features(x: Tensor) -> Tensor:
            padding = kernel_size // 2
            x_padded = F.pad(x, (padding, padding, padding, padding))
            return F.unfold(x_padded, kernel_size)

        # Get local features directly from deltas
        local_a = get_local_features(a)
        local_b = get_local_features(b)

        # Calculate local structure agreement
        spatial_agreement = F.cosine_similarity(local_a, local_b, dim=1)

        # Calculate channel-wise agreement
        channel_agreement = F.cosine_similarity(
            a.view(a.size(0), -1),
            b.view(b.size(0), -1),
            dim=-1
        )

        # Combine agreements
        combined_agreement = 0.6 * spatial_agreement + 0.4 * channel_agreement
        return torch.clamp(combined_agreement, 0, 1).unsqueeze(-1)

    @convert_to_recipe
    def laplacian_difference(
            a: Tensor | SameMergeSpace,
            b: Tensor | SameMergeSpace,
            c: Tensor | SameMergeSpace,
            *,
            alpha: Hyper = 1.0,
            n_levels: int = 4,
            sigma: float = 1.0,  # Added Gaussian blur parameter
            **kwargs,
    ) -> Tensor | SameMergeSpace:
        def gaussian_downsample(x: Tensor) -> Tensor:
            # Use actual Gaussian blur instead of avg_pool approximation
            kernel_size = int(6 * sigma)
            if kernel_size % 2 == 0:
                kernel_size += 1
            gauss = torch.nn.Conv1d(
                x.shape[1], x.shape[1], kernel_size,
                padding=kernel_size // 2, groups=x.shape[1], bias=False
            )
            # Generate Gaussian kernel
            kernel = torch.exp(-torch.linspace(-3 * sigma, 3 * sigma, kernel_size) ** 2 / (2 * sigma ** 2))
            kernel = kernel / kernel.sum()
            gauss.weight.data = kernel.view(1, 1, -1).repeat(x.shape[1], 1, 1)
            gauss.weight.requires_grad = False

            # Apply Gaussian blur then downsample
            return torch.nn.functional.avg_pool1d(gauss(x), kernel_size=2, stride=2)

        def gaussian_upsample(x: Tensor) -> Tensor:
            # Use bicubic interpolation for better quality
            return torch.nn.functional.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)

        def build_pyramid(x: Tensor) -> List[Tensor]:
            gaussian = [x]
            laplacian = []

            # Build Gaussian pyramid
            for _ in range(n_levels - 1):
                gaussian.append(gaussian_downsample(gaussian[-1]))

            # Build Laplacian pyramid
            for i in range(n_levels - 1):
                upsampled = gaussian_upsample(gaussian[i + 1])
                # Pad/trim upsampled to match current level size
                if upsampled.shape != gaussian[i].shape:
                    diff = gaussian[i].shape[-1] - upsampled.shape[-1]
                    if diff > 0:
                        upsampled = torch.nn.functional.pad(upsampled, (0, diff))
                    else:
                        upsampled = upsampled[..., :gaussian[i].shape[-1]]
                laplacian.append(gaussian[i] - upsampled)

            # Add smallest Gaussian level as last Laplacian level
            laplacian.append(gaussian[-1])

            return laplacian

        # Build pyramids for all three models
        a_pyr = build_pyramid(a)
        b_pyr = build_pyramid(b)
        c_pyr = build_pyramid(c)

        merged_pyr = []
        for level in range(n_levels):
            # Adjust alpha based on level
            # More aggressive at lower frequencies (higher levels)
            level_alpha = alpha * (1.0 + level / (n_levels - 1))

            # Get changes from base for both models
            a_delta = a_pyr[level] - c_pyr[level]
            b_delta = b_pyr[level] - c_pyr[level]

            # Calculate agreement mask
            # Higher when changes are similar, lower when divergent
            agreement = torch.cosine_similarity(a_delta, b_delta, dim=-1, eps=1e-8)
            agreement = torch.clamp(agreement, 0, 1).unsqueeze(-1)

            # Calculate magnitude mask
            # Favor stronger changes but prevent extreme differences
            mag_a = torch.norm(a_delta, dim=-1, keepdim=True)
            mag_b = torch.norm(b_delta, dim=-1, keepdim=True)
            mag_ratio = torch.minimum(mag_a, mag_b) / torch.maximum(mag_a, mag_b).clamp(min=1e-8)

            # Combined mask considers both agreement and relative magnitudes
            mask = agreement * mag_ratio

            # For highest frequencies (level 0), be more conservative
            if level == 0:
                mask *= 0.5

            # Merge this level
            # Start with a's changes, blend in b's changes where mask is high
            merged_delta = a_delta * (1 - mask * level_alpha) + b_delta * (mask * level_alpha)
            merged_pyr.append(c_pyr[level] + merged_delta)

        # Reconstruct from pyramid
        result = merged_pyr[-1]
        for level in range(n_levels - 2, -1, -1):
            upsampled = gaussian_upsample(result)
            # Handle size mismatch
            if upsampled.shape != merged_pyr[level].shape:
                diff = merged_pyr[level].shape[-1] - upsampled.shape[-1]
                if diff > 0:
                    upsampled = torch.nn.functional.pad(upsampled, (0, diff))
                else:
                    upsampled = upsampled[..., :merged_pyr[level].shape[-1]]
            result = upsampled + merged_pyr[level]

        return result

    class LayerType2(Enum):
        ATTENTION_QKV = auto()
        ATTENTION_OUTPUT = auto()
        CONV_3X3 = auto()
        CONV_1X1 = auto()
        LINEAR = auto()
        NORM = auto()
        EMBEDDING = auto()
        TIME_EMBEDDING = auto()
        SCALAR = auto()

    @dataclass
    class LayerInfo:
        type: 'LayerType'
        shape: Tuple[int, ...]
        head_dim: Optional[int] = None
        num_heads: Optional[int] = None

    def identify_layer(key: str, tensor: Tensor) -> LayerInfo:
        """Identify layer type and extract relevant shape information"""
        shape = tuple(tensor.shape)

        # Handle attention layers specially
        if 'attn' in key:
            if any(x in key for x in ['query', 'key', 'value']):
                # Most SD models use 8 attention heads
                num_heads = 8 if len(shape) >= 2 else 1
                head_dim = shape[-1] // num_heads if len(shape) >= 2 else shape[-1]
                return LayerInfo(LayerType.ATTENTION_QKV, shape, head_dim, num_heads)
            if 'output' in key:
                return LayerInfo(LayerType.ATTENTION_OUTPUT, shape)

        # Convolution layers
        if len(shape) == 4:
            if shape[-1] == 1:
                return LayerInfo(LayerType.CONV_1X1, shape)
            return LayerInfo(LayerType.CONV_3X3, shape)

        # Other common layer types
        if 'norm' in key or 'ln_' in key:
            return LayerInfo(LayerType.NORM, shape)
        if 'emb' in key:
            if 'time' in key:
                return LayerInfo(LayerType.TIME_EMBEDDING, shape)
            return LayerInfo(LayerType.EMBEDDING, shape)
        if not shape:  # Scalar parameters
            return LayerInfo(LayerType.SCALAR, (1,))

        # Default to linear for other matrix operations
        return LayerInfo(LayerType.LINEAR, shape)

    def reshape_for_processing(tensor: Tensor, layer_info: LayerInfo) -> Tuple[Tensor, callable]:
        """Reshape tensor for processing while preserving structural information"""
        original_shape = tensor.shape

        if layer_info.type == LayerType.ATTENTION_QKV:
            # Preserve attention head structure
            reshaped = tensor.view(-1, layer_info.num_heads, layer_info.head_dim)
            restore_fn = lambda x: x.view(original_shape)
            return reshaped, restore_fn

        if layer_info.type in [LayerType.CONV_3X3, LayerType.CONV_1X1]:
            # Preserve channel structure for convolutions
            flat_shape = (-1, functools.reduce(operator.mul, original_shape[1:]))
            reshaped = tensor.view(flat_shape)
            restore_fn = lambda x: x.view(original_shape)
            return reshaped, restore_fn

        if layer_info.type == LayerType.NORM:
            # Keep norm parameters as is
            return tensor, lambda x: x

        if layer_info.type == LayerType.SCALAR:
            return tensor.view(1), lambda x: x.view(())

        # Default reshape for linear layers
        if len(original_shape) > 2:
            flat_shape = (-1, original_shape[-1])
            reshaped = tensor.view(flat_shape)
            restore_fn = lambda x: x.view(original_shape)
            return reshaped, restore_fn

        return tensor, lambda x: x
