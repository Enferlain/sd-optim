import sd_mecha
import functools
import operator
import torch
import math
import torch.nn.functional as F

from torch import Tensor
from typing import Tuple, TypeVar, Dict, Optional
from pytorch_wavelets import DWTForward, DWTInverse

from sd_mecha.hypers import Hyper
from sd_mecha.merge_space import MergeSpace
from sd_mecha.extensions.merge_method import LiftFlag, convert_to_recipe

EPSILON = 1e-10
SameMergeSpace = TypeVar("SameMergeSpace", bound=LiftFlag[MergeSpace.BASE | MergeSpace.DELTA])


class MergeMethods:
    @staticmethod
    def weighted_sum(a, b, alpha: Hyper, device=None):
        return sd_mecha.weighted_sum(a, b, alpha=alpha, device=device)

    @staticmethod
    def rotate(a, b, alignment: Hyper = 1.0, alpha: Hyper = 0.0, device=None, cache=None):  # Use None as default
        if cache is None:
            cache = {}  # Create a new dictionary if cache is None

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

    @staticmethod
    @convert_to_recipe
    def wavelet_merge(
        a: Tensor | SameMergeSpace,
        b: Tensor | SameMergeSpace,
        *,
        alpha: Hyper = 0.5,
        ** kwargs,
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
        ** kwargs,
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
