import sd_mecha
import functools
import operator
import torch
import math
import torch.nn.functional as F

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

    ### CUSTOM METHODS ###


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
    @convert_to_recipe
    def clyb_merge(
            a: Tensor | SameMergeSpace,
            b: Tensor | SameMergeSpace,
            c: Tensor | SameMergeSpace,
            *,
            alpha: Hyper = 1.0,
            use_perp: Hyper = 0,
            ab_only: Hyper = 0,
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
        res = MergeMethods.clyb_align(ac, bc)

        # Interpolate between original tensor 'b' and the merged result based on 'alpha'
        if ab_only:
            return torch.lerp(b.reshape(original_shape), res.reshape(original_shape), alpha)
        else:
            return torch.lerp(b.reshape(original_shape), (c + res).reshape(original_shape), alpha)

    def clyb_align(a, b):
        """
        Performs the core merging operation using QR decomposition, low-rank approximation, and orthogonal projection.

        Args:
            a (Tensor): The source tensor (2D).
            b (Tensor): The target tensor (2D).

        Returns:
            Tensor: The merged tensor (2D).
        """
        compression = 16
        Qb, _ = torch.qr(b)
        q_size = max(int(torch.linalg.matrix_rank(Qb)) // compression, 1)
        iters = min(max(int(math.exp(math.log(640 / q_size))), 2), 64)

        print(q_size, iters)

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
        """
        Merges tensors 'a' and 'b' using a weighted sum based on vector projection, with the 'perplexity'
        hyperparameter controlling the level of granularity in the projection calculation.

        Args:
            a (Tensor): The first tensor.
            b (Tensor): The second tensor.
            c (Tensor): The reference tensor.
            perplexity (float): A hyperparameter that controls the blending between different levels of
                                projection granularity (0.0 <= perplexity <= 1.0).

                                - perplexity = 0.0: Merging is based on a single alpha value calculated for the entire key.
                                - perplexity = 0.5: Merging is based on alpha values calculated for each neuron.
                                - perplexity = 1.0: Merging is based on alpha values calculated for each individual parameter.
            **kwargs: Additional keyword arguments.

        Returns:
            Tensor: The merged tensor.
        """
        # Handle special case for concatenated attention projection layers
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

def advanced_original_merge(tensor1, tensor2, lambda_val=0.5, p=0.5, beta=1.0, gamma=0.5):
    """
    Advanced Original method for tensor merging.

    Args:
    tensor1, tensor2: Input tensors to be merged
    lambda_val: Blending factor (default: 0.5)
    p: Probability for the binomial distribution (default: 0.5)
    beta: Adaptive weight function parameter (default: 1.0)
    gamma: Weight smoothing parameter (default: 0.5)

    Returns:
    merged_tensor: The merged tensor
    """
    # Ensure tensors are on the same device and have the same dtype
    device = tensor1.device
    dtype = tensor1.dtype
    tensor2 = tensor2.to(device).to(dtype)

    # Compute the difference between the tensors
    delta = tensor2 - tensor1

    # Compute adaptive weights using a non-linear function (sigmoid)
    adaptive_weights = torch.sigmoid(beta * delta)

    # Compute the binomial mask
    m = torch.from_numpy(np.random.binomial(1, p, delta.shape)).to(device).to(dtype)

    # Apply the binomial mask to the adaptive weights
    weighted_delta = m * adaptive_weights * delta

    # Apply weight smoothing
    smoothed_weights = gamma * adaptive_weights + (1 - gamma) * 0.5

    # Merge the tensors
    merged_tensor = tensor1 + lambda_val * smoothed_weights * weighted_delta

    return merged_tensor


def auto_adaptive_gradient_based_fusion(tensor1, tensor2, temperature=1.0, min_smoothing_factor=0.1,
                                        max_smoothing_factor=1.5):
    """
    Auto-Adaptive Gradient-Based Fusion method for tensor merging.

    Args:
    tensor1, tensor2: Input tensors to be merged
    temperature: Temperature for softmax (default: 1.0)
    min_smoothing_factor: Minimum smoothing factor (default: 0.1)
    max_smoothing_factor: Maximum smoothing factor (default: 0.9)

    Returns:
    merged_tensor: The merged tensor
    """
    # Compute gradient difference
    grad_diff = torch.abs(tensor1 - tensor2)

    # Compute dynamic weights using softmax
    weights = F.softmax(grad_diff / temperature, dim=-1)

    # Apply weights
    merged_tensor = weights * tensor1 + (1 - weights) * tensor2

    # Compute adaptive smoothing factor based on gradient difference
    adaptive_smoothing_factor = min_smoothing_factor + (max_smoothing_factor - min_smoothing_factor) * torch.sigmoid(
        grad_diff)

    # Apply adaptive smoothing
    smoothed_tensor = adaptive_smoothing_factor * merged_tensor + (1 - adaptive_smoothing_factor) * (
                tensor1 + tensor2) / 2

    return smoothed_tensor


def quantum_inspired_tensor_fusion(tensor1, tensor2, entanglement_factor=0.5, superposition_threshold=0.1,
                                   scaling_factor=1.0):
    """
    Improved Quantum-Inspired Tensor Fusion method for merging two tensors.

    Args:
    tensor1, tensor2: Input tensors to be merged
    entanglement_factor: Controls the strength of entanglement (default: 0.5)
    superposition_threshold: Threshold for superposition effects (default: 0.1)
    scaling_factor: Factor to scale the final result (default: 1.0)

    Returns:
    merged_tensor: The merged tensor
    """
    # Ensure tensors are on the same device and have the same dtype
    dtype = tensor1.dtype
    tensor1 = tensor1.to(torch.float32)
    tensor2 = tensor2.to(torch.float32)

    # Normalize input tensors
    tensor1_norm = tensor1 / (torch.norm(tensor1) + 1e-8)
    tensor2_norm = tensor2 / (torch.norm(tensor2) + 1e-8)

    # Compute the quantum state representation
    quantum_state = torch.complex(tensor1_norm, tensor2_norm)

    # Apply entanglement
    entangled_state = torch.exp(1j * np.pi * entanglement_factor * quantum_state)

    # Compute the interference pattern
    interference = torch.abs(entangled_state) ** 2

    # Apply superposition effects
    superposition_mask = (interference > superposition_threshold).float()
    superposition_state = superposition_mask * quantum_state.real + (1 - superposition_mask) * quantum_state.imag

    # Compute the final merged tensor
    merged_tensor = torch.real(superposition_state * torch.conj(entangled_state))

    # Rescale to original magnitude
    original_magnitude = (torch.norm(tensor1) + torch.norm(tensor2)) / 2
    merged_tensor = merged_tensor * (original_magnitude / (torch.norm(merged_tensor) + 1e-8))

    # Apply additional scaling
    merged_tensor = merged_tensor * scaling_factor

    # Ensure the output is in the same range as the inputs
    min_val = torch.min(torch.min(tensor1), torch.min(tensor2))
    max_val = torch.max(torch.max(tensor1), torch.max(tensor2))
    merged_tensor = torch.clamp(merged_tensor, min_val, max_val)

    return merged_tensor.to(dtype)


def momentum_elastic_fusion(tensor1, tensor2, momentum=0.9, elasticity=0.5, iterations=5):
    """
    Momentum-based Elastic Fusion (MEF) method for tensor merging.

    Args:
    tensor1, tensor2: Input tensors to be merged
    momentum: Controls the influence of previous iterations (default: 0.9)
    elasticity: Controls the resistance to change (default: 0.5)
    iterations: Number of fusion iterations (default: 5)

    Returns:
    merged_tensor: The merged tensor
    """
    # Ensure tensors are on the same device and have the same dtype
    device = tensor1.device
    dtype = tensor1.dtype
    tensor2 = tensor2.to(device).to(dtype)

    # Convert to float32 for calculations if needed
    if tensor1.dtype in [torch.float16, torch.bfloat16]:
        tensor1 = tensor1.to(torch.float32)
        tensor2 = tensor2.to(torch.float32)

    # Initialize merged tensor and velocity
    merged_tensor = (tensor1 + tensor2) / 2
    velocity = torch.zeros_like(merged_tensor)

    for _ in range(iterations):
        # Compute elastic force
        force1 = elasticity * (tensor1 - merged_tensor)
        force2 = elasticity * (tensor2 - merged_tensor)

        # Update velocity using momentum
        velocity = momentum * velocity + (force1 + force2)

        # Update merged tensor
        merged_tensor = merged_tensor + velocity

    # Ensure the merged tensor stays within the original range
    min_val = torch.min(torch.min(tensor1), torch.min(tensor2))
    max_val = torch.max(torch.max(tensor1), torch.max(tensor2))
    merged_tensor = torch.clamp(merged_tensor, min_val, max_val)

    # Convert back to original dtype if needed
    merged_tensor = merged_tensor.to(dtype)

    return merged_tensor


def adaptive_gradient_based_fusion(tensor1, tensor2, temperature=1.0, smoothing_factor=0.5):
    """
    Adaptive Gradient-Based Fusion method for tensor merging.

    Args:
    tensor1, tensor2: Input tensors to be merged
    temperature: Temperature for softmax (default: 1.0)
    smoothing_factor: Smoothing factor for adaptive smoothing (default: 0.5)

    Returns:
    merged_tensor: The merged tensor
    """
    # Compute gradient difference
    grad_diff = torch.abs(tensor1 - tensor2)

    # Compute dynamic weights using softmax
    weights = F.softmax(grad_diff / temperature, dim=-1)

    # Apply weights
    merged_tensor = weights * tensor1 + (1 - weights) * tensor2

    # Apply adaptive smoothing based on gradient difference
    adaptive_smoothing = smoothing_factor * torch.sigmoid(grad_diff)
    smoothed_tensor = adaptive_smoothing * merged_tensor + (1 - adaptive_smoothing) * (tensor1 + tensor2) / 2

    return smoothed_tensor


def non_linear_activation_merge(tensor1, tensor2, alpha=0.5, activation='tanh'):
    """
    Non-linear Activation Merge (NAM) method for tensor merging.

    Args:
    tensor1, tensor2: Input tensors to be merged
    alpha: Balance factor between linear and non-linear components (default: 0.5)
    activation: Non-linear activation function to use ('tanh', 'sigmoid', or 'relu') (default: 'tanh')

    Returns:
    merged_tensor: The merged tensor
    """
    device = tensor1.device
    dtype = tensor1.dtype
    tensor2 = tensor2.to(device).to(dtype)

    if activation == 'tanh':
        act_fn = torch.tanh
    elif activation == 'sigmoid':
        act_fn = torch.sigmoid
    elif activation == 'relu':
        act_fn = torch.relu
    else:
        raise ValueError("Unsupported activation function")

    diff = tensor2 - tensor1
    activated_diff = act_fn(diff)

    merged_tensor = tensor1 + alpha * diff + (1 - alpha) * activated_diff

    return merged_tensor


def statistical_moment_guided_merge(tensor1, tensor2, alpha=0.5, beta=1.0):
    """
    Statistical Moment Guided Merge (SMGM) method for tensor merging.

    Args:
    tensor1, tensor2: Input tensors to be merged
    alpha: Weight for the first moment (mean) guidance (default: 0.5)
    beta: Weight for the second moment (variance) guidance (default: 1.0)

    Returns:
    merged_tensor: The merged tensor
    """
    # Ensure tensors are on the same device and have the same dtype
    device = tensor1.device
    dtype = tensor1.dtype
    tensor2 = tensor2.to(device).to(dtype)

    # Compute statistical moments
    mean1, mean2 = tensor1.mean(), tensor2.mean()
    var1, var2 = tensor1.var(), tensor2.var()

    # Compute moment-based weights
    mean_weight = torch.sigmoid(alpha * (mean1 - mean2))
    var_weight = torch.sigmoid(beta * (var1 - var2))

    # Combine weights
    combined_weight = (mean_weight + var_weight) / 2

    # Merge tensors
    merged_tensor = combined_weight * tensor1 + (1 - combined_weight) * tensor2

    # Adjust the merged tensor to preserve the overall statistical properties
    merged_mean = merged_tensor.mean()
    merged_var = merged_tensor.var()
    target_mean = (mean1 + mean2) / 2
    target_var = (var1 + var2) / 2

    # Scale and shift to match target statistics
    scaled_tensor = (merged_tensor - merged_mean) * torch.sqrt(target_var / merged_var) + target_mean

    return scaled_tensor


def statistical_distribution_alignment_merge(tensor1, tensor2, weight=0.5):
    """
    統計分佈對齊合併方法。

    參數：
    tensor1, tensor2：要合併的輸入張量
    weight：合併時的權重（默認為0.5）

    返回：
    merged_tensor：合併後的張量
    """
    # 確保張量在同一設備和數據類型上
    device = tensor1.device
    dtype = tensor1.dtype
    tensor2 = tensor2.to(device).to(dtype)

    # 計算第一個張量的均值和標準差
    mean1 = tensor1.mean()
    std1 = tensor1.std()

    # 計算第二個張量的均值和標準差
    mean2 = tensor2.mean()
    std2 = tensor2.std()

    # 對齊第二個張量的分佈到第一個張量
    adjusted_tensor2 = (tensor2 - mean2) / (std2 + 1e-6) * (std1 + 1e-6) + mean1

    # 合併張量
    merged_tensor = weight * tensor1 + (1 - weight) * adjusted_tensor2

    return merged_tensor


def local_linear_interpolation_merge(tensor1, tensor2, alpha=0.5, threshold=0.1):
    """
    局部線性插值合併方法

    Args:
        tensor1, tensor2: 要合併的兩個張量
        alpha: 全局合併係數，範圍 [0, 1]，默認為 0.5
        threshold: 差異閾值，超過該值的元素會動態調整權重，默認為 0.1

    Returns:
        merged_tensor: 合併後的張量
    """
    # 確保兩個張量形狀相同，否則報錯
    if tensor1.shape != tensor2.shape:
        raise ValueError("Input tensors must have the same shape")

    # 計算兩個張量之間的絕對差異
    diff = torch.abs(tensor1 - tensor2)

    # 根據差異動態計算權重
    dynamic_weight = torch.where(diff < threshold, torch.tensor(alpha), torch.tensor(0.5))

    # 局部線性插值
    merged_tensor = dynamic_weight * tensor1 + (1 - dynamic_weight) * tensor2

    return merged_tensor


def adaptive_merge_tensors(tensor1, tensor2, alpha):
    delta = tensor2 - tensor1
    adaptive_weight = torch.sigmoid(torch.abs(delta))
    merged_tensor = tensor1 + alpha * adaptive_weight * delta
    return merged_tensor


def enhanced_dynamic_weight_distribution(tensor1, tensor2, gamma=0.5, epsilon=1e-6, temperature=1.0):
    """
    Enhanced Dynamic Weight Distribution method for tensor merging.

    Args:
    tensor1, tensor2: Input tensors to be merged
    gamma: Global smoothing factor (default: 0.5)
    epsilon: Small value to avoid division by zero (default: 1e-6)
    temperature: Temperature for softmax (default: 1.0)

    Returns:
    merged_tensor: The merged tensor
    """
    # Compute relative importance of each element
    importance1 = torch.abs(tensor1) / (torch.sum(torch.abs(tensor1)) + epsilon)
    importance2 = torch.abs(tensor2) / (torch.sum(torch.abs(tensor2)) + epsilon)

    # Compute element-wise difference
    diff = torch.abs(tensor1 - tensor2)

    # Compute dynamic weights using softmax with temperature
    weights = F.softmax(torch.stack([importance1, importance2]) / temperature, dim=0)

    # Apply dynamic weights
    merged_tensor = weights[0] * tensor1 + weights[1] * tensor2

    # Apply adaptive smoothing based on element-wise difference
    adaptive_gamma = gamma * torch.sigmoid(diff)
    smoothed_tensor = adaptive_gamma * merged_tensor + (1 - adaptive_gamma) * (tensor1 + tensor2) / 2

    return smoothed_tensor


def gradient_weighted_merge(tensor1, tensor2, temperature=1.0):
    """
    Gradient-Weighted Merge method for tensor merging.

    Args:
    tensor1, tensor2: Input tensors to be merged
    temperature: Temperature for softmax (default: 1.0)

    Returns:
    merged_tensor: The merged tensor
    """
    # Compute gradient difference
    grad_diff = torch.abs(tensor1 - tensor2)

    # Compute weights using softmax
    weights = F.softmax(grad_diff / temperature, dim=-1)

    # Apply weights
    merged_tensor = weights * tensor1 + (1 - weights) * tensor2

    return merged_tensor


def adaptive_frequency_domain_merge(tensor1, tensor2, alpha=0.5, beta=2.0):
    """
    Adaptive Frequency Domain Merge method for tensor merging.

    Args:
    tensor1, tensor2: Input tensors to be merged
    alpha: Balance between low and high frequencies (default: 0.5)
    beta: Sharpness of the frequency response curve (default: 2.0)

    Returns:
    merged_tensor: The merged tensor
    """
    # Ensure tensors are on the same device and have the same dtype
    device = tensor1.device
    dtype = tensor1.dtype
    tensor2 = tensor2.to(device).to(dtype)

    # Compute frequency response in spatial domain
    diff = torch.abs(tensor1 - tensor2)
    freq_response = 1 / (1 + torch.exp(-beta * (diff - alpha)))

    # Merge tensors
    merged_tensor = freq_response * tensor1 + (1 - freq_response) * tensor2

    return merged_tensor


def dynamic_element_wise_fusion(tensor1, tensor2, temperature=1.0):
    """
    Dynamic Element-wise Fusion method for tensor merging.

    Args:
    tensor1, tensor2: Input tensors to be merged
    temperature: Temperature for softmax (default: 1.0)

    Returns:
    merged_tensor: The merged tensor
    """
    # Compute importance of each element
    importance1 = torch.abs(tensor1) / (torch.sum(torch.abs(tensor1)) + 1e-6)
    importance2 = torch.abs(tensor2) / (torch.sum(torch.abs(tensor2)) + 1e-6)

    # Compute dynamic weights using softmax with temperature
    weights = F.softmax(torch.stack([importance1, importance2]) / temperature, dim=0)

    # Apply dynamic weights
    merged_tensor = weights[0] * tensor1 + weights[1] * tensor2

    return merged_tensor


def dual_extremum_merging(tensor1, tensor2):
    """
    Dual Extremum Merging method for tensor merging.

    Args:
    tensor1, tensor2: Input tensors to be merged

    Returns:
    merged_tensor: The merged tensor
    """
    # Ensure tensors are on the same device and have the same dtype
    device = tensor1.device
    dtype = tensor1.dtype
    tensor2 = tensor2.to(device).to(dtype)

    # Compute max and min weights
    max_weight = torch.max(tensor1) / (torch.max(tensor1) + torch.max(tensor2))
    min_weight = torch.min(tensor1) / (torch.min(tensor1) + torch.min(tensor2))

    # Compute merging weights
    weight1 = (max_weight + min_weight) / 2
    weight2 = 1 - weight1

    # Merge tensors
    merged_tensor = weight1 * tensor1 + weight2 * tensor2

    return merged_tensor


def adaptive_harmonic_blending(tensor1, tensor2, alpha=0.5, beta=1.0, gamma=0.5):
    """
    Adaptive Harmonic Blending method for tensor merging.

    Args:
    tensor1, tensor2: Input tensors to be merged
    alpha: Blending factor (default: 0.5)
    beta: Harmonic function parameter (default: 1.0)
    gamma: Adaptive weight adjustment factor (default: 0.5)

    Returns:
    merged_tensor: The merged tensor
    """
    # Ensure tensors are on the same device and have the same dtype
    device = tensor1.device
    dtype = tensor1.dtype
    tensor2 = tensor2.to(device).to(dtype)

    # Compute the difference between the tensors
    delta = tensor2 - tensor1

    # Compute harmonic weights using a sinusoidal function
    harmonic_weights = 0.5 * (1 + torch.sin(beta * delta))

    # Compute adaptive weights based on the difference
    adaptive_weights = torch.sigmoid(gamma * delta)

    # Combine harmonic and adaptive weights
    combined_weights = alpha * harmonic_weights + (1 - alpha) * adaptive_weights

    # Apply combined weights to blend the tensors
    merged_tensor = combined_weights * tensor1 + (1 - combined_weights) * tensor2

    return merged_tensor


def synergistic_resonance_amplification(tensor1, tensor2, resonance_factor=0.1, amplification_threshold=0.5):
    """
    Synergistic Resonance Amplification (SRA) method for tensor merging.

    Args:
    tensor1, tensor2: Input tensors to be merged
    resonance_factor: Factor controlling the strength of resonance (default: 0.1)
    amplification_threshold: Threshold for amplification (default: 0.5)

    Returns:
    merged_tensor: The merged tensor
    """
    device = tensor1.device
    dtype = tensor1.dtype
    tensor2 = tensor2.to(device).to(dtype)

    # Compute the resonance
    resonance = torch.abs(tensor1 - tensor2) * resonance_factor

    # Compute the amplification mask
    amplification_mask = (resonance > amplification_threshold).float()

    # Merge tensors with resonance and amplification
    merged_tensor = (tensor1 + tensor2) / 2 + amplification_mask * resonance * torch.sign(tensor1 + tensor2)

    # Ensure the output is in the same range as the inputs
    min_val = torch.min(torch.min(tensor1), torch.min(tensor2))
    max_val = torch.max(torch.max(tensor1), torch.max(tensor2))
    merged_tensor = torch.clamp(merged_tensor, min_val, max_val)

    return merged_tensor