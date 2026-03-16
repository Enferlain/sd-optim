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
