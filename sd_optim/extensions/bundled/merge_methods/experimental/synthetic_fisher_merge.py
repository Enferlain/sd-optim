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
# def synthetic_fisher_merge(
#         a: Parameter(Tensor, "weight"),
#         b: Parameter(Tensor, "weight"),
#         *,
#         num_samples: Parameter(Tensor) =256,
#         noise_scale: Parameter(Tensor) =1.5,
#         epsilon: Parameter(Tensor) =1e-8,
#         use_diverse: float = 1.0,
#         **kwargs
# ) -> Return(Tensor):
#     """
#     v1.51 Fisher merge using synthetic data with bias mitigation
#
#     Features:
#     - Adversarial noise generation
#     - Entropy maximization
#     - Gradient clipping
#     - Sign alignment
#     """
#     key = kwargs["key"]
#
#     # Handle scalar parameters
#     if a.dim() == 0 or b.dim() == 0:
#         print("Scalar parameter detected, using simple average")
#         return 0.5 * a + 0.5 * b
#
#     # Ensure gradient tracking
#     a = a.detach().clone().requires_grad_(True)
#     b = b.detach().clone().requires_grad_(True)
#
#     if use_diverse == 1.0:
#         synthetic_input = MergeMethods.generate_diverse_input(a, num_samples, a.device).to(dtype=a.dtype)
#     else:
#         # Dimension-aware input generation
#         if a.dim() == 4:  # Conv2D
#             print(f"Conv2D tensor detected: {a.shape}")
#             # For conv layers: [batch, in_channels, kernel_h, kernel_w]
#             synthetic_input = torch.randn(num_samples, a.size(1),
#                                           max(8, a.size(2) * 2), max(8, a.size(3) * 2),
#                                           device=a.device, dtype=a.dtype) * noise_scale
#         elif a.dim() == 2:  # Linear
#             print(f"Linear tensor detected: {a.shape}")
#             # For linear layers: [batch, in_features] - IMPORTANT: Use size(1) not size(0)
#             synthetic_input = torch.randn(num_samples, a.size(1),
#                                           device=a.device, dtype=a.dtype) * noise_scale
#         elif a.dim() == 1:  # Bias
#             print(f"Bias tensor detected: {a.shape}")
#             # For bias terms: [batch, out_features]
#             synthetic_input = torch.randn(num_samples, a.size(0),
#                                           device=a.device, dtype=a.dtype) * noise_scale
#         else:
#             raise ValueError(f"Unsupported dimension {a.dim()}")
#
#     print(f"Generated synthetic input: {synthetic_input.shape}")
#
#     # Safe feature extraction with dimension checking
#     try:
#         if a.dim() == 4:
#             # For conv layers, use proper padding
#             feats_a = F.conv2d(synthetic_input, a, padding=a.size(2) // 2)
#             feats_b = F.conv2d(synthetic_input, b, padding=b.size(2) // 2)
#         elif a.dim() == 2:
#             # Check matrix dimensions
#             if a.size(0) != synthetic_input.size(1):
#                 print(f"Matrix dimension mismatch: {synthetic_input.shape} @ {a.shape}")
#                 # For SDXL transformer weights, we need to handle [out_features, in_features] format
#                 if a.size(1) == synthetic_input.size(1):
#                     # This is correct - we want to multiply with the transpose
#                     feats_a = synthetic_input @ a.t()
#                     feats_b = synthetic_input @ b.t()
#                 else:
#                     # If dimensions still don't match, regenerate with correct size
#                     synthetic_input = MergeMethods.generate_statistical_input(a, num_samples,
#                                                                  a.device) if use_statistical else torch.randn(
#                         num_samples, a.size(1), device=a.device, dtype=a.dtype) * noise_scale
#                     print(f"Regenerated input: {synthetic_input.shape}")
#                     feats_a = synthetic_input @ a.t()
#                     feats_b = synthetic_input @ b.t()
#             else:
#                 feats_a = synthetic_input @ a
#                 feats_b = synthetic_input @ b
#         else:  # Bias terms
#             # For bias, we add the bias to pre-activations
#             feats_a = synthetic_input + a.unsqueeze(0)
#             feats_b = synthetic_input + b.unsqueeze(0)
#     except RuntimeError as e:
#         print(f"Error in feature extraction: {e}")
#         print(f"Falling back to simple average for this tensor")
#         return 0.5 * a.detach() + 0.5 * b.detach()
#
#     print(f"Feature shapes: A={feats_a.shape}, B={feats_b.shape}")
#
#     def compute_fisher_improved(param, feats_a, feats_b, key=""):  # Pass feats_a and feats_b
#         """Enhanced Fisher computation with layer-aware adaptation, dynamic scaling and normalization."""
#         param.grad = None
#
#         with torch.enable_grad():
#
#             # Use feats_a and feats_b to compute pseudo_labels
#             comparison_noise = torch.randn_like(feats_a) * 0.3  # noise must have the same shape as feats_a
#             pseudo_labels = (feats_a > feats_b + comparison_noise).float()  # must have same shape as feats_a and b
#             print(f"Generated {pseudo_labels.sum().item()}/{feats_a.numel()} positive pseudo-labels")
#
#             focal_loss = F.binary_cross_entropy_with_logits(
#                 feats_a,  # [B, ...] Use original features, remove reduction. Same for feats_b
#                 pseudo_labels,  # [B, ...] Use per-element labels
#                 reduction='none'  # Keep per-sample loss, and per-element
#             )
#             loss = focal_loss.mean()  # Average over the batch
#             print(f"Loss: {loss.item():.4f}")
#
#         loss.backward(retain_graph=True)
#         grad = param.grad
#
#         avg_grad_magnitude = torch.mean(torch.abs(grad.detach())).item()
#         if avg_grad_magnitude < 1e-10:  # Increased threshold slightly
#             scale_factor = 1.0
#         else:
#             scale_factor = 1.0 / avg_grad_magnitude
#
#         # Layer-type specific adaptation (Keep as is)
#         layer_scale = 1.0
#         if param.dim() == 4:  # Convolutional layers
#             if 'op' in key or 'skip_connection' in key:
#                 layer_scale = 0.7  # Down-weight skip connections
#             elif 'out.2' in key:  # Final output conv
#                 layer_scale = 1.2
#             else:
#                 layer_scale = 1.2  # Other conv layers
#
#         elif param.dim() == 2:  # Linear layers
#             if 'ff.net' in key:
#                 layer_scale = 1.4 if 'proj' in key else 1.2
#             elif 'attn1' or 'attn2' in key:
#                 layer_scale = 1.6 if 'to_out' in key else 1.5
#             elif 'emb_layers' in key:  # Added emb layers
#                 layer_scale = 0.9
#             elif 'time_embed' in key:  # time embed, linear
#                 layer_scale = 1.0
#             elif 'out.0' in key:
#                 layer_scale = 1.0  # final output block linear
#             else:
#                 layer_scale = 1.0
#
#         elif param.dim() == 1:  # Bias terms
#             if 'norm' in key:
#                 layer_scale = 0.6
#             elif 'time_embed' in key:
#                 layer_scale = 0.8  # time_embed bias
#             else:
#                 layer_scale = 0.8  # Slightly down-weight biases
#
#         # --- Fisher Calculation and Normalization ---
#         fisher = grad.pow(2) * scale_factor * layer_scale
#
#         # Numerical stability (Adjust clamping, potentially making it less aggressive)
#         clamp_min = 0.0  # Fisher should be non-negative
#         clamp_max = 5e2  # Allow higher values for differentiation, though maybe don't need it
#         fisher = torch.clamp(fisher, min=clamp_min, max=clamp_max)  # Keep clamp
#
#         print(
#             f"Layer: {param.shape} | Type: {layer_scale}x | Avg Grad Mag: {avg_grad_magnitude:.4e} | Scale Factor: {scale_factor:.4e} | Fisher Mean: {fisher.mean().item():.4f} ±{fisher.std().item():.4f}")
#
#         return fisher
#
#     def diagonal_rescaling(fisher_diag, param, scaling_factor=1.0):  # CHANGED scaling_factor
#         param_scale = torch.var(param.detach(), dim=0, keepdim=True) + 1e-6
#         rescaled_fisher = fisher_diag / (param_scale * scaling_factor + 1e-6)
#         return rescaled_fisher  # REMOVED max normalization
#
#     print("Computing Fisher information...")
#     fisher_a = compute_fisher_improved(a, feats_a, feats_b, key=key)
#     fisher_b = compute_fisher_improved(b, feats_b, feats_a, key=key)  # FIXED
#
#     # After computing fisher_a and fisher_b
#     fisher_a = diagonal_rescaling(fisher_a, a)
#     fisher_b = diagonal_rescaling(fisher_b, b)
#
#     # Cleanup gradients
#     # Clear gradients more thoroughly
#     def clear_grads(*tensors):
#         for t in tensors:
#             if t.grad is not None:
#                 t.grad.detach_()
#                 t.grad = None
#
#     clear_grads(a, b, synthetic_input)
#
#     sign_mask = (a * b) > 0
#     sign_agreement = sign_mask.float().mean().item() * 100
#     print(f"Sign agreement: {sign_agreement:.1f}%")
#
#     # Adaptive merging with improved weighting
#     raw_weights = fisher_b / (fisher_a + fisher_b + epsilon)
#     clamped_weights = MergeMethods.adaptive_clamp(raw_weights, sign_agreement)
#     weights = torch.where(sign_mask, clamped_weights, 0.5)
#
#     if raw_weights.numel() == 1:
#         print(f"[{key}] Scalar weight: {raw_weights.item():.4f}")
#     else:
#         # Calculate statistics with sampling
#         mean_val = raw_weights.mean().item()
#         std_val = raw_weights.std().item() if raw_weights.numel() > 1 else 0.0
#
#         # Quantile calculation with sampling
#         if raw_weights.numel() > 10000:
#             flat = raw_weights.view(-1)
#             sample = flat[torch.randperm(flat.size(0), device=flat.device)[:10000]]
#             q25, q75 = torch.quantile(sample,
#                                       torch.tensor([0.25, 0.75], device=sample.device, dtype=raw_weights.dtype))
#         else:
#             q25, q75 = torch.quantile(raw_weights, torch.tensor([0.25, 0.75], device=raw_weights.device,
#                                                                 dtype=raw_weights.dtype))
#
#         # Formatting (keep existing)
#         print(f"""
#         Weight Analysis:
#           - Distribution: μ={mean_val:.4f} ±{std_val:.4f}
#           - Quartiles: 25%={q25:.4f} | 75%={q75:.4f}
#           - Range: [{raw_weights.min().item():.4f}, {raw_weights.max().item():.4f}]
#           - Fisher Ratio: {fisher_b.mean().item() / (fisher_a.mean().item() + 1e-8):.2f}
#         """)
#
#     merged = torch.where(sign_mask, (1 - weights) * a + weights * b, 0.5 * (a + b))
#     print("Weights:", weights)
#
#     return merged
#
# def adaptive_clamp(weights, sign_agreement):
#     agreement = sign_agreement / 100
#     # Allow mild extrapolation for high agreement cases
#     extrapolation_scale = 0.2  # 20% beyond 0-1 at 100% agreement
#
#     lower_bound = 0.0 - extrapolation_scale * agreement ** 3
#     upper_bound = 1.0 + extrapolation_scale * agreement ** 3
#
#     return torch.clamp(weights, lower_bound, upper_bound)
#
# def generate_diverse_input(param, num_samples=256, device='cuda'):
#     """Generate highly diverse synthetic inputs with multiple modes"""
#     if param.dim() == 4:  # Conv layers
#         n_channels = param.size(1)
#
#         # Create multi-modal distribution (4 distinct clusters)
#         clusters = []
#         for i in range(4):
#             # Each cluster has different mean and variance
#             mean_shift = torch.randn(1, n_channels, 1, 1, device=device) * (i + 1) / 2
#             var_scale = 0.5 + i * 0.5  # Different variance per cluster
#             cluster = torch.randn(num_samples // 4, n_channels,
#                                   max(8, param.size(2) * 2), max(8, param.size(3) * 2),
#                                   device=device) * var_scale + mean_shift
#             clusters.append(cluster)
#
#         return torch.cat(clusters, dim=0)
#
#     elif param.dim() <= 2:  # Linear/bias layers
#         n_features = param.size(-1) if param.dim() == 2 else param.size(0)
#
#         # Create adversarial inputs that maximize differences
#         base = torch.randn(num_samples // 4, n_features, device=device)
#
#         # Create 4 distinct input distributions
#         diverse_inputs = [
#             base,  # Standard normal
#             base * 2.0,  # Scaled inputs
#             torch.sign(base) * torch.abs(base).sqrt(),  # Non-linear transformation
#             torch.sin(base * 3.14159)  # Periodic transformation
#         ]
#
#         return torch.cat(diverse_inputs, dim=0)
