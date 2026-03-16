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


@merge_method
def svd_ties_sum_extended_v13(
    *models: Parameter(Tensor, "delta"),
    passthrough_index: Parameter(int) = 0,
    k: Parameter(float) = 1.0,
    max_singular_values: Parameter(int) = 64,
    energy_threshold: Parameter(float) = 0.9,
    power_iterations: Parameter(int) = 1,
    vote_sgn: Parameter(float) = 1.0,
    apply_stock: Parameter(float) = 0.0,
    cos_eps: Parameter(float) = 1e-6,
    apply_median: Parameter(float) = 1.0,
    eps: Parameter(float) = 1e-6,
    maxiter: Parameter(int) = 150,
    ftol: Parameter(float) = 1e-22,
    weight_decay: Parameter(float) = 0.0218,
    min_agreement: Parameter(float) = 0.3,
    memory_safety_margin: Parameter(float) = 0.8,
    **kwargs,
) -> Return(Tensor, "delta"):
    """
    Correctly implements hybrid chunking for massive tensors in concurrent environments.
    - Outer loop performs SPATIAL chunking (slicing large tensors).
    - Inner loop performs BATCH chunking (processing a few tensors at a time).
    - This robustly handles scenarios with many, very large input tensors.
    """
    if not models:
        raise ValueError("Onii-chan, you have to give me at least one model tensor!")
    layer_key = str(kwargs.get("key", "<unknown>"))

    if k == 0.0 and min_agreement == 0.0 and energy_threshold == 0.0:
        if 0 <= passthrough_index < len(models):
            return models[passthrough_index].clone()
        else:
            return torch.zeros_like(models[0])

    tensor_template = models[0]
    original_shape = tensor_template.shape
    original_ndim = tensor_template.ndim
    reshape_back = False

    if original_ndim <= 1:
        with torch.no_grad():
            stacked = torch.stack([m.to(torch.float32) for m in models])
            average_tensor = torch.mean(stacked, dim=0)
            return average_tensor.to(dtype=tensor_template.dtype)

    if original_ndim != 2:
        # Keep output channels/features (dim 0) separate for all multi-dimensional tensors:
        # [d0, d1, ...] -> [d0, d1*...]
        reshaped_models = [m.reshape(m.shape[0], -1) for m in models]
        models = tuple(reshaped_models)
        reshape_back = True

    device = models[0].device
    dtype = models[0].dtype
    total_tensors = len(models)
    passthrough_tensor = models[passthrough_index] if 0 <= passthrough_index < total_tensors else None

    tensor_batch_size, spatial_chunk_size = _get_optimized_chunks_v12(
        models[0], total_tensors, memory_safety_margin, dtype, max_singular_values
    )

    use_pinned_memory = device.type == "cuda" and torch.cuda.is_available()
    final_result_2d = torch.zeros_like(models[0], device="cpu", pin_memory=use_pinned_memory)

    with torch.no_grad():
        filtered_tensors = [filter_top_k_v2(m, k) for m in models]

        tensor_shape = models[0].shape
        chunk_dim = 0 if tensor_shape[0] >= tensor_shape[1] else 1
        tensor_len = tensor_shape[chunk_dim]

        for tensor_start in range(0, tensor_len, spatial_chunk_size):
            tensor_end = min(tensor_start + spatial_chunk_size, tensor_len)
            slice_obj = tuple(slice(tensor_start, tensor_end) if d == chunk_dim else slice(None) for d in range(len(tensor_shape)))

            collected_deltas = []
            for i in range(0, total_tensors, tensor_batch_size):
                batch_tensors_cpu = filtered_tensors[i : i + tensor_batch_size]
                batch_slices_gpu = torch.stack([t[slice_obj].to(device, non_blocking=True) for t in batch_tensors_cpu])

                reconstructed_batch = _approximate_svd_v2(
                    batch_slices_gpu,
                    max_rank=max_singular_values,
                    power_iterations=power_iterations,
                    energy_threshold=energy_threshold,
                    log_context=layer_key,
                )

                collected_deltas.append(reconstructed_batch)
                del batch_slices_gpu
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            filtered_delta = torch.cat(collected_deltas)
            signs = torch.sign(filtered_delta)
            passthrough_slice = passthrough_tensor[slice_obj] if passthrough_tensor is not None else None

            result_chunk = _compute_final_chunk_v2(
                filtered_delta,
                signs,
                vote_sgn,
                min_agreement,
                weight_decay,
                apply_stock,
                cos_eps,
                apply_median,
                eps,
                maxiter,
                ftol,
                tensor_batch_size,
                passthrough_slice,
            )

            final_result_2d[slice_obj] = result_chunk.to("cpu", non_blocking=True)

            del collected_deltas, filtered_delta, signs, result_chunk
            if device.type == "cuda":
                torch.cuda.empty_cache()

    final_result = final_result_2d.to(device)
    if reshape_back:
        final_result = final_result.reshape(original_shape)

    return final_result.nan_to_num(0.0, 0.0, 0.0)

def _get_optimized_chunks_v12(
    tensor_template: Tensor, total_tensors: int, margin: float, dtype: torch.dtype, max_rank: int
) -> tuple[int, int]:
    """
    An aggressive and more precise memory calculator.
    - It correctly identifies the single largest memory allocation.
    - It uses more precise formulas for memory costs.
    - Goal: Maximize VRAM utilization without crashing.
    """
    if tensor_template.device.type != "cuda":
        return total_tensors, max(tensor_template.shape)

    total_mem = torch.cuda.get_device_properties(tensor_template.device).total_memory
    free_mem = total_mem - torch.cuda.memory_allocated(tensor_template.device)
    usable_mem = free_mem * margin

    if tensor_template.ndim == 2:
        m, n = tensor_template.shape
    elif tensor_template.ndim > 2:
        m = tensor_template.shape[0]
        n = math.prod(tensor_template.shape[1:])
    else:
        # Scalar-like tensors are handled by caller, but keep this safe.
        return max(1, total_tensors), 1

    element_size = torch.tensor([], dtype=dtype).element_size()
    rank = min(m, n, max_rank)

    # --- Let's calculate the memory cost PER ROW for our two main operations ---

    # 1. Cost per row for the SVD step (for a single tensor in a batch)
    #    Cost is: (base_slice + A_matrix + B_matrix)
    #    For one row: (n_cols + rank_cols + n_cols)
    mem_per_row_svd = (n + rank + n) * element_size

    # 2. Cost per row for the final concatenated tensor (`filtered_delta` + `signs`)
    #    For one row: (n_cols * total_tensors * 2) because we hold both tensors.
    mem_per_row_final = (n * total_tensors * 2) * element_size

    # The LARGEST of these two determines our memory bottleneck.
    dominant_mem_per_row = max(mem_per_row_svd, mem_per_row_final)

    if dominant_mem_per_row == 0:
        safe_spatial_chunk = m
    else:
        # How many rows can we afford, based on the most expensive operation?
        num_rows_can_fit = math.floor(usable_mem / dominant_mem_per_row)
        safe_spatial_chunk = max(256, int(num_rows_can_fit))

    # With our new, aggressive spatial chunk, let's find the batch size.
    # We can now calculate the exact memory needed for SVD on one slice.
    current_m = min(safe_spatial_chunk, m)
    mem_for_one_svd_slice = (
        (current_m * n)  # Base slice
        + (current_m * rank)  # Matrix A
        + (n * rank)  # Matrix B
    ) * element_size

    if mem_for_one_svd_slice == 0:
        safe_batch_size = total_tensors
    else:
        # How many SVD operations can we stack in our memory budget?
        num_batches_can_fit = math.floor(usable_mem / mem_for_one_svd_slice)
        safe_batch_size = max(1, int(num_batches_can_fit))

    return max(1, safe_batch_size), max(1, min(safe_spatial_chunk, m))

# --- Helper methods from v2/v3 can be reused as they are clean ---
def filter_top_k_v2(a: Tensor, k: float) -> Tensor:
    if k >= 1.0:
        return a
    if k <= 0.0:
        return torch.zeros_like(a)
    flat_abs = a.abs().flatten()
    k_val = max(1, int((1.0 - k) * flat_abs.numel()))
    if k_val >= flat_abs.numel():
        return torch.zeros_like(a)
    threshold = torch.kthvalue(flat_abs, k_val).values
    return a * (a.abs() >= threshold)

def _approximate_svd_v2(
    matrices: Tensor,
    max_rank: int,
    power_iterations: int,
    energy_threshold: float,
    log_context: str | None = None,
) -> Tensor:
    def _safe_lstsq(lhs: Tensor, rhs: Tensor) -> Tensor:
        try:
            return torch.linalg.lstsq(lhs, rhs).solution
        except RuntimeError as e:
            # Rank-deficient batches can fail on some backends; use pinv fallback.
            solve_dtype = torch.float32 if lhs.dtype in (torch.float16, torch.bfloat16) else lhs.dtype
            lhs_solve = lhs.to(dtype=solve_dtype)
            rhs_solve = rhs.to(dtype=solve_dtype)
            solution = torch.linalg.pinv(lhs_solve) @ rhs_solve
            fallback_count = getattr(MergeMethods, "_svd_lstsq_fallback_count", 0) + 1
            setattr(MergeMethods, "_svd_lstsq_fallback_count", fallback_count)
            if fallback_count <= 5 or fallback_count % 100 == 0:
                logger.warning(
                    "lstsq fallback to pinv in _approximate_svd_v2 (count=%s, key=%s): %s",
                    fallback_count,
                    log_context or "<unknown>",
                    e,
                )
            return solution.to(dtype=rhs.dtype)

    if matrices.ndim < 2:
        return matrices
    if matrices.ndim == 2:
        matrices = matrices.unsqueeze(0)

    batch_size, m, n = matrices.shape
    rank = min(m, n, max_rank)
    if rank <= 0:
        return torch.zeros_like(matrices)

    A = torch.empty(batch_size, m, rank, device=matrices.device, dtype=matrices.dtype)
    torch.nn.init.orthogonal_(A)
    B = _safe_lstsq(A, matrices)

    for _ in range(max(0, int(power_iterations))):
        A = _safe_lstsq(B.mT, matrices.mT).mT
        B = _safe_lstsq(A, matrices)

    singular_values_sq = torch.sum(B**2, dim=2)
    total_energy = torch.sum(singular_values_sq, dim=-1, keepdim=True)
    energy_cumsum = torch.cumsum(singular_values_sq, dim=-1)
    safe_energy_threshold = max(0.0, min(1.0, float(energy_threshold)))
    rank_indices = torch.argmax((energy_cumsum >= safe_energy_threshold * total_energy).float(), dim=-1)
    final_rank = torch.median(rank_indices).int().clamp(min=1, max=rank).item()

    A_trunc, B_trunc = A[..., :final_rank], B[..., :final_rank, :]
    reconstructed = A_trunc @ B_trunc

    # --- THE CORRECTED SIGN ALIGNMENT LOGIC ---
    # We calculate the dot product for each matrix in the batch.
    # This is much more robust than my old, buggy method.
    dot_products = torch.sum(reconstructed * matrices, dim=(-1, -2), keepdim=True)
    signs = torch.sign(dot_products)

    # We use .detach() on the signs to ensure no weird gradients flow back, just in case.
    return reconstructed * signs.detach()

def _compute_final_chunk_v2(
    filtered_delta: Tensor,
    signs: Tensor,
    vote_sgn: float,
    min_agreement: float,
    weight_decay: float,
    apply_stock: float,
    cos_eps: float,
    apply_median: float,
    eps: float,
    maxiter: int,
    ftol: float,
    processing_batch_size: int,
    passthrough_slice: Tensor | None = None,
) -> Tensor:
    vote_tensor = signs if vote_sgn > 0.0 else filtered_delta
    sign_sum = torch.sum(vote_tensor, dim=0)
    agreement_mask = signs != 0
    agreement_ratio = agreement_mask.float().sum(dim=0) / signs.shape[0]
    final_sign = torch.sign(sign_sum)
    final_sign[agreement_ratio < min_agreement] = 0
    delta_filters = (signs == final_sign).float()
    param_counts = torch.sum(delta_filters, dim=0)
    if weight_decay > 0.0:
        filtered_delta = filtered_delta * (1.0 - weight_decay)
    filtered_delta *= delta_filters
    if apply_median > 0.0:
        result = _compute_geometric_median_chunked_v2(filtered_delta, eps, maxiter, ftol, processing_batch_size)
    else:
        t = 1.0
        if apply_stock > 0.0:
            t = _compute_model_stock_chunked_v2(filtered_delta, cos_eps, processing_batch_size)
        result = (filtered_delta.sum(dim=0) * t) / param_counts.clamp(min=eps)

    if passthrough_slice is not None:
        no_agreement_mask = param_counts <= eps
        if torch.any(no_agreement_mask):
            passthrough_slice = passthrough_slice.to(device=result.device, dtype=result.dtype)
            result = torch.where(no_agreement_mask, passthrough_slice, result)

    return result

# Note: The chunked median and stock methods are still useful for memory,
# so I just cleaned them up and renamed them to _v2.
def _compute_geometric_median_chunked_v2(points: Tensor, eps: float, maxiter: int, ftol: float, chunk_size: int) -> Tensor:
    """Optimized geometric median with chunking."""
    chunk_size = max(1, int(chunk_size))
    n_points, *dims = points.shape
    points_flat = points.view(n_points, -1)
    median = torch.mean(points_flat, dim=0)

    for _ in range(maxiter):
        prev_median = median.clone()

        weighted_sum = torch.zeros_like(median)
        weight_sum = torch.zeros(1, device=median.device)

        for i in range(0, n_points, chunk_size):
            chunk = points_flat[i : i + chunk_size]
            dist = torch.norm(chunk - median, dim=1)
            inv_dist = 1.0 / dist.clamp(min=eps)

            weighted_sum += torch.sum(chunk * inv_dist[:, None], dim=0)
            weight_sum += torch.sum(inv_dist)

        median = weighted_sum / weight_sum.clamp(min=eps)
        if torch.norm(median - prev_median) < ftol:
            break

    return median.view(*dims)

def _compute_model_stock_chunked_v2(filtered_delta: Tensor, cos_eps: float, chunk_size: int) -> float:
    """Memory-efficient cosine similarity calculation."""
    chunk_size = max(1, int(chunk_size))
    n_models = filtered_delta.shape[0]
    flat_delta = filtered_delta.flatten(1)
    total_sum = 0.0
    total_pairs = 0

    for i in range(0, n_models, chunk_size):
        chunk_i = flat_delta[i : i + chunk_size]
        norm_i = torch.norm(chunk_i, p=2, dim=1, keepdim=True)
        for j in range(i, n_models, chunk_size):
            chunk_j = flat_delta[j : j + chunk_size]
            norm_j = torch.norm(chunk_j, p=2, dim=1, keepdim=True)

            # Cosine similarity
            cos_sim = (chunk_i @ chunk_j.T) / ((norm_i @ norm_j.T) + cos_eps)

            # In the original, it seems you just wanted the positive ratio
            total_sum += torch.sum(cos_sim > 0).item()
            total_pairs += cos_sim.numel()

    return total_sum / total_pairs if total_pairs > 0 else 0.0

# @staticmethod
# @merge_method
# def ties_sum_with_dropout(
#         *models: Parameter(Tensor, "delta"),
#         probability: Parameter(Tensor) =0.9,
#         della_eps: Parameter(Tensor) =0.0,
#         rescale: Parameter(Tensor) =1.0,
#         lambda_scale: Parameter(Tensor) =2.0,
#         k: Parameter(Tensor) =0.218,
#         vote_sgn: Parameter(Tensor) =0,
#         apply_stock: Parameter(Tensor) =0.0,
#         cos_eps: Parameter(Tensor) =1e-6,
#         apply_median: Parameter(Tensor) =1.0,
#         eps: Parameter(Tensor) =1e-5,
#         maxiter: Parameter(Tensor) =150,
#         ftol: Parameter(Tensor) =1e-11,
#         seed: Parameter(Tensor) =218,
#         **kwargs,
# ) -> Return(Tensor, "delta"):
#     """
#     Applies TIES merging with dropout to a variable number of delta tensors.
#
#     Args:
#         *models: The delta tensors to merge.
#         probability: The dropout probability (0 <= probability <= 1).
#         della_eps: The DELLA epsilon parameter, controlling magnitude-based dropout.
#         rescale:  The rescaling factor for the merged delta.
#         k: The TIES parameter trimming threshold.
#         vote_sgn:  The TIES-SOUP mode activation parameter.
#         apply_stock:  The Model Stock activation parameter.
#         cos_eps: The cosine similarity epsilon for Model Stock.
#         apply_median:  The Geometric Median activation parameter.
#         eps: The epsilon for the Geometric Median calculation.
#         maxiter: The maximum number of iterations for Geometric Median.
#         ftol:  The tolerance for convergence for Geometric Median.
#         seed: The random seed for dropout.
#         **kwargs: Additional keyword arguments.
#
#     Returns:
#         The merged delta tensor.
#     """
#     if not models or probability == 1:
#         return torch.tensor(0.0, device=models[0].device if models else 'cpu')
#
#     device = models[0].device
#     generator = torch.Generator(device)
#     if seed is not None:
#         generator.manual_seed(seed)
#
#     # Apply dropout to each delta tensor
#     dropped_deltas = []
#     for delta in models:
#         dropout_mask = create_dropout_mask(delta, probability, della_eps, generator)
#         dropped_deltas.append(delta * dropout_mask)
#
#     # Apply TIES merging to the dropped deltas
#     merged_delta = streaming_ties_sum_extended.__wrapped__(
#         *dropped_deltas,
#         k=k,
#         vote_sgn=vote_sgn,
#         apply_stock=apply_stock,
#         cos_eps=cos_eps,
#         apply_median=apply_median,
#         eps=eps,
#         maxiter=maxiter,
#         ftol=ftol
#     )
#
#     active_ratio = 1.0 - probability
#     rescalar = 1.0 / (active_ratio ** rescale + 1e-7)
#     return merged_delta * rescalar * lambda_scale
#
# def create_dropout_mask(delta: Tensor, probability: float, della_eps: float, generator: torch.Generator) -> Tensor:
#     """Paper-correct MAGPRUNE dropout mask."""
#     # 1. Descending magnitude ranking
#     flat_abs = delta.abs().flatten()
#     ranks = torch.argsort(flat_abs, descending=True).argsort().float().reshape(delta.shape)
#
#     # 2. Paper's DELLA formula (Section 3.2)
#     n = delta.numel()
#     median_rank = n // 2
#     delta_i = (ranks - median_rank) * della_eps / n
#
#     # 3. Clamp probabilities
#     p_min = 1 - probability
#     probabilities = torch.clamp(p_min + delta_i, 0.0, 1.0)
#
#     return torch.bernoulli(probabilities, generator=generator)
