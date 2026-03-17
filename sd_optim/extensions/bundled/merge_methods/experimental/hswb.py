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

from sd_optim.extensions.bundled.merge_methods import torch_svd_lowrank
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
def hswb_merge(
        *deltas: Parameter(Tensor, "delta"),
        hessian_curvature_threshold: Parameter(float) = 0.1,
        parallel_reinforcement: Parameter(float) = 1.0,
        orthogonal_contribution: Parameter(float) = 1.0,
        num_projections: Parameter(int) = 128,
        **kwargs,
) -> Return(Tensor, "delta"):
    """The final H-SWB Merge with landscape reconstruction and proper importance weighting."""
    if not deltas:
        raise ValueError("This function received no deltas.")

    key = kwargs["key"]
    core_indices = [1, 3, 4, 6, 7, 8, 9]  # Fixed: removed EPS model (index 10)

    # NO FILTERING - work with original deltas
    cleaned_deltas = list(deltas)

    num_deltas = len(cleaned_deltas)
    valid_core_indices = [i for i in core_indices if i < num_deltas]
    if not valid_core_indices:
        valid_core_indices = list(range(num_deltas))

    logger.info("=== DELTA INDEX MAPPING ===")
    logger.info("Total deltas: %s", num_deltas)
    logger.info("Core indices: %s", valid_core_indices)
    outlier_indices = [i for i in range(num_deltas) if i not in valid_core_indices]
    logger.info("Outlier indices: %s", outlier_indices)

    core_deltas = [cleaned_deltas[i] for i in valid_core_indices]
    outlier_deltas = [d for i, d in enumerate(cleaned_deltas) if i not in valid_core_indices]

    logger.info("=== OUTLIER MODELS ===")
    for i, idx in enumerate(outlier_indices):
        logger.info("OUTLIER_%s = Delta index %s", i, idx)

    logger.info("Expected shadowforge at index 10, got outlier indices: %s", outlier_indices)

    if not core_deltas:
        return torch.mean(torch.stack(cleaned_deltas), dim=0)

    delta_core, _ = torch.median(torch.stack(core_deltas), dim=0)

    # Process outliers into parallel/perpendicular components
    parallel_components = []
    perpendicular_components = []

    track_tensor_quality(delta_core, "DELTA_CORE", key)

    for i, outlier in enumerate(outlier_deltas):
        track_tensor_quality(outlier, f"OUTLIER_{i}", key)

        parallel = _hswb_projection(outlier, delta_core)
        track_tensor_quality(parallel, f"PARALLEL_{i}", key)

        if torch.isnan(parallel).any():
            logger.warning("NaN detected in parallel projection, using zero instead")
            parallel = torch.zeros_like(outlier)

        perpendicular = outlier - parallel
        track_tensor_quality(perpendicular, f"PERPENDICULAR_{i}", key)

        if torch.isnan(perpendicular).any():
            logger.warning("NaN detected in perpendicular, using original outlier")
            perpendicular = outlier
            parallel = torch.zeros_like(outlier)

        parallel_components.append(parallel)
        perpendicular_components.append(perpendicular)

    # Merge components
    if parallel_components:
        merged_parallel = torch.mean(torch.stack(parallel_components), dim=0)
    else:
        merged_parallel = torch.zeros_like(delta_core)

    if not perpendicular_components:
        merged_perpendicular = torch.zeros_like(delta_core)
    else:
        merged_perpendicular = _hswb_swd_barycenter(
            tensors=perpendicular_components,
            reference_tensor=delta_core,
            num_projections=num_projections,
        )

    # LANDSCAPE RECONSTRUCTION: Get importance weights from Hessian approximation
    importance_weights = _hswb_reconstruct_hessian_diag(list(deltas))

    # Apply threshold to importance weights
    importance_mask = (importance_weights >= hessian_curvature_threshold).to(dtype=delta_core.dtype)
    final_importance = importance_weights * importance_mask

    # Add tracking after merging
    track_tensor_quality(merged_parallel, "MERGED_PARALLEL", key)
    track_tensor_quality(merged_perpendicular, "MERGED_PERPENDICULAR", key)
    track_tensor_quality(final_importance, "FINAL_IMPORTANCE", key)

    # Print norm statistics
    core_norm = torch.norm(delta_core)
    parallel_norm = torch.norm(merged_parallel) if parallel_components else 0
    perp_norm = torch.norm(merged_perpendicular) if perpendicular_components else 0
    importance_norm = torch.norm(final_importance)

    logger.info("Core norm: %.6f", core_norm)
    logger.info("Parallel norm: %.6f", parallel_norm)
    logger.info("Perpendicular norm: %.6f", perp_norm)
    logger.info("Importance norm: %.6f", importance_norm)
    logger.info("Parallel ratio: %.3f", parallel_norm / core_norm)
    logger.info("Perpendicular ratio: %.3f", perp_norm / core_norm)
    logger.info("Importance ratio: %.3f", importance_norm / core_norm)

    # FINAL COMBINATION: Use importance weights for final combination
    final_delta = delta_core + final_importance * (orthogonal_contribution * merged_perpendicular)

    # Add final tracking
    track_tensor_quality(final_delta, "FINAL_DELTA", key)

    return final_delta

def _hswb_reconstruct_hessian_diag(deltas: list[Tensor]) -> Tensor:
    """
    Reconstructs Hessian diagonal by finding the quadratic bowl that best
    explains the geometric arrangement of the input deltas.
    """
    if len(deltas) < 2:
        return torch.ones_like(deltas[0])

    device = deltas[0].device
    num_params = deltas[0].numel()
    num_models = len(deltas)

    # Initialize: Hessian diagonal + individual energy levels per model
    hessian_diag_candidate = torch.ones(num_params, device=device, requires_grad=True)
    energy_levels = torch.ones(num_models, device=device, requires_grad=True)

    optimizer = torch.optim.LBFGS([hessian_diag_candidate, energy_levels], max_iter=20, history_size=10)

    # Pre-compute squared deltas for efficiency
    deltas_sq = torch.stack([d.flatten() ** 2 for d in deltas])

    def closure():
        optimizer.zero_grad()

        # Ensure positive values (curvature can't be negative)
        positive_hessian = torch.nn.functional.softplus(hessian_diag_candidate)
        positive_energies = torch.nn.functional.softplus(energy_levels)

        # Calculate predicted energy for each model: E = sum(H * d^2)
        predicted_energies = torch.sum(positive_hessian * deltas_sq, dim=1)

        # Loss: How well does our Hessian explain the model positions?
        energy_errors = predicted_energies - positive_energies

        # Use robust loss (less sensitive to outlier models)
        loss = torch.mean(torch.abs(energy_errors))  # L1 instead of L2

        loss.backward()
        return loss

    optimizer.step(closure)

    # Return normalized importance weights
    final_hessian = torch.nn.functional.softplus(hessian_diag_candidate).detach()
    max_importance = torch.max(final_hessian)
    if max_importance > 0:
        importance_weights = final_hessian / max_importance
    else:
        importance_weights = torch.ones_like(final_hessian)

    return importance_weights.reshape(deltas[0].shape)

# --- Helper 2: Vector Projection ---
def _hswb_projection(vector_to_project: Tensor, target_vector: Tensor) -> Tensor:
    """Calculates the projection of one vector onto another."""
    target_norm_sq = torch.sum(target_vecto r* *2)
    if target_norm_sq < 1e-12:
        return torch.zeros_like(vector_to_project)
    dot_product = torch.sum(vector_to_project * target_vector)
    return target_vector * (dot_product / target_norm_sq)

# --- Helper 3: The Reconstruction Engine (The Real, Iterative Version) ---
def _hswb_reconstruct_from_projections(target_projections: Tensor, projection_dirs: Tensor, initial_guess: Tensor) -> Tensor:
    """
    Reconstructs a high-dimensional tensor from its target 1D projections using LBFGS optimization.
    This is the "sculpting" process that finds the true optimal barycenter.
    """
    # The tensor we are optimizing, our "block of clay". It needs requires_grad=True.
    candidate = initial_guess.clone().requires_grad_(True)
    # The LBFGS optimizer is very effective for this kind of problem.
    optimizer = torch.optim.LBFGS([candidate], max_iter=20, history_size=10, line_search_fn="strong_wolfe")

    # Flatten the projection directions for efficient matrix multiplication.
    projection_dirs_flat = projection_dirs.view(projection_dirs.shape[0], -1)

    # The closure is a function that the optimizer calls repeatedly.
    def closure():
        optimizer.zero_grad()
        # Project our current guess to see what its shadows look like.
        current_projections = candidate.flatten() @ projection_dirs_flat.T

        # We compare the distribution of our current shadows to the target shadows.
        # Sorting is the key to comparing distributions in 1D.
        current_sorted, _ = torch.sort(current_projections)
        target_sorted, _ = torch.sort(target_projections)

        # The loss is the Mean Squared Error between the perfect shadows and our current ones.
        loss = torch.mean((current_sorted - target_sorted) ** 2)
        loss.backward()
        return loss

    # This is the magic line. It runs the optimization loop.
    optimizer.step(closure)

    # Return the final, sculpted statue, detached from the computation graph.
    return candidate.detach()

# --- Helper 4: Sliced-Wasserstein Barycenter (Using the Real Engine) ---
def _hswb_swd_barycenter(
        tensors: list[Tensor],
        reference_tensor: Tensor,
        num_projections: int = 128,  # Reduced from 128 to prevent memory issues
        max_iter: int = 20,  # Iterations for the barycenter refinement
) -> Tensor:
    """Computes the proper SWB using iterative reconstruction."""
    debug_memory_usage("SWB_START")
    num_tensors = len(tensors)
    if num_tensors == 0:
        return torch.zeros_like(reference_tensor)
    if num_tensors == 1:
        return tensors[0]

    device = tensors[0].device
    shape = tensors[0].shape

    # Start with a simple average as our initial guess.
    debug_memory_usage("BEFORE_INITIAL_STACK")
    barycenter_guess = torch.mean(torch.stack(tensors), dim=0)  # SUSPECT #3
    debug_memory_usage("AFTER_INITIAL_STACK")

    # Clear memory after initial stack operation
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for iteration in range(max_iter):
        debug_memory_usage(f"ITER_{iteration}_START")

        # Process projections in smaller batches to reduce memory pressure
        batch_size = 16  # Process projections in smaller batches
        batch_results = []  # Store reconstruction results per batch (FIXED: was all_projected)

        for batch_start in range(0, num_projections, batch_size):
            batch_end = min(batch_start + batch_size, num_projections)
            batch_size_actual = batch_end - batch_start

            # Create projection directions for this batch only
            projection_dirs = torch.randn(batch_size_actual, barycenter_guess.numel(), device=device)
            projection_dirs /= torch.linalg.norm(projection_dirs, dim=1, keepdim=True)

            # Project all input style vectors onto the random directions.
            debug_memory_usage(f"ITER_{iteration}_BATCH_{batch_start // batch_size}_BEFORE_PROJECTION")

            # Project tensors in batch
            batch_projected = []
            for tensor in tensors:
                proj = tensor.flatten() @ projection_dirs.T
                batch_projected.append(proj)

            batch_tensor = torch.stack(batch_projected)

            # FIXED: Process this batch immediately instead of accumulating
            sorted_batch, _ = torch.sort(batch_tensor, dim=0)
            target_batch_1d = torch.mean(sorted_batch, dim=0)

            debug_memory_usage(f"ITER_{iteration}_BATCH_{batch_start // batch_size}_AFTER_PROJECTION")

            # FIXED: Apply reconstruction TO THIS BATCH ONLY
            debug_memory_usage(f"ITER_{iteration}_BATCH_{batch_start // batch_size}_BEFORE_RECONSTRUCTION")
            batch_reconstruction = _hswb_reconstruct_from_projections(
                target_batch_1d,  # Small batch target
                projection_dirs,  # Small batch projection directions
                barycenter_guess,  # This is the only large tensor
            )
            debug_memory_usage(f"ITER_{iteration}_BATCH_{batch_start // batch_size}_AFTER_RECONSTRUCTION")

            batch_results.append(batch_reconstruction)

            # Explicit cleanup after each batch
            del projection_dirs, batch_projected, batch_tensor, sorted_batch, target_batch_1d
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # FIXED: Combine batch reconstruction results (not the raw projections!)
        debug_memory_usage(f"ITER_{iteration}_BEFORE_COMBINE_BATCH_RESULTS")
        if batch_results:
            # Average the reconstruction results from all batches
            new_barycenter = torch.mean(torch.stack(batch_results), dim=0)
        else:
            new_barycenter = barycenter_guess
        debug_memory_usage(f"ITER_{iteration}_AFTER_COMBINE_BATCH_RESULTS")

        # Cleanup batch results
        del batch_results
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Check convergence
        debug_memory_usage(f"ITER_{iteration}_BEFORE_CONVERGENCE_CHECK")
        if torch.norm(new_barycenter - barycenter_guess) < 1e-5:
            barycenter_guess = new_barycenter
            break

        barycenter_guess = new_barycenter

        # Cleanup after each iteration
        debug_memory_usage(f"ITER_{iteration}_END_CLEANUP")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    debug_memory_usage("SWB_END")
    return barycenter_guess.reshape(shape)

def debug_memory_usage(label):
    pass

    # if torch.cuda.is_available():
    #     allocated = torch.cuda.memory_allocated() / 1024**3
    #     reserved = torch.cuda.memory_reserved() / 1024**3
    #     print(f"{label}: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    # else:
    #     import psutil
    #     memory = psutil.virtual_memory().used / 1024**3
    #     print(f"{label}: System RAM: {memory:.2f}GB")

def track_tensor_quality(tensor, label, key=None):
    """Track tensor quality metrics"""
    norm = torch.norm(tensor)
    mean_val = torch.mean(tensor)
    std_val = torch.std(tensor)
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()

    key_info = f"[{key}] " if key else ""
    logger.debug(
        "%s%s: norm=%.6f, mean=%.6f, std=%.6f, min=%.6f, max=%.6f, nan=%s, inf=%s",
        key_info,
        label,
        norm,
        mean_val,
        std_val,
        min_val,
        max_val,
        has_nan,
        has_inf,
    )
