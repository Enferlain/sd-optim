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
def magnitude_corrected_orthogonal(
        c: Parameter(Tensor),
        *deltas: Parameter(Tensor, merge_space="delta"),
        alpha: Parameter(float) = 1.0,
        conflict_aware: Parameter(bool) = False,
) -> Return(Tensor):
    """
    Orthogonal Model Merging (OrthoMerge)
    Implements Orthogonal-Residual Decoupling strategy for N deltas.
    Optionally supports Strategy 2: Conflict-Aware Decoupling.
    Reference: https://arxiv.org/abs/2602.05943
    Author: Clybius
    """
    if len(deltas) == 0:
        return c

    # Quick exit for all-zero deltas
    if all(torch.allclose(d, torch.zeros_like(d)) for d in deltas):
        return c

    if c.ndim < 2:
        merged_delta = sum(deltas) / len(deltas)
        return c + alpha * merged_delta

    orig_shape = c.shape
    W0 = c.flatten(1).float()

    out_dim, in_dim = W0.shape
    transpose_mode = False

    # Optimize SVD: apply Procrustes on the smaller dimension
    if out_dim > in_dim:
        W0 = W0.T
        transpose_mode = True

    # Compute average task vector (tau_mean) if conflict-aware
    d_2d_list = []
    for d in deltas:
        d_2d = d.flatten(1).float()
        if transpose_mode:
            d_2d = d_2d.T
        d_2d_list.append(d_2d)

    tau_mean = sum(d_2d_list) / len(d_2d_list)

    def extract_orthogonal_and_residual(d_2d_i, W_base):
        W_i = W_base + d_2d_i

        # Determine target matrix for orthogonal extraction
        if conflict_aware:
            # Strategy 2: Conflict-Aware Decoupling
            # Identify conflicting neurons (columns) where the local update opposes the global trend
            # Check signs of dot products between tau_i and tau_mean per neuron (column)
            # A negative dot product indicates a conflict.
            dot_products = torch.sum(d_2d_i * tau_mean, dim=0)  # shape: (dim2,)
            conflicts = dot_products < 0  # boolean mask of length dim2

            # tau_i^conf zeroes out non-conflicting columns
            tau_i_conf = torch.zeros_like(d_2d_i)
            # Apply the mask. Since conflicts is shape (dim2,), and tau_i is (dim1, dim2),
            # we can broadcast it to selectively copy columns
            tau_i_conf[:, conflicts] = d_2d_i[:, conflicts]

            W_target_i = W_base + tau_i_conf
        else:
            # Strategy 1: Global Decoupling
            W_target_i = W_i

        target_prod = W_target_i @ W_base.T
        # Guard against NaN/Inf which cause SVD convergence failure
        if not torch.isfinite(target_prod).all():
            target_prod = torch.nan_to_num(target_prod, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            U, S, Vh = torch.linalg.svd(target_prod, full_matrices=False)
        except RuntimeError:
            U, S, Vh = torch.linalg.svd(target_prod.cpu(), full_matrices=False)
            U, Vh = U.to(target_prod.device), Vh.to(target_prod.device)

        R_i = U @ Vh  # Orthogonal matrix

        # Residual component is ALWAYS calculated using the true W_i, not the conflict target
        rho_i = W_i - R_i @ W_base
        return R_i, rho_i

    def inverse_cayley(R):
        # Q = (R - I)(R + I)^-1
        r_p = R.clone()
        r_p.diagonal()[:] += 1
        r_n = R.clone()
        r_n.diagonal()[:] -= 1
        try:
            Q = torch.linalg.solve(r_p, r_n, left=False)
        except RuntimeError:
            r_p.diagonal()[:] += 1e-5
            Q = torch.linalg.solve(r_p, r_n, left=False)
        return Q

    Q_list = []
    rho_list = []

    for d_2d_i in d_2d_list:
        R_i, rho_i = extract_orthogonal_and_residual(d_2d_i, W0)
        Q_list.append(inverse_cayley(R_i))
        rho_list.append(rho_i)

    # Magnitude-Corrected Merging
    num_deltas = len(deltas)
    Q_mean = sum(Q_list) / num_deltas

    norm_sum = sum(torch.linalg.norm(Q) for Q in Q_list) / num_deltas
    mean_norm = torch.linalg.norm(Q_mean) + 1e-8

    c_factor = norm_sum / mean_norm
    # Prevent extreme scaling if heavily misaligned
    c_factor = torch.clamp(c_factor, max=10.0)

    Q_merged = c_factor * Q_mean

    q_p = Q_merged.clone()
    q_p.diagonal()[:] += 1
    q_n = -Q_merged
    q_n.diagonal()[:] += 1
    try:
        R_merged = torch.linalg.solve(q_n, q_p, left=False)
    except RuntimeError:
        q_n.diagonal()[:] += 1e-5
        R_merged = torch.linalg.solve(q_n, q_p, left=False)

    # Residual Component Merging
    rho_merged = sum(rho_list) / num_deltas

    # Hybrid Merging computations
    merged_delta_2d = (R_merged @ W0 + rho_merged) - W0

    if transpose_mode:
        merged_delta_2d = merged_delta_2d.T

    merged_delta = merged_delta_2d.view(orig_shape).to(c.dtype)
    return c + alpha * merged_delta