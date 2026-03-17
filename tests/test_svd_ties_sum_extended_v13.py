from __future__ import annotations

import importlib

import pytest
import torch

runtime_merge_methods = importlib.import_module("sd_optim.extensions.bundled.merge_methods.experimental.svd_ties_sum_extended")


def test_approximate_svd_v2_allows_zero_power_iterations() -> None:
    matrices = torch.randn(2, 4, 3, dtype=torch.float32)

    out = runtime_merge_methods._approximate_svd_v2(
        matrices,
        max_rank=3,
        power_iterations=0,
        energy_threshold=0.9,
    )

    assert out.shape == matrices.shape
    assert torch.isfinite(out).all()


def test_compute_model_stock_chunked_v2_returns_unit_ratio_for_identical_positive_vectors() -> None:
    # All pairwise cosine similarities are > 0, so stock factor should be exactly 1.0.
    filtered_delta = torch.ones(3, 2, 2, dtype=torch.float32)

    ratio = runtime_merge_methods._compute_model_stock_chunked_v2(filtered_delta, cos_eps=1e-6, chunk_size=1)

    assert ratio == 1.0


def test_compute_final_chunk_v2_uses_passthrough_when_no_sign_agreement() -> None:
    filtered_delta = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float32)
    signs = torch.sign(filtered_delta)
    passthrough = torch.tensor([0.25, -0.75], dtype=torch.float32)

    out = runtime_merge_methods._compute_final_chunk_v2(
        filtered_delta=filtered_delta,
        signs=signs,
        vote_sgn=1.0,
        min_agreement=0.0,
        weight_decay=0.0,
        apply_stock=0.0,
        cos_eps=1e-6,
        apply_median=0.0,
        eps=1e-6,
        maxiter=10,
        ftol=1e-10,
        processing_batch_size=1,
        passthrough_slice=passthrough,
    )

    assert torch.allclose(out, passthrough)


def test_svd_ties_sum_extended_v13_handles_non_2d_tensors() -> None:
    model_a = torch.randn(4, 3, 2, dtype=torch.float32)
    model_b = torch.randn(4, 3, 2, dtype=torch.float32)

    out = runtime_merge_methods.svd_ties_sum_extended_v13.__wrapped__(
        model_a,
        model_b,
        k=1.0,
        max_singular_values=2,
        energy_threshold=0.9,
        power_iterations=1,
        vote_sgn=1.0,
        apply_stock=0.0,
        apply_median=0.0,
        min_agreement=0.0,
    )

    assert out.shape == model_a.shape
    assert torch.isfinite(out).all()


def test_approximate_svd_v2_handles_rank_deficient_inputs_without_lstsq_crash() -> None:
    # Rank-deficient by construction: duplicated columns.
    base = torch.tensor([[1.0, 2.0], [1.0, 2.0], [3.0, 6.0]], dtype=torch.float32)
    matrices = base.unsqueeze(0).repeat(2, 1, 1)

    out = runtime_merge_methods._approximate_svd_v2(
        matrices,
        max_rank=2,
        power_iterations=1,
        energy_threshold=0.9,
    )

    assert out.shape == matrices.shape
    assert torch.isfinite(out).all()


def test_runtime_merge_methods_copy_handles_rank_deficient_inputs() -> None:
    # Ensures the implementation used in actual recipe runs has the same safeguard.
    base = torch.tensor([[1.0, 2.0], [1.0, 2.0], [3.0, 6.0]], dtype=torch.float32)
    matrices = base.unsqueeze(0).repeat(2, 1, 1)

    out = runtime_merge_methods._approximate_svd_v2(
        matrices,
        max_rank=2,
        power_iterations=1,
        energy_threshold=0.9,
    )

    assert out.shape == matrices.shape
    assert torch.isfinite(out).all()


def test_legacy_merge_method_shims_are_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("sd_optim.merge_methods")

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("sd_optim.svd_ties_sum_extended")
