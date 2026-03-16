"""Bundled merge-method package surface and shared SVD helpers.

Keep this package import light: SVD helpers come from ``sd_optim.svd`` so
importing the package does not eagerly import the bundled ``svd`` merge-method
module and its optional dependencies.
"""

from sd_optim.svd import (
    fractional_matrix_power,
    get_approximate_basis,
    get_floating_dtype,
    orthogonal_extend,
    orthogonal_procrustes,
    torch_complex_dtype_map,
    torch_svd_lowrank,
)

__all__ = [
    "fractional_matrix_power",
    "get_approximate_basis",
    "get_floating_dtype",
    "orthogonal_extend",
    "orthogonal_procrustes",
    "torch_complex_dtype_map",
    "torch_svd_lowrank",
]
