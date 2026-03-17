"""Legacy compatibility surface for class-based merge helper access.

Bundled merge-method implementations now live under
``sd_optim.extensions.bundled.merge_methods``. This module keeps the historic
``MergeMethods`` class importable for callers and tests that still reach for
helper methods via ``sd_optim.merge_methods``.
"""

from sd_optim.extensions.bundled.merge_methods.experimental import (
    svd_ties_sum_extended as _svd_ties_sum_extended,
)


class MergeMethods:
    """Compatibility wrapper exposing selected bundled helpers as class attrs."""

    svd_ties_sum_extended_v13 = staticmethod(_svd_ties_sum_extended.svd_ties_sum_extended_v13)
    _get_optimized_chunks_v12 = staticmethod(_svd_ties_sum_extended._get_optimized_chunks_v12)
    filter_top_k_v2 = staticmethod(_svd_ties_sum_extended.filter_top_k_v2)
    _approximate_svd_v2 = staticmethod(_svd_ties_sum_extended._approximate_svd_v2)
    _compute_final_chunk_v2 = staticmethod(_svd_ties_sum_extended._compute_final_chunk_v2)
    _compute_geometric_median_chunked_v2 = staticmethod(_svd_ties_sum_extended._compute_geometric_median_chunked_v2)
    _compute_model_stock_chunked_v2 = staticmethod(_svd_ties_sum_extended._compute_model_stock_chunked_v2)


# Keep the bundled helper module's internal fallback counter targeting the same
# compatibility class that legacy callers import from this module.
_svd_ties_sum_extended.MergeMethods = MergeMethods

__all__ = ["MergeMethods"]
