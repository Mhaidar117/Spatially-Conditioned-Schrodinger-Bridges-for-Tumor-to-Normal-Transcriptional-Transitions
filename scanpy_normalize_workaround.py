"""
Workaround for scanpy 1.11.4 bug: UnboundLocalError for counts_per_cols when
exclude_highly_expressed=False (default) with sparse CSR data.

Use this instead of sc.pp.normalize_total(adata, target_sum=1e4) until scanpy is updated.

Usage:
    from scanpy_normalize_workaround import normalize_total_workaround
    normalize_total_workaround(adata, target_sum=1e4)
"""

import numpy as np
import scipy.sparse as sp


def normalize_total_workaround(adata, target_sum=1e4):
    """Manual normalization equivalent to sc.pp.normalize_total(adata, target_sum=target_sum)."""
    X = adata.X
    counts_per_cell = np.array(X.sum(axis=1)).flatten()
    scale_factors = target_sum / np.maximum(counts_per_cell, 1.0)
    if sp.issparse(X):
        from scipy.sparse import diags
        adata.X = diags(scale_factors.astype(np.float32)) @ X.astype(np.float32)
    else:
        adata.X = (X.astype(np.float32) * scale_factors[:, np.newaxis])
    return adata
