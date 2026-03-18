from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .config import PipelineConfig
from .types import DatasetBundle


def build_spatial_knn(bundle: DatasetBundle, cfg: PipelineConfig) -> np.ndarray:
    x_col = cfg.spatial.x_column if cfg.spatial.x_column in bundle.obs.columns else "x"
    y_col = cfg.spatial.y_column if cfg.spatial.y_column in bundle.obs.columns else "y"
    coords = bundle.obs[[x_col, y_col]].to_numpy(dtype=float)
    k = min(cfg.spatial.k_neighbors + 1, len(coords))
    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(coords)
    indices = nbrs.kneighbors(coords, return_distance=False)
    return indices[:, 1:]


def spatial_context(expr: np.ndarray, knn_idx: np.ndarray) -> np.ndarray:
    ctx = np.zeros_like(expr)
    for i in range(expr.shape[0]):
        ctx[i] = expr[knn_idx[i]].mean(axis=0)
    return ctx
