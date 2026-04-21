"""
Shared utilities for the omega_spatial pipeline.

These helpers were previously duplicated across several stage modules
(``validation``, ``reporting``, ``synthetic_validation``, ``spatial``,
``perturbations``, ``programs``).  Centralizing them here keeps behavior
consistent and removes maintenance drift.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def repo_relative(repo_root: Path | str, path: Path | str) -> str:
    """Return ``path`` as a POSIX path relative to ``repo_root`` when possible.

    Falls back to the absolute path string when ``path`` is outside the repo.
    Used by every artifact manifest writer in the pipeline so that downstream
    tools receive stable, repo-rooted paths.
    """
    root = Path(repo_root).resolve()
    p = Path(path).expanduser().resolve()
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return str(p)


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation that tolerates NaNs / infs and short inputs."""
    aa = np.asarray(a, dtype=float).ravel()
    bb = np.asarray(b, dtype=float).ravel()
    mask = np.isfinite(aa) & np.isfinite(bb)
    if int(mask.sum()) < 4:
        return float("nan")
    return float(np.corrcoef(aa[mask], bb[mask])[0, 1])


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """Return a unit vector, passing through vectors of near-zero norm."""
    arr = np.asarray(vec, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return arr
    return arr / norm


def safe_rowwise_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity that returns 0 for near-zero rows."""
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    numer = np.sum(aa * bb, axis=1)
    denom = np.linalg.norm(aa, axis=1) * np.linalg.norm(bb, axis=1)
    out = np.zeros(aa.shape[0], dtype=float)
    good = denom > 1e-12
    out[good] = numer[good] / denom[good]
    return out


def l2_to_target(x: np.ndarray, target_mu: np.ndarray) -> np.ndarray:
    """Row-wise L2 distance to a reference row (broadcast friendly)."""
    arr = np.asarray(x, dtype=float)
    ref = np.asarray(target_mu, dtype=float).reshape(1, -1)
    return np.linalg.norm(arr - ref, axis=1)


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def umap_or_pca_2d(
    x: np.ndarray,
    random_state: int,
    *,
    label: str = "features",
) -> tuple[np.ndarray, str]:
    """Return a 2-D embedding (UMAP if available, else PCA).

    ``label`` is embedded into the returned basis-note so downstream figure
    captions remain informative ("UMAP on perturbation vectors (seed=42)" vs
    "UMAP on normalized features (seed=7)").  The fallback branch keeps the
    pipeline running in environments where ``umap-learn`` is not installed.
    """
    arr = np.asarray(x, dtype=float)
    try:
        import umap  # type: ignore

        n_neighbors = max(2, min(15, arr.shape[0] - 1))
        emb = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=0.1,
            random_state=random_state,
            metric="euclidean",
        ).fit_transform(arr)
        return emb, f"UMAP on {label} (euclidean; seed={random_state})"
    except Exception:  # noqa: BLE001
        from sklearn.decomposition import PCA

        emb = PCA(n_components=2, random_state=random_state).fit_transform(arr)
        return emb, f"PCA 2D fallback on {label} — install umap-learn for UMAP"


def zscore_columns(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Column-wise z-score with NaN/inf sanitization and zero-variance guard."""
    arr = np.asarray(x, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = arr - np.mean(arr, axis=0, keepdims=True)
    scale = np.std(arr, axis=0, keepdims=True)
    scale[scale < eps] = 1.0
    return arr / scale
