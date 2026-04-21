from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .config import PipelineConfig
from .types import DatasetBundle, SpatialNeighborhoodDiagnostics, SpatialNeighborhoodResult

logger = logging.getLogger(__name__)

# --- Stage 4 handoff (see also logs/stage_3_artifacts.json) ---
# Graph: knn_indices[i, :] are global row indices of k neighbors of spot i in bundle.expr / obs.
#        All neighbors lie in the same section as spot i. Integer dtype int64.
# Context: context_matrix has shape (n_spots, n_genes), row i is the mean of expr over knn_indices[i].
# Ordering: Row i matches bundle.obs.iloc[i] and bundle.expr[i]. Do not reorder without reordering all three.
# Failure modes: missing/NaN coordinates (filled with per-section median, logged); single-spot sections
#   (neighbors padded with self, distance 0); k_neighbors > n_section-1 (last neighbor repeated).


def _resolve_xy_columns(bundle: DatasetBundle, cfg: PipelineConfig) -> tuple[str, str]:
    x_col = cfg.spatial.x_column if cfg.spatial.x_column in bundle.obs.columns else "x"
    y_col = cfg.spatial.y_column if cfg.spatial.y_column in bundle.obs.columns else "y"
    return x_col, y_col


def _resolve_section_column(bundle: DatasetBundle, cfg: PipelineConfig) -> str:
    col = cfg.state.section_column if cfg.state.section_column in bundle.obs.columns else "section_id"
    if col not in bundle.obs.columns:
        raise ValueError(
            f"Section column '{cfg.state.section_column}' not in obs; required for section-restricted kNN."
        )
    return col


def _sanitize_coords(
    coords: np.ndarray,
    sections: pd.Series,
    warnings_out: list[str],
) -> np.ndarray:
    """Replace NaN coordinates with per-section median; log fallbacks."""
    out = coords.copy()
    sec_str = sections.astype(str)
    bad = ~np.isfinite(out).any(axis=1)
    if bad.any():
        warnings_out.append(f"Non-finite coordinates on {int(bad.sum())} spots; imputing per-section median.")
    for s in sec_str.unique():
        m = sec_str == s
        sub = out[m]
        for j in range(2):
            col = sub[:, j]
            med = np.nanmedian(col) if np.isfinite(np.nanmedian(col)) else 0.0
            fix = ~np.isfinite(col)
            if fix.any():
                sub[fix, j] = med
        out[m] = sub
    if not np.isfinite(out).all():
        warnings_out.append("Coordinates still non-finite after imputation; clipping to 0.")
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def build_spatial_neighborhoods(
    bundle: DatasetBundle,
    cfg: PipelineConfig,
    *,
    log: logging.Logger | None = None,
) -> SpatialNeighborhoodResult:
    """
    Section-restricted Euclidean kNN on (x, y) and neighbor-mean expression context.
    """
    lg = log or logger
    warnings: list[str] = []
    n, g = bundle.expr.shape
    x_col, y_col = _resolve_xy_columns(bundle, cfg)
    section_col = _resolve_section_column(bundle, cfg)
    sections = bundle.obs[section_col]
    sec_str = sections.astype(str)

    coords = bundle.obs[[x_col, y_col]].to_numpy(dtype=float)
    coords = _sanitize_coords(coords, sections, warnings)

    k_target = max(1, int(cfg.spatial.k_neighbors))
    knn_idx = np.zeros((n, k_target), dtype=np.int64)
    dist_mat = np.zeros((n, k_target), dtype=np.float64)

    per_section_counts: dict[str, int] = {}
    per_section_eff_k: dict[str, int] = {}
    isolated: list[int] = []

    for sec in sec_str.unique():
        mask = (sec_str == sec).to_numpy()
        idx_global = np.flatnonzero(mask)
        n_sec = int(idx_global.size)
        per_section_counts[sec] = n_sec
        if n_sec == 0:
            continue
        if n_sec == 1:
            i0 = int(idx_global[0])
            knn_idx[i0, :] = i0
            dist_mat[i0, :] = 0.0
            per_section_eff_k[sec] = 0
            isolated.append(i0)
            warnings.append(f"Section {sec!r} has a single spot; neighbors padded with self.")
            continue

        k_eff = min(k_target, n_sec - 1)
        per_section_eff_k[sec] = k_eff
        X = coords[idx_global]
        nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean")
        nn.fit(X)
        dists, ind_local = nn.kneighbors(X)
        neigh_local = ind_local[:, 1 : 1 + k_eff]
        neigh_dist = dists[:, 1 : 1 + k_eff]
        neigh_global = idx_global[neigh_local]

        if k_eff < k_target:
            last_n = neigh_global[:, -1:]
            last_d = neigh_dist[:, -1:]
            pad_n = np.repeat(last_n, k_target - k_eff, axis=1)
            pad_d = np.repeat(last_d, k_target - k_eff, axis=1)
            neigh_global = np.hstack([neigh_global, pad_n])
            neigh_dist = np.hstack([neigh_dist, pad_d])
            warnings.append(
                f"Section {sec!r}: requested k={k_target} but only {k_eff} distinct neighbors; padded."
            )

        for row, gi in enumerate(idx_global.tolist()):
            knn_idx[gi] = neigh_global[row]
            dist_mat[gi] = neigh_dist[row]

    dup_spots: list[int] = []
    for sec in sec_str.unique():
        m = sec_str == sec
        sub_idx = np.flatnonzero(m.to_numpy())
        sub_xy = coords[sub_idx]
        keys = [tuple(row) for row in sub_xy]
        bucket: dict[tuple[float, float], list[int]] = defaultdict(list)
        for local_i, key in enumerate(keys):
            bucket[key].append(int(sub_idx[local_i]))
        for members in bucket.values():
            if len(members) > 1:
                dup_spots.extend(members)
    dup_spots = sorted(set(dup_spots))
    if dup_spots:
        warnings.append(f"Found {len(dup_spots)} spots with duplicate (x,y) within the same section.")

    mean_dist = dist_mat.mean(axis=1)
    local_density = 1.0 / (mean_dist + 1e-8)
    purity = np.ones(n, dtype=np.float64)
    for i in range(n):
        si = sec_str.iloc[i]
        neigh = knn_idx[i]
        same = sec_str.iloc[neigh].astype(str) == si
        purity[i] = float(same.mean())

    ctx = spatial_context(bundle.expr, knn_idx)

    diagnostics = SpatialNeighborhoodDiagnostics(
        n_spots=n,
        n_genes=g,
        k_neighbors=k_target,
        x_column=x_col,
        y_column=y_col,
        section_column=section_col,
        per_section_spot_counts=per_section_counts,
        per_section_effective_k=per_section_eff_k,
        isolated_spot_indices=isolated,
        duplicate_coordinate_spot_indices=dup_spots,
        n_duplicate_coord_spots=len(dup_spots),
        warnings=warnings,
    )

    if purity.min() < 1.0 - 1e-9:
        msg = "Cross-section neighbor leakage detected in purity check."
        lg.error(msg)
        raise RuntimeError(msg)

    for w in warnings:
        lg.warning("%s", w)

    return SpatialNeighborhoodResult(
        knn_indices=knn_idx,
        neighbor_distances=dist_mat,
        context_matrix=ctx,
        mean_neighbor_distance=mean_dist,
        local_density=local_density,
        section_neighbor_purity=purity,
        diagnostics=diagnostics,
    )


def build_spatial_knn(bundle: DatasetBundle, cfg: PipelineConfig) -> np.ndarray:
    """Backward-compatible API: section-restricted neighbor indices, shape (n_spots, k)."""
    return build_spatial_neighborhoods(bundle, cfg).knn_indices


def spatial_context(expr: np.ndarray, knn_idx: np.ndarray) -> np.ndarray:
    """Neighbor-mean expression; knn_idx rows are global indices into expr."""
    expr = np.asarray(expr, dtype=float)
    knn_idx = np.asarray(knn_idx, dtype=np.int64)
    return expr[knn_idx].mean(axis=1)


def _prepare_expression_for_embedding(x: np.ndarray) -> np.ndarray:
    """Z-score columns with NaN/inf sanitization (delegates to utils.zscore_columns)."""
    from .utils import zscore_columns

    return zscore_columns(x)


def _embedding_for_umap(x: np.ndarray, random_state: int) -> tuple[np.ndarray, str]:
    from .utils import umap_or_pca_2d

    return umap_or_pca_2d(_prepare_expression_for_embedding(x), random_state, label="normalized features")


def _plot_umap_categorical(
    emb: np.ndarray,
    labels: pd.Series | np.ndarray,
    title: str,
    basis_note: str,
    out_path: Path,
    legend_title: str,
) -> None:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    labs = pd.Series(labels).astype(str)
    uniq = sorted(labs.unique().tolist())
    cmap = plt.colormaps["tab20"]
    denom = max(len(uniq) - 1, 1)
    for i, u in enumerate(uniq):
        m = labs == u
        ax.scatter(
            emb[m, 0],
            emb[m, 1],
            s=10,
            alpha=0.85,
            color=cmap(i / denom),
            label=u,
            edgecolors="none",
        )
    ax.set_xlabel("UMAP 1" if "UMAP" in basis_note else "Component 1")
    ax.set_ylabel("UMAP 2" if "UMAP" in basis_note else "Component 2")
    ax.set_title(f"{title}\n{basis_note}")
    ax.legend(title=legend_title, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_umap_continuous(
    emb: np.ndarray,
    values: np.ndarray,
    title: str,
    basis_note: str,
    out_path: Path,
    cbar_label: str,
) -> None:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=values, cmap="viridis", s=10, alpha=0.88, edgecolors="none")
    ax.set_xlabel("UMAP 1" if "UMAP" in basis_note else "Component 1")
    ax.set_ylabel("UMAP 2" if "UMAP" in basis_note else "Component 2")
    ax.set_title(f"{title}\n{basis_note}")
    fig.colorbar(sc, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_stage3_umap_figures(
    bundle: DatasetBundle,
    result: SpatialNeighborhoodResult,
    fig_dir: str | Path,
    logger: logging.Logger,
    *,
    random_state: int = 42,
) -> list[str]:
    """
    Stage 3 production figures: pre-context embedding, context-enriched embedding, and neighborhood diagnostics.
    """
    fig_dir = Path(fig_dir).expanduser().resolve()
    fig_dir.mkdir(parents=True, exist_ok=True)
    section_col = result.diagnostics.section_column
    x_obs = _prepare_expression_for_embedding(bundle.expr)
    x_ctx = np.hstack([x_obs, _prepare_expression_for_embedding(result.context_matrix)])
    emb_pre, note_pre = _embedding_for_umap(x_obs, random_state=random_state)
    emb_ctx, note_ctx = _embedding_for_umap(x_ctx, random_state=random_state + 1)

    paths: list[str] = []

    p1 = fig_dir / "stage_3_umap_pre_context_section_id.png"
    _plot_umap_categorical(
        emb_pre,
        bundle.obs[section_col],
        "Stage 3 — pre-context expression",
        note_pre,
        p1,
        section_col,
    )
    paths.append(str(p1.resolve()))
    logger.info("Stage 3 UMAP artifact: %s", p1)

    p2 = fig_dir / "stage_3_umap_context_enriched_section_id.png"
    _plot_umap_categorical(
        emb_ctx,
        bundle.obs[section_col],
        "Stage 3 — context-enriched representation ([expr || neighbor-mean expr])",
        note_ctx,
        p2,
        section_col,
    )
    paths.append(str(p2.resolve()))
    logger.info("Stage 3 UMAP artifact: %s", p2)

    p3 = fig_dir / "stage_3_umap_neighborhood_diagnostics_mean_neighbor_distance.png"
    _plot_umap_continuous(
        emb_pre,
        result.mean_neighbor_distance,
        "Stage 3 — neighborhood diagnostics on pre-context embedding",
        note_pre,
        p3,
        "mean_neighbor_distance",
    )
    paths.append(str(p3.resolve()))
    logger.info("Stage 3 UMAP artifact: %s", p3)

    p4 = fig_dir / "stage_3_umap_neighborhood_diagnostics_section_purity.png"
    _plot_umap_continuous(
        emb_pre,
        result.section_neighbor_purity,
        "Stage 3 — section-neighborhood purity diagnostics",
        note_pre,
        p4,
        "section_neighbor_purity",
    )
    paths.append(str(p4.resolve()))
    logger.info("Stage 3 UMAP artifact: %s", p4)

    if "org1" in bundle.obs.columns:
        org = bundle.obs["org1"].astype(str).str.lower()
        structured = org.str.contains("str")
        disorganized = org.str.contains("dis")
        if structured.any() and disorganized.any():
            import matplotlib.pyplot as plt

            p5 = fig_dir / "stage_3_structured_vs_disorganized_neighbor_distance.png"
            fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=150)
            vals = [
                result.mean_neighbor_distance[structured.to_numpy()].mean(),
                result.mean_neighbor_distance[disorganized.to_numpy()].mean(),
            ]
            ax.bar(["structured(org1)", "disorganized(org1)"], vals, color=["#4c78a8", "#f58518"])
            ax.set_ylabel("mean_neighbor_distance")
            ax.set_title("Stage 3 — neighborhood compactness by organization zone")
            fig.tight_layout()
            fig.savefig(p5, dpi=300, bbox_inches="tight")
            plt.close(fig)
            paths.append(str(p5.resolve()))
            logger.info("Stage 3 diagnostic figure: %s", p5)

    return paths


def write_stage3_artifact_manifest(
    manifest_path: str | Path,
    *,
    graph_paths: dict[str, str],
    umap_paths: list[str],
    diagnostics: SpatialNeighborhoodDiagnostics,
    known_limitations: list[str],
    test_script_path: str,
    extra: dict[str, object] | None = None,
) -> None:
    payload: dict[str, object] = {
        "graph_artifact_paths": graph_paths,
        "umap_figure_paths": list(umap_paths),
        "hyperparameters": {
            "k_neighbors": diagnostics.k_neighbors,
            "x_column": diagnostics.x_column,
            "y_column": diagnostics.y_column,
            "section_column": diagnostics.section_column,
        },
        "test_script_path": test_script_path,
        "known_limitations": list(known_limitations),
        "per_section_spot_counts": diagnostics.per_section_spot_counts,
        "per_section_effective_k": diagnostics.per_section_effective_k,
        "isolated_spot_indices": diagnostics.isolated_spot_indices,
        "n_duplicate_coordinate_spots": diagnostics.n_duplicate_coord_spots,
        "stage_4_handoff": {
            "graph_format": (
                "knn_indices[i, j] is a global row index into expr/obs; all neighbors share section_id with spot i."
            ),
            "neighbor_distances": "float64 array, shape (n_spots, k); Euclidean (x,y) distances to each neighbor.",
            "context_matrix": "float64 array, shape (n_spots, n_genes); row-wise mean over neighbors.",
            "ordering": "Row i aligns with bundle.obs.iloc[i] and bundle.expr[i] in pipeline order.",
        },
    }
    if extra:
        payload.update(extra)
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def verify_section_restriction(
    knn_indices: np.ndarray,
    section_ids: pd.Series | np.ndarray,
) -> bool:
    """Return True iff every neighbor shares the query spot's section."""
    sec = pd.Series(section_ids).astype(str).reset_index(drop=True)
    n = knn_indices.shape[0]
    for i in range(n):
        si = sec.iloc[i]
        neigh = knn_indices[i]
        if not (sec.iloc[neigh].astype(str) == si).all():
            return False
    return True
