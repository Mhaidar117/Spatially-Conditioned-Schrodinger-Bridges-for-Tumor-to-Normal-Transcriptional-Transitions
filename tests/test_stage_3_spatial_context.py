"""
Stage 3: Section-restricted spatial kNN, neighbor-mean context, diagnostics, UMAP validation, logs.

Run:  python tests/test_stage_3_spatial_context.py
Or:   pytest tests/test_stage_3_spatial_context.py -v
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omega_spatial.config import PipelineConfig  # noqa: E402
from omega_spatial.types import DatasetBundle  # noqa: E402
from omega_spatial.spatial import (  # noqa: E402
    build_spatial_neighborhoods,
    spatial_context,
    verify_section_restriction,
)

FIG_DIR = REPO_ROOT / "results" / "stage_3_figures"
ARTIFACT_DIR = REPO_ROOT / "results" / "stage_3_artifacts"
LOG_PATH = REPO_ROOT / "logs" / "stage_3_spatial_context.log"
ARTIFACT_JSON_PATH = REPO_ROOT / "logs" / "stage_3_artifacts.json"
TEST_SCRIPT_PATH = REPO_ROOT / "tests" / "test_stage_3_spatial_context.py"
GBM_DEFAULT = REPO_ROOT / "Data" / "Inputs" / "general" / "GBM_data"
UMAP_RANDOM_STATE = 42


def _setup_file_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stage_3_spatial_context")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)
    return logger


def _prepare_expression_matrix(expr: np.ndarray, max_genes: int = 1500) -> np.ndarray:
    x = np.asarray(expr, dtype=float)
    g = min(max_genes, x.shape[1])
    x = x[:, :g]
    lib = x.sum(axis=1, keepdims=True) + 1e-8
    x = np.log1p(x / lib * 1e4)
    return x


def _embedding_for_umap(X: np.ndarray, random_state: int = UMAP_RANDOM_STATE) -> tuple[np.ndarray, str]:
    try:
        import umap  # type: ignore

        nn = max(2, min(15, X.shape[0] - 1))
        emb = umap.UMAP(
            n_neighbors=nn,
            min_dist=0.1,
            random_state=random_state,
            metric="euclidean",
        ).fit_transform(X)
        return emb, "UMAP (euclidean; reproducible seed=%s)" % random_state
    except Exception:
        from sklearn.decomposition import PCA

        emb = PCA(n_components=2, random_state=random_state).fit_transform(X)
        return emb, "PCA fallback (2D) — install umap-learn for UMAP"


def _plot_umap_categorical(
    emb: np.ndarray,
    labels: pd.Series,
    title: str,
    y_note: str,
    out_path: Path,
    legend_title: str,
) -> None:
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    labs = labels.astype(str).fillna("NA")
    uniq = sorted(labs.unique())
    cmap = plt.colormaps["tab20"]
    colors = cmap(np.linspace(0, 1, max(len(uniq), 1), endpoint=False))
    for i, u in enumerate(uniq):
        m = labs == u
        ax.scatter(emb[m, 0], emb[m, 1], s=10, alpha=0.8, color=colors[i % len(colors)], label=u)
    ax.set_xlabel("UMAP 1" if "UMAP" in y_note else "Component 1")
    ax.set_ylabel("UMAP 2" if "UMAP" in y_note else "Component 2")
    ax.set_title(f"{title}\n{y_note}\nAxes: low-dimensional embedding of spot features (validation).")
    ncol = 2 if len(uniq) <= 12 else 3
    ax.legend(title=legend_title, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, ncol=ncol)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_umap_continuous(
    emb: np.ndarray,
    values: np.ndarray,
    title: str,
    y_note: str,
    out_path: Path,
    cbar_label: str,
) -> None:
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=values, s=10, alpha=0.85, cmap="viridis")
    ax.set_xlabel("UMAP 1" if "UMAP" in y_note else "Component 1")
    ax.set_ylabel("UMAP 2" if "UMAP" in y_note else "Component 2")
    ax.set_title(f"{title}\n{y_note}\nColormap: perceptually uniform (viridis).")
    fig.colorbar(sc, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def synthetic_multi_section_bundle(rng: np.random.Generator | None = None) -> DatasetBundle:
    """Two dense sections far apart in space (global kNN would leak), duplicate coords, one singleton section."""
    rng = rng or np.random.default_rng(0)
    rows_a: list[dict] = []
    for i in range(6):
        for j in range(6):
            rows_a.append({"section_id": "sec_A", "x": float(i), "y": float(j)})
    rows_a.append({"section_id": "sec_A", "x": 2.0, "y": 2.0})
    rows_a.append({"section_id": "sec_A", "x": 2.0, "y": 2.0})

    rows_b: list[dict] = []
    for i in range(5):
        for j in range(4):
            rows_b.append({"section_id": "sec_B", "x": 100.0 + float(i), "y": 100.0 + float(j)})

    rows_c = [{"section_id": "sec_C", "x": 50.0, "y": 50.0}]

    obs = pd.DataFrame(rows_a + rows_b + rows_c)
    n, n_genes = len(obs), 80
    expr = rng.poisson(5, size=(n, n_genes)).astype(float)
    for s in obs["section_id"].unique():
        m = obs["section_id"] == s
        expr[m] *= 0.3 + 0.4 * (hash(s) % 7) / 7.0
    return DatasetBundle(
        expr=expr,
        obs=obs,
        var_names=[f"g{i}" for i in range(n_genes)],
        source_path=Path(__file__),
        dataset_kind="synthetic_stage3",
    )


def per_section_degree_summary(knn_idx: np.ndarray, section_ids: pd.Series) -> dict[str, dict[str, float]]:
    sec = section_ids.astype(str).reset_index(drop=True)
    out: dict[str, dict[str, float]] = {}
    for s in sec.unique():
        m = sec == s
        idx = np.flatnonzero(m.to_numpy())
        degs = []
        for i in idx:
            neigh = set(knn_idx[i].tolist())
            degs.append(len(neigh))
        out[s] = {
            "n_spots": float(m.sum()),
            "mean_unique_neighbor_count": float(np.mean(degs)) if degs else 0.0,
            "min_unique_neighbor_count": float(np.min(degs)) if degs else 0.0,
            "max_unique_neighbor_count": float(np.max(degs)) if degs else 0.0,
        }
    return out


def assert_no_cross_section_neighbors(knn_idx: np.ndarray, section_ids: pd.Series) -> None:
    assert verify_section_restriction(knn_idx, section_ids), "Cross-section neighbor leakage"


def test_synthetic_section_restriction_and_shapes() -> None:
    bundle = synthetic_multi_section_bundle()
    cfg = PipelineConfig()
    cfg.spatial.k_neighbors = 8
    cfg.state.section_column = "section_id"

    result = build_spatial_neighborhoods(bundle, cfg)
    knn = result.knn_indices
    n, g = bundle.expr.shape
    k = cfg.spatial.k_neighbors

    assert knn.shape == (n, k)
    assert result.neighbor_distances.shape == (n, k)
    assert result.context_matrix.shape == (n, g)
    assert_no_cross_section_neighbors(knn, bundle.obs["section_id"])

    ctx2 = spatial_context(bundle.expr, knn)
    assert np.allclose(ctx2, result.context_matrix)

    assert result.diagnostics.n_duplicate_coord_spots >= 2


def write_stage3_umaps(
    bundle: DatasetBundle,
    result,
    logger: logging.Logger,
    *,
    variant: str | None = None,
) -> list[str]:
    """If variant is set, filenames include it so optional Visium runs do not clobber synthetic figures."""
    obs = bundle.obs
    section_col = "section_id"
    X_obs = _prepare_expression_matrix(bundle.expr)
    enriched = np.hstack([X_obs, _prepare_expression_matrix(result.context_matrix)])
    emb_pre, note_pre = _embedding_for_umap(X_obs)
    emb_ctx, note_ctx = _embedding_for_umap(enriched)

    def _name(base: str) -> str:
        return f"stage_3_umap_{variant}_{base}.png" if variant else f"stage_3_umap_{base}.png"

    paths: list[str] = []
    p1 = FIG_DIR / _name("pre_context_section_id")
    _plot_umap_categorical(
        emb_pre,
        obs[section_col],
        "Stage 3 — pre-context expression (observed, normalized)",
        note_pre,
        p1,
        "section_id",
    )
    paths.append(str(p1.resolve()))
    logger.info("UMAP artifact: %s", p1)

    p2 = FIG_DIR / _name("context_enriched_section_id")
    _plot_umap_categorical(
        emb_ctx,
        obs[section_col],
        "Stage 3 — context-enriched representation ([expr || neighbor-mean expr], normalized)",
        note_ctx,
        p2,
        "section_id",
    )
    paths.append(str(p2.resolve()))
    logger.info("UMAP artifact: %s", p2)

    p3 = FIG_DIR / _name("neighborhood_diagnostics_mean_neighbor_distance")
    _plot_umap_continuous(
        emb_pre,
        result.mean_neighbor_distance,
        "Stage 3 — neighborhood diagnostics on pre-context UMAP embedding",
        note_pre,
        p3,
        "mean neighbor distance",
    )
    paths.append(str(p3.resolve()))
    logger.info("UMAP artifact: %s", p3)

    return paths


def run_synthetic_validation(logger: logging.Logger) -> dict:
    bundle = synthetic_multi_section_bundle()
    cfg = PipelineConfig()
    cfg.spatial.k_neighbors = 8
    cfg.state.section_column = "section_id"

    logger.info("Coordinate columns: x, y; section column: %s", cfg.state.section_column)
    logger.info("k_neighbors=%s", cfg.spatial.k_neighbors)

    result = build_spatial_neighborhoods(bundle, cfg, log=logger)
    knn = result.knn_indices
    n = bundle.expr.shape[0]
    k = cfg.spatial.k_neighbors

    print(f"n_spots={n}, k={k}")
    print("per_section_spot_counts:", result.diagnostics.per_section_spot_counts)
    print("per_section_effective_k (max distinct neighbors before pad):", result.diagnostics.per_section_effective_k)
    deg = per_section_degree_summary(knn, bundle.obs["section_id"])
    print("per-section unique-neighbor degree summary:", json.dumps(deg, indent=2))
    print("isolated_spot_indices:", result.diagnostics.isolated_spot_indices)
    print("duplicate_coordinate_spot_indices (count=%s):" % len(result.diagnostics.duplicate_coordinate_spot_indices))
    print(result.diagnostics.duplicate_coordinate_spot_indices[:20], "...")

    assert_no_cross_section_neighbors(knn, bundle.obs["section_id"])

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    knn_path = ARTIFACT_DIR / "knn_indices.npy"
    dist_path = ARTIFACT_DIR / "neighbor_distances.npy"
    ctx_path = ARTIFACT_DIR / "context_matrix.npy"
    np.save(knn_path, knn)
    np.save(dist_path, result.neighbor_distances)
    np.save(ctx_path, result.context_matrix)
    logger.info("Wrote graph/context arrays to %s", ARTIFACT_DIR)

    umap_paths = write_stage3_umaps(bundle, result, logger)
    for p in umap_paths:
        logger.info("Logged UMAP path: %s", p)

    dist_summary = {
        "mean": float(result.mean_neighbor_distance.mean()),
        "std": float(result.mean_neighbor_distance.std()),
        "p50": float(np.median(result.mean_neighbor_distance)),
        "p90": float(np.quantile(result.mean_neighbor_distance, 0.9)),
    }
    logger.info("Neighbor distance summaries (mean over k per spot): %s", json.dumps(dist_summary))

    stage4_handoff = {
        "graph_format": (
            "knn_indices: int64 array, shape (n_spots, k). "
            "knn_indices[i, j] is a global row index into expr/obs; all neighbors share section_id with spot i."
        ),
        "neighbor_distances": "float64 array, shape (n_spots, k); Euclidean (x,y) distances to each neighbor.",
        "context_matrix": "float64, shape (n_spots, n_genes); row-wise mean of expr over knn_indices[i].",
        "ordering": "Row i aligns with bundle.obs.iloc[i] and bundle.expr[i] (same as Stage 1 canonical order).",
        "failure_modes_to_check_before_training": [
            "NaN/inf in expr or coordinates (coordinates imputed with per-section median; see warnings in log).",
            "Sections with fewer than k+1 spots: neighbors padded by repeating the farthest available neighbor.",
            "Single-spot sections: all neighbors are self; mean context equals self expression.",
            "Duplicate (x,y) within a section: kNN ties arbitrary among coincident spots.",
            "If section_neighbor_purity < 1, graph construction is invalid — do not train.",
        ],
    }

    manifest: dict = {
        "graph_artifact_paths": [str(knn_path.resolve()), str(dist_path.resolve())],
        "context_artifact_paths": [str(ctx_path.resolve())],
        "umap_figure_paths": umap_paths,
        "hyperparameters": {
            "k_neighbors": k,
            "x_column": result.diagnostics.x_column,
            "y_column": result.diagnostics.y_column,
            "section_column": result.diagnostics.section_column,
            "umap_random_state": UMAP_RANDOM_STATE,
        },
        "test_script_path": str(TEST_SCRIPT_PATH.resolve()),
        "known_limitations": [
            "UMAPs use separate fits for pre-context vs context-enriched features (same hyperparameters and seed).",
            "Synthetic fixture uses Poisson expression; real Visium runs optional below.",
        ],
        "per_section_spot_counts": result.diagnostics.per_section_spot_counts,
        "neighbor_distance_summary": dist_summary,
        "isolated_spot_indices": result.diagnostics.isolated_spot_indices,
        "n_duplicate_coordinate_spots": result.diagnostics.n_duplicate_coord_spots,
        "stage_4_handoff": stage4_handoff,
    }
    return manifest


def run_optional_visium_umaps(logger: logging.Logger, manifest: dict) -> None:
    if not GBM_DEFAULT.is_dir():
        manifest["known_limitations"].append("GBM_data not found — skipped optional Visium Stage 3 UMAPs.")
        logger.warning(manifest["known_limitations"][-1])
        return
    try:
        from omega_spatial.io import load_dataset, resolve_dataset
    except Exception as e:
        manifest["known_limitations"].append(f"Optional Visium load failed: {e}")
        return

    os.environ.setdefault("OMEGA_STAGE1_MAX_SAMPLES", "3")
    p, kind = resolve_dataset(str(GBM_DEFAULT))
    bundle = load_dataset(p, kind)
    for col in ("section_id", "x", "y"):
        if col not in bundle.obs.columns:
            manifest["known_limitations"].append(f"Optional Visium bundle missing {col}; skip.")
            logger.warning(manifest["known_limitations"][-1])
            return

    cfg = PipelineConfig()
    cfg.spatial.k_neighbors = min(8, max(1, bundle.expr.shape[0] // 20))
    result = build_spatial_neighborhoods(bundle, cfg, log=logger)
    assert_no_cross_section_neighbors(result.knn_indices, bundle.obs["section_id"])
    extra = write_stage3_umaps(bundle, result, logger, variant="visium_subset")
    manifest.setdefault("optional_visium_umap_paths", []).extend(extra)
    manifest["optional_visium_note"] = "Visium UMAPs use distinct filenames (variant visium_subset)."


def main() -> None:
    logger = _setup_file_logging()
    logger.info("Stage 3 spatial context validation (repo root: %s)", REPO_ROOT)

    manifest = run_synthetic_validation(logger)

    run_optional_visium_umaps(logger, manifest)

    ARTIFACT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_JSON_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote artifact manifest: %s", ARTIFACT_JSON_PATH)

    produced = (
        [str(LOG_PATH), str(ARTIFACT_JSON_PATH)]
        + manifest.get("graph_artifact_paths", [])
        + manifest.get("context_artifact_paths", [])
        + manifest.get("umap_figure_paths", [])
        + manifest.get("optional_visium_umap_paths", [])
    )
    logger.info("Produced artifacts: %s", json.dumps(produced, indent=2))

    print("Stage 3 checks complete. Log:", LOG_PATH)
    print("Manifest:", ARTIFACT_JSON_PATH)


if __name__ == "__main__":
    main()
