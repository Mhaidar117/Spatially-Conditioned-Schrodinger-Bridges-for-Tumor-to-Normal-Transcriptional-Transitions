"""
Stage 1: GBM Visium ingestion, metadata join, schema checks, UMAP validation, logs and manifest.

Run:  python tests/test_stage_1_data_ingestion.py
Or:   pytest tests/test_stage_1_data_ingestion.py -v
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omega_spatial.io import _join_true_cna_scores, discover_visium_metadata_path, load_dataset, resolve_dataset  # noqa: E402
from omega_spatial.readiness import diagnose_stage1_ingestion, get_ingestion_diagnostics  # noqa: E402

FIG_DIR = REPO_ROOT / "results" / "stage_1_figures"
LOG_PATH = REPO_ROOT / "logs" / "stage_1_data_ingestion.log"
ARTIFACT_PATH = REPO_ROOT / "logs" / "stage_1_artifacts.json"
TEST_SCRIPT_PATH = REPO_ROOT / "tests" / "test_stage_1_data_ingestion.py"
GBM_DEFAULT = REPO_ROOT / "Data" / "Inputs" / "general" / "GBM_data"
META_DEFAULT = REPO_ROOT / "Data" / "Inputs" / "general" / "visium_metadata.csv"


def _setup_file_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stage_1_data_ingestion")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)
    return logger


def _assert_canonical_schema(bundle) -> None:
    obs = bundle.obs
    for col in ("section_id", "x", "y"):
        assert col in obs.columns, f"Missing canonical column {col}"
    assert not obs["section_id"].isna().all(), "section_id all NaN"
    assert obs["x"].notna().sum() > 0 and obs["y"].notna().sum() > 0, "x/y missing"
    n, g = bundle.expr.shape
    assert n == len(obs), f"expr rows {n} != obs rows {len(obs)}"
    assert g == len(bundle.var_names), "expr cols != len(var_names)"


def _embedding_for_umap(X: np.ndarray, random_state: int = 42) -> tuple[np.ndarray, str]:
    """UMAP if available; else PCA fallback (logged)."""
    try:
        import umap  # type: ignore

        nn = max(2, min(15, X.shape[0] - 1))
        emb = umap.UMAP(
            n_neighbors=nn,
            min_dist=0.1,
            random_state=random_state,
            metric="euclidean",
        ).fit_transform(X)
        return emb, "UMAP on log1p-normalized expression (observed counts)"
    except Exception:
        from sklearn.decomposition import PCA

        emb = PCA(n_components=2, random_state=random_state).fit_transform(X)
        return emb, "PCA fallback (2D) — umap-learn unavailable; install dev extras for UMAP"


def _prepare_expression_matrix(expr: np.ndarray, max_genes: int = 1500) -> np.ndarray:
    x = np.asarray(expr, dtype=float)
    g = min(max_genes, x.shape[1])
    x = x[:, :g]
    lib = x.sum(axis=1, keepdims=True) + 1e-8
    x = np.log1p(x / lib * 1e4)
    return x


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
        ax.scatter(emb[m, 0], emb[m, 1], s=6, alpha=0.75, color=colors[i % len(colors)], label=u)
    ax.set_xlabel("UMAP 1" if "UMAP" in y_note else "Component 1")
    ax.set_ylabel("UMAP 2" if "UMAP" in y_note else "Component 2")
    ax.set_title(f"{title}\n{y_note}")
    ncol = 2 if len(uniq) <= 12 else 3
    ax.legend(title=legend_title, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, ncol=ncol)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_stage1_umaps(bundle, logger: logging.Logger) -> list[str]:
    obs = bundle.obs
    X = _prepare_expression_matrix(bundle.expr)
    emb, note = _embedding_for_umap(X)
    paths: list[str] = []

    p1 = FIG_DIR / "stage_1_umap_section_id.png"
    _plot_umap_categorical(emb, obs["section_id"], "Stage 1 — section identity", note, p1, "section_id")
    paths.append(str(p1))
    logger.info("Wrote %s", p1)

    batch_col = "metadata_sample" if "metadata_sample" in obs.columns else "section_id"
    p2 = FIG_DIR / "stage_1_umap_sample_batch.png"
    _plot_umap_categorical(
        emb,
        obs[batch_col],
        "Stage 1 — sample / batch-like identifier",
        note + f"\nColor: {batch_col} (evaluation-style grouping; not a model input by default).",
        p2,
        batch_col,
    )
    paths.append(str(p2))
    logger.info("Wrote %s", p2)

    if "metadata_joined" in obs.columns:
        mj = obs["metadata_joined"].map({True: "joined", False: "not_joined"})
    else:
        mj = pd.Series(["unknown"] * len(obs))
    p3 = FIG_DIR / "stage_1_umap_metadata_join.png"
    _plot_umap_categorical(
        emb,
        mj,
        "Stage 1 — visium_metadata join status",
        note + "\nGreenwald cohort metadata join (spot_id ↔ barcode).",
        p3,
        "metadata_join",
    )
    paths.append(str(p3))
    logger.info("Wrote %s", p3)

    return paths


def test_resolve_minimal_gbm_layout(tmp_path: Path) -> None:
    from scipy.io import mmwrite  # type: ignore
    from scipy.sparse import coo_matrix  # type: ignore

    general = tmp_path / "general"
    gbm = general / "GBM_data"
    # 3 genes x 2 spots each
    for sample, barcodes in (("S1", ["AA-1", "BB-1"]), ("S2", ["CC-1", "DD-1"])):
        mtx_dir = gbm / sample / "outs" / "filtered_feature_bc_matrix"
        mtx_dir.mkdir(parents=True)
        (mtx_dir.parent / "spatial").mkdir(parents=True)
        n = len(barcodes)
        rows, cols, data = [], [], []
        for si in range(n):
            for gi in range(3):
                rows.append(gi)
                cols.append(si)
                data.append(float(si + gi + 1))
        mat = coo_matrix((data, (rows, cols)), shape=(3, n))
        mmwrite(mtx_dir / "matrix.mtx", mat)
        with open(mtx_dir / "features.tsv", "w", encoding="utf-8") as f:
            for g in ("G1", "G2", "G3"):
                f.write(f"{g}\t{g}\tGene Expression\n")
        with open(mtx_dir / "barcodes.tsv", "w", encoding="utf-8") as f:
            for b in barcodes:
                f.write(f"{b}\n")
        pos_path = gbm / sample / "outs" / "spatial" / "tissue_positions_list.csv"
        with open(pos_path, "w", encoding="utf-8") as f:
            for i, b in enumerate(barcodes):
                f.write(f"{b},1,0,{i},{100+i*10},{200+i*10}\n")

    meta_rows = [
        {"spot_id": "AA-1", "sample": "S1", "centroid_x": 1.0, "centroid_y": 2.0, "mp": "A", "layer": "L1", "ivygap": "CT", "org1": "d", "org2": "d", "cc": 0.1, "cna_bin": "low"},
        {"spot_id": "BB-1", "sample": "S1", "centroid_x": 3.0, "centroid_y": 4.0, "mp": "B", "layer": "L2", "ivygap": "CT", "org1": "d", "org2": "d", "cc": 0.2, "cna_bin": "high"},
        {"spot_id": "CC-1", "sample": "S2", "centroid_x": 5.0, "centroid_y": 6.0, "mp": "C", "layer": "L1", "ivygap": "CT", "org1": "d", "org2": "d", "cc": 0.3, "cna_bin": "low"},
        {"spot_id": "DD-1", "sample": "S2", "centroid_x": 7.0, "centroid_y": 8.0, "mp": "D", "layer": "L2", "ivygap": "CT", "org1": "d", "org2": "d", "cc": 0.4, "cna_bin": "high"},
    ]
    pd.DataFrame(meta_rows).to_csv(general / "visium_metadata.csv", index=False)

    p, kind = resolve_dataset(str(gbm))
    assert kind == "gbm_cohort"
    assert p == gbm.resolve()
    bundle = load_dataset(p, kind)
    _assert_canonical_schema(bundle)
    assert bundle.expr.shape == (4, 3)
    assert bundle.obs["metadata_joined"].all()
    ing = get_ingestion_diagnostics(bundle)
    assert ing is not None
    assert ing.metadata_match_rate == 1.0


def test_true_cna_join_uses_sample_and_barcode_keys() -> None:
    obs = pd.DataFrame(
        {
            "section_id": ["S1", "S1", "S2"],
            "metadata_sample": ["S1", "S1", "S2"],
            "barcode": ["AA-1", "BB-1", "CC-1"],
        }
    )
    cna = pd.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "spot_id": ["AA-1", "CC-1"],
            "cna_score": [0.9, 0.2],
        }
    )
    joined, stats = _join_true_cna_scores(obs, cna)
    assert "cna_score" in joined.columns
    assert float(joined.loc[joined["barcode"] == "AA-1", "cna_score"].iloc[0]) == 0.9
    assert float(joined.loc[joined["barcode"] == "CC-1", "cna_score"].iloc[0]) == 0.2
    assert joined.loc[joined["barcode"] == "BB-1", "cna_score"].isna().all()
    assert bool(stats["true_cna_joined"])
    assert stats["true_cna_join_matches"] == 2


def run_integration_with_optional_data(logger: logging.Logger) -> dict:
    """Load real GBM_data (subset) when present; return manifest dict (caller writes JSON)."""
    import os

    limitations: list[str] = []
    blockers: list[str] = []
    umap_paths: list[str] = []
    stage1: dict = {}
    ing = None
    meta_path = None
    bundle = None

    if not GBM_DEFAULT.is_dir():
        limitations.append("Data/Inputs/general/GBM_data not found — skipped live Visium integration; UMAPs not generated.")
        logger.warning(limitations[-1])
    else:
        os.environ.setdefault("OMEGA_STAGE1_MAX_SAMPLES", "3")
        meta_path = discover_visium_metadata_path(GBM_DEFAULT)
        if meta_path is None or not meta_path.is_file():
            limitations.append("visium_metadata.csv not found next to GBM_data.")
        logger.info("Input GBM_data: %s", GBM_DEFAULT)
        logger.info("Resolved metadata path: %s", meta_path)

        p, kind = resolve_dataset(str(GBM_DEFAULT))
        logger.info("resolve_dataset kind=%s path=%s", kind, p)
        bundle = load_dataset(p, kind)

        _assert_canonical_schema(bundle)
        stage1 = diagnose_stage1_ingestion(bundle)
        logger.info("Shapes: expr=%s obs=%s", bundle.expr.shape, bundle.obs.shape)
        logger.info("Sections: %s", stage1.get("n_sections"))
        logger.info("Join coverage: %s", stage1.get("metadata_join_coverage"))
        logger.info("Evaluation columns present: %s", stage1.get("evaluation_columns_present"))
        logger.info("Malignancy-related columns: %s", stage1.get("malignancy_related_columns_present"))
        ing = get_ingestion_diagnostics(bundle)
        if ing:
            logger.info("Join key used: %s", ing.join_key_used)
            logger.info("Samples loaded: %s", ing.samples_loaded)
            for w in ing.warnings:
                logger.warning("%s", w)

        umap_paths = write_stage1_umaps(bundle, logger)
        for u in umap_paths:
            logger.info("UMAP artifact: %s", u)

        if stage1.get("metadata_join_coverage") is not None and stage1["metadata_join_coverage"] < 0.95:
            limitations.append(
                f"Metadata join coverage {stage1['metadata_join_coverage']:.4f} < 0.95 — inspect spot_id/barcode alignment."
            )
        elif stage1.get("metadata_join_coverage") is not None and stage1["metadata_join_coverage"] < 1.0:
            limitations.append(
                f"Partial metadata join ({stage1['metadata_join_coverage']:.2%}); unmatched spots retain Space Ranger pixels for x/y."
            )

    src = [str(GBM_DEFAULT)]
    if meta_path:
        src.append(str(meta_path))
    if ing:
        src.extend(ing.source_files)

    manifest = {
        "source_files": list(dict.fromkeys(src)),
        "output_schema": (
            {
                "obs_columns": list(bundle.obs.columns.astype(str)),
                "var_names_count": len(bundle.var_names),
                "n_spots": int(bundle.expr.shape[0]),
                "n_genes": int(bundle.expr.shape[1]),
                "dataset_kind": bundle.dataset_kind,
            }
            if bundle is not None
            else {}
        ),
        "umap_figure_paths": umap_paths,
        "test_script_path": str(TEST_SCRIPT_PATH.resolve()),
        "known_limitations": limitations,
        "blockers_for_downstream_stages": blockers,
        "stage1_diagnosis": stage1,
        "handoff_malignancy_columns": {
            "cna_bin_available": stage1.get("cna_bin_available") if stage1 else None,
            "columns_present": stage1.get("malignancy_related_columns_present") if stage1 else [],
            "note": "cna_score/malignancy_score may be absent until Stage 2 inference; cna_bin is evaluation-oriented from visium_metadata when joined.",
        },
    }
    return manifest


def main() -> None:
    logger = _setup_file_logging()
    logger.info("Stage 1 data ingestion validation starting (repo root: %s)", REPO_ROOT)

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        test_resolve_minimal_gbm_layout(Path(td))
    print("OK: minimal GBM layout fixture test passed.")
    test_true_cna_join_uses_sample_and_barcode_keys()
    print("OK: true CNA join keying")

    manifest = run_integration_with_optional_data(logger)
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote artifact manifest: %s", ARTIFACT_PATH)
    produced = [str(LOG_PATH), str(ARTIFACT_PATH)] + manifest.get("umap_figure_paths", [])
    logger.info("Produced artifacts: %s", json.dumps(produced, indent=2))

    print("Stage 1 checks complete. Log:", LOG_PATH)
    print("Manifest:", ARTIFACT_PATH)


if __name__ == "__main__":
    main()
