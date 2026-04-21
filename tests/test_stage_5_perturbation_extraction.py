"""
Stage 5: Perturbation extraction (transported - observed), artifacts, UMAP validation, logs.

Run:  python tests/test_stage_5_perturbation_extraction.py
Or:   pytest tests/test_stage_5_perturbation_extraction.py -v
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
(REPO_ROOT / ".mpl_cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mpl_cache"))

import matplotlib

matplotlib.use("Agg")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omega_spatial.config import PipelineConfig  # noqa: E402
from omega_spatial.model import train_spatial_bridge  # noqa: E402
from omega_spatial.model import train_transport_backend  # noqa: E402
from omega_spatial.perturbations import (  # noqa: E402
    compute_perturbation_matrix,
    extract_perturbations,
    save_perturbation_artifacts,
    stage5_handoff_for_downstream,
    write_stage5_umap_figures,
    write_stage5_artifact_manifest,
)

FIG_DIR = REPO_ROOT / "results" / "stage_5_figures"
ARTIFACT_DIR = REPO_ROOT / "results" / "stage_5_artifacts"
LOG_PATH = REPO_ROOT / "logs" / "stage_5_perturbation_extraction.log"
ARTIFACT_JSON_PATH = REPO_ROOT / "logs" / "stage_5_artifacts.json"
TEST_SCRIPT_PATH = REPO_ROOT / "tests" / "test_stage_5_perturbation_extraction.py"
UMAP_RANDOM_STATE = 42


def _setup_file_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stage_5_perturbation_extraction")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)
    return logger


def synthetic_stage5_fixture(
    rng: np.random.Generator | None = None,
    *,
    n_spots: int = 320,
    n_genes: int = 48,
    tumor_extra_counts: int = 22,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    """Visium-like counts + marginal labels (mirrors Stage 4 synthetic bridge fixture)."""
    rng = rng or np.random.default_rng(7)
    n_n = n_spots // 2
    n_t = n_spots - n_n
    expr = rng.poisson(10, size=(n_spots, n_genes)).astype(float)
    expr[n_n : n_n + n_t, 0] += rng.poisson(tumor_extra_counts, size=(n_t,)).astype(float)

    global_mu = expr.mean(axis=0, keepdims=True)
    context = np.maximum(
        0.0,
        0.72 * expr + 0.2 * global_mu + rng.poisson(1, size=expr.shape).astype(float),
    )

    labels = np.array(["normal"] * n_n + ["tumor"] * n_t, dtype=object)
    var_names = [f"gene_{j}" for j in range(n_genes)]
    obs = pd.DataFrame(
        {
            "spot_id": [f"s{i}" for i in range(n_spots)],
            "marginal_label": labels.astype(str),
        }
    )
    return expr.astype(float), context.astype(float), labels, obs, var_names


def test_perturbation_identity_exact() -> None:
    observed = np.array([[1.0, 2.0], [3.0, 4.0]])
    transported = observed + np.array([[0.5, -0.25], [1.0, 0.0]])
    p = compute_perturbation_matrix(observed, transported)
    assert np.array_equal(p, transported - observed)
    assert p.shape == observed.shape


def test_extract_raises_on_gene_mismatch() -> None:
    rng = np.random.default_rng(1)
    expr, ctx, labels, obs, var_names = synthetic_stage5_fixture(rng, n_spots=20, n_genes=10)
    cfg = PipelineConfig()
    model = train_spatial_bridge(expr, ctx, labels, cfg)
    bad_names = var_names[:-1]
    try:
        extract_perturbations(model, expr, ctx, obs, bad_names)
    except ValueError as e:
        assert "var_names" in str(e) or "columns" in str(e).lower()
    else:
        raise AssertionError("expected ValueError for gene count mismatch")


def test_extract_raises_on_row_mismatch() -> None:
    rng = np.random.default_rng(0)
    expr, ctx, _labels, obs, var_names = synthetic_stage5_fixture(rng, n_spots=20, n_genes=10)
    cfg = PipelineConfig()
    model = train_spatial_bridge(expr, ctx, obs["marginal_label"].to_numpy(), cfg)
    bad_obs = obs.iloc[:-1].copy()
    try:
        extract_perturbations(model, expr, ctx, bad_obs, var_names)
    except ValueError as e:
        assert "len(obs)" in str(e) or "match" in str(e).lower()
    else:
        raise AssertionError("expected ValueError for obs/expr row mismatch")


def test_ordering_and_identity_with_trained_bridge() -> None:
    rng = np.random.default_rng(13)
    expr, ctx, labels, obs, var_names = synthetic_stage5_fixture(rng)
    cfg = PipelineConfig()
    cfg.bridge.ridge_lambda = 1e-2
    cfg.bridge.transport_n_steps = 10
    cfg.bridge.reverse_step_size = 0.18
    cfg.spatial.spatial_smoothing_alpha = 0.25

    model = train_spatial_bridge(expr, ctx, labels, cfg)
    res = extract_perturbations(model, expr, ctx, obs, var_names, n_steps=cfg.bridge.transport_n_steps)

    assert res.obs.index.equals(obs.index)
    assert res.var_names == var_names
    assert res.transported.shape == expr.shape
    assert not np.allclose(res.transported, expr), "transport should change states on toy data"

    if not np.allclose(res.perturbation, res.transported - res.observed):
        raise AssertionError("perturbation must equal transported - observed")
    if not np.allclose(res.perturbation_norm, np.linalg.norm(res.perturbation, axis=1)):
        raise AssertionError("perturbation_norm column inconsistent with L2 row norms")

    assert np.all(np.isfinite(res.perturbation)), "non-finite perturbation values"
    d = res.diagnostics["perturbation_norm_stats"]
    assert d["nan_count"] == 0 and d["inf_count"] == 0


def test_extract_with_backend_selector_linear() -> None:
    rng = np.random.default_rng(31)
    expr, ctx, labels, obs, var_names = synthetic_stage5_fixture(rng, n_spots=128, n_genes=32)
    cfg = PipelineConfig()
    cfg.bridge.backend = "linear"
    model = train_transport_backend(expr, ctx, labels, cfg)
    res = extract_perturbations(model, expr, ctx, obs, var_names)
    assert res.transported.shape == expr.shape
    assert np.isfinite(res.transported).all()
    assert np.allclose(res.perturbation, res.transported - res.observed)


def test_umap_validation_figures_written() -> None:
    logger = logging.getLogger("stage_5_umap_test")
    logger.addHandler(logging.NullHandler())
    rng = np.random.default_rng(21)
    expr, ctx, labels, obs, var_names = synthetic_stage5_fixture(rng)
    cfg = PipelineConfig()
    cfg.bridge.ridge_lambda = 1e-2
    cfg.bridge.transport_n_steps = 10
    cfg.bridge.reverse_step_size = 0.18
    cfg.spatial.spatial_smoothing_alpha = 0.25
    model = train_spatial_bridge(expr, ctx, labels, cfg)
    res = extract_perturbations(model, expr, ctx, obs, var_names)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = write_stage5_umap_figures(
        expr,
        res.transported,
        res.perturbation,
        labels,
        FIG_DIR,
        logger,
        random_state=UMAP_RANDOM_STATE,
    )
    assert len(paths) == 3
    for p in paths:
        assert Path(p).is_file(), f"missing figure: {p}"


def run_stage5_validation(logger: logging.Logger) -> dict:
    rng = np.random.default_rng(7)
    expr, ctx, labels, obs, var_names = synthetic_stage5_fixture(rng)

    cfg = PipelineConfig()
    cfg.train.random_seed = 7
    cfg.bridge.ridge_lambda = 1e-2
    cfg.bridge.transport_n_steps = 12
    cfg.bridge.reverse_step_size = 0.18
    cfg.spatial.spatial_smoothing_alpha = 0.25

    logger.info("Bridge config: %s", json.dumps(cfg.bridge.__dict__, default=str))
    print("matrix shapes: expr", expr.shape, "context", ctx.shape)
    print("gene count:", len(var_names))
    print("label counts:\n", pd.Series(labels).value_counts().to_string())

    model = train_spatial_bridge(expr, ctx, labels, cfg)
    res = extract_perturbations(model, expr, ctx, obs, var_names, n_steps=cfg.bridge.transport_n_steps)

    logger.info("Inference entry point: %s", res.inference_entry_point)
    logger.info("Observed shape %s, transported %s, perturbation %s", res.observed.shape, res.transported.shape, res.perturbation.shape)
    logger.info("Perturbation norm stats: %s", json.dumps(res.diagnostics["perturbation_norm_stats"], indent=2))
    logger.info("Class-specific norm summaries: %s", json.dumps(res.diagnostics["class_perturbation_norm_summaries"], indent=2))
    logger.info("NaN in perturbation: %s", int(np.isnan(res.perturbation).sum()))
    logger.info("Inf in perturbation: %s", int(np.isinf(res.perturbation).sum()))

    if not np.allclose(res.perturbation, res.transported - res.observed):
        logger.error("Perturbation identity violated")
        raise AssertionError("perturbation != transported - observed")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    paths = save_perturbation_artifacts(ARTIFACT_DIR, res)
    for k, v in paths.items():
        logger.info("Wrote artifact [%s]: %s", k, v)

    umap_paths = write_stage5_umap_figures(
        expr,
        res.transported,
        res.perturbation,
        labels,
        FIG_DIR,
        logger,
        random_state=UMAP_RANDOM_STATE,
    )
    for p in umap_paths:
        logger.info("UMAP figure: %s", p)

    handoff = stage5_handoff_for_downstream(ARTIFACT_DIR)
    logger.info("Stages 6–7 handoff: %s", json.dumps(handoff, indent=2))

    annotated_paths = {
        "note": (
            "This test run writes matrices under results/stage_5_artifacts/. "
            "run_pipeline() additionally writes annotated_output_obs.csv, perturbation_vectors.csv, "
            "and annotated_output.h5ad under cfg.output_path."
        ),
    }

    manifest_extra = {
        "perturbation_identity": "perturbation = transported_state - observed_state",
        "transport_sanity_metrics": res.transport_sanity,
        "diagnostics": res.diagnostics,
        "umap_random_state": UMAP_RANDOM_STATE,
        "pipeline_run": False,
    }
    write_stage5_artifact_manifest(
        ARTIFACT_JSON_PATH,
        ARTIFACT_DIR,
        perturbation_paths=paths,
        annotated_paths=annotated_paths,
        umap_paths=umap_paths,
        summary_figure_paths=[],
        test_script_path=str(TEST_SCRIPT_PATH.resolve()),
        known_limitations=[
            "Perturbations are computed for all spots; restrict with marginal_label for tumor-only analyses.",
            "Toy fixture uses Poisson-like counts; real Visium may need QC and transport hyperparameter tuning.",
            "Linear ridge bridge (Stage 4) is not a full generative or optimal-transport counterfactual model.",
        ],
        extra=manifest_extra,
    )
    logger.info("Wrote manifest: %s", ARTIFACT_JSON_PATH)

    print("perturbation norm stats:", json.dumps(res.diagnostics["perturbation_norm_stats"], indent=2))
    print("class summaries:", json.dumps(res.diagnostics["class_perturbation_norm_summaries"], indent=2))
    print("transport sanity:", json.dumps(res.transport_sanity, indent=2))

    return {
        "artifact_paths": paths,
        "umap_paths": umap_paths,
        "manifest_path": str(ARTIFACT_JSON_PATH.resolve()),
        "log_path": str(LOG_PATH.resolve()),
    }


def main() -> None:
    logger = _setup_file_logging()
    logger.info("Stage 5 perturbation extraction validation (repo root: %s)", REPO_ROOT)
    out = run_stage5_validation(logger)
    produced = [out["log_path"], out["manifest_path"]] + list(out["artifact_paths"].values()) + out["umap_paths"]
    logger.info("Produced artifacts: %s", json.dumps(produced, indent=2))
    print("Stage 5 checks complete. Log:", LOG_PATH)
    print("Manifest:", ARTIFACT_JSON_PATH)


if __name__ == "__main__":
    main()
