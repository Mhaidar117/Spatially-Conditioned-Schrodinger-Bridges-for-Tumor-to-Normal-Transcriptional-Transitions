"""
Stage 7: reporting, benchmarking, UMAP validation, logs, artifact manifest.

Run:  python tests/test_stage_7_reporting_benchmarking.py
Or:   pytest tests/test_stage_7_reporting_benchmarking.py -v
"""
from __future__ import annotations

import json
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

from omega_spatial.benchmarks import (  # noqa: E402
    run_benchmarks,
    run_benchmarks_and_baselines,
)
from omega_spatial.config import PipelineConfig  # noqa: E402
from omega_spatial.reporting import (  # noqa: E402
    render_stage7_html_block,
    run_stage7_reporting,
    write_reports,
)
from omega_spatial.synthetic_validation import run_synthetic_validation  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

LOG_PATH = REPO_ROOT / "logs" / "stage_7_reporting_benchmarking.log"
ARTIFACT_JSON_PATH = REPO_ROOT / "logs" / "stage_7_artifacts.json"
TEST_SCRIPT_PATH = REPO_ROOT / "tests" / "test_stage_7_reporting_benchmarking.py"
STAGE7_PERSIST_DIR = REPO_ROOT / "results" / "stage_7_artifacts"


def synthetic_pipeline_bundle(
    *,
    n_spots: int = 96,
    n_genes: int = 48,
    seed: int = 7,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    rng = np.random.default_rng(seed)
    expr = rng.poisson(4, size=(n_spots, n_genes)).astype(np.float64)
    noise = rng.normal(0, 0.08, size=expr.shape)
    ctx = np.clip(expr + noise, 0.0, None)
    transported = np.clip(expr * 0.92 + rng.normal(0, 0.12, size=expr.shape), 0.0, None)
    perturb = transported - expr
    n_n = n_spots // 2
    labels = np.array(["normal"] * n_n + ["tumor"] * (n_spots - n_n), dtype=object)
    section = np.array(["sec_A"] * (n_spots // 2) + ["sec_B"] * (n_spots - n_spots // 2), dtype=object)
    xy = rng.uniform(0, 10, size=(n_spots, 2))
    obs = pd.DataFrame(
        {
            "marginal_label": labels.astype(str),
            "x": xy[:, 0],
            "y": xy[:, 1],
            "layer": rng.choice(["L1", "L2", "L3"], size=n_spots),
            "mp": rng.choice(["mp1", "mp2"], size=n_spots),
            "cna_score": np.concatenate(
                [rng.normal(0.15, 0.04, size=n_n), rng.normal(0.85, 0.06, size=n_spots - n_n)]
            ),
        }
    )
    nn = NearestNeighbors(n_neighbors=min(8, n_spots - 1), metric="euclidean")
    nn.fit(xy)
    _, knn_idx = nn.kneighbors(xy)
    program_scores = pd.DataFrame(
        rng.random(size=(n_spots, 4)),
        columns=[f"program_{i}" for i in range(4)],
    )
    cna = obs["cna_score"].to_numpy(dtype=float)
    return expr, ctx, transported, perturb, program_scores, obs, section, cna, knn_idx


def test_run_benchmarks_and_baselines_schema() -> None:
    expr, _, transported, _, _, obs, section, cna, _ = synthetic_pipeline_bundle(n_spots=64)
    labels = obs["marginal_label"].to_numpy()
    df, baselines = run_benchmarks_and_baselines(expr, transported, labels, section, cna)
    assert "method" in df.columns and "metric" in df.columns and "value" in df.columns
    assert "split_scope" in df.columns
    assert "SpatialBridge" in df["method"].values
    assert "DE_shift" in df["method"].values
    assert "LatentNN_normal_blend" in df["method"].values
    assert set(baselines.keys()) >= {"observed", "SpatialBridge", "StaticOT_centroid"}


def test_run_benchmarks_backward_compatible() -> None:
    expr, _, transported, _, _, obs, section, cna, _ = synthetic_pipeline_bundle(n_spots=40)
    labels = obs["marginal_label"].to_numpy()
    df = run_benchmarks(expr, transported, labels, section, cna)
    assert len(df) > 0
    assert not df["value"].isna().all()


def test_run_stage7_produces_umaps_log_and_manifest(tmp_path: Path) -> None:
    out_dir = tmp_path / "stage7_out"
    out_dir.mkdir()
    expr, ctx, transported, perturb, program_scores, obs, section, cna, knn = synthetic_pipeline_bundle()
    labels = obs["marginal_label"].to_numpy()
    transported_linear = np.clip(expr * 0.95 + 0.03, 0.0, None)
    transported_neural = np.clip(expr * 0.90 + 0.02, 0.0, None)
    benchmark_df, baselines = run_benchmarks_and_baselines(
        expr,
        transported,
        labels,
        section,
        cna,
        spatial_methods={
            "SpatialBridge_linear": transported_linear,
            "SpatialBridge_neural": transported_neural,
        },
    )
    benchmark_df.to_csv(out_dir / "benchmark_metrics.csv", index=False)

    cfg = PipelineConfig()
    cfg.programs.umap_random_state = 42
    cfg.cna.canonical_column = "cna_score"

    manifest_file = tmp_path / "stage_7_artifacts.json"
    manifest, html_frag = run_stage7_reporting(
        repo_root=REPO_ROOT,
        out_dir=out_dir,
        cfg=cfg,
        expr=expr,
        context=ctx,
        transported=transported,
        perturb=perturb,
        program_scores=program_scores,
        obs=obs,
        section=section,
        cna=cna,
        benchmark_df=benchmark_df,
        baseline_counterfactuals=baselines,
        knn_indices=knn,
        artifact_manifest_path=manifest_file,
    )

    assert (out_dir / "stage_7_spatial_coherence.csv").is_file()
    assert (out_dir / "stage_7_biological_plausibility.csv").is_file()
    assert (out_dir / "benchmark_metrics.csv").is_file()

    umap_paths = manifest.get("umap_figure_paths") or []
    assert len(umap_paths) >= 3, f"expected >=3 UMAP figures, got {umap_paths}"
    for p in umap_paths:
        path = Path(p)
        assert path.is_file(), f"missing UMAP file: {path}"
        assert path.stat().st_size > 500, f"figure too small: {path}"

    assert LOG_PATH.is_file(), f"missing Stage 7 log: {LOG_PATH}"
    assert manifest_file.is_file(), f"missing manifest: {manifest_file}"

    loaded = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert "umap_figure_paths" in loaded
    assert "metrics_artifact_paths" in loaded
    assert "report_paths" in loaded
    assert "stage4b_gain_attribution_paths" in loaded
    ts = loaded.get("test_script_path", "").replace("\\", "/")
    assert ts.endswith("tests/test_stage_7_reporting_benchmarking.py")
    assert "unresolved_validation_gaps" in loaded

    assert "Stage 7" in html_frag
    block = render_stage7_html_block(
        pd.read_csv(out_dir / "stage_7_spatial_coherence.csv"),
        pd.read_csv(out_dir / "stage_7_biological_plausibility.csv"),
        benchmark_df,
        loaded,
    )
    assert "Coordinator handoff" in block
    attr_csv = out_dir / "stage_7_stage4b_program_gain_attribution.csv"
    assert attr_csv.is_file()


def test_write_reports_with_stage7_html(tmp_path: Path) -> None:
    out_dir = tmp_path / "report_out"
    out_dir.mkdir()
    df = pd.DataFrame([{"a": 1}])
    write_reports(
        out_dir=out_dir,
        run_name="test_run",
        readiness=df,
        qc_summary=df,
        benchmark=df,
        programs=pd.DataFrame({"p": [1.0]}, index=["g1"]),
        loadings=pd.DataFrame({"p": [1.0]}, index=["g1"]),
        figures=[],
        malignancy_summary=df,
        malignancy_counts=df,
        stage7_html="<h2>Stage 7 test</h2><p>ok</p>",
    )
    html = (out_dir / "report.html").read_text(encoding="utf-8")
    assert "Stage 7 test" in html
    assert (out_dir / "pathway_enrichment.csv").is_file()
    assert (out_dir / "best_term_per_program.csv").is_file()


def test_run_stage7_includes_synthetic_validation_when_available(tmp_path: Path) -> None:
    synth_dir = tmp_path / "synthetic_validation"
    synth_dir.mkdir()
    run_synthetic_validation(
        repo_root=REPO_ROOT,
        out_dir=synth_dir,
        cfg=PipelineConfig(),
        grid_shape=(10, 10),
        n_genes=20,
        seed=9,
    )
    synthetic_manifest = synth_dir / "synthetic_validation_manifest.json"

    out_dir = tmp_path / "stage7_out"
    out_dir.mkdir()
    expr, ctx, transported, perturb, program_scores, obs, section, cna, knn = synthetic_pipeline_bundle()
    labels = obs["marginal_label"].to_numpy()
    benchmark_df, baselines = run_benchmarks_and_baselines(expr, transported, labels, section, cna)
    benchmark_df.to_csv(out_dir / "benchmark_metrics.csv", index=False)

    cfg = PipelineConfig()
    cfg.programs.umap_random_state = 42
    cfg.cna.canonical_column = "cna_score"

    manifest, html_frag = run_stage7_reporting(
        repo_root=REPO_ROOT,
        out_dir=out_dir,
        cfg=cfg,
        expr=expr,
        context=ctx,
        transported=transported,
        perturb=perturb,
        program_scores=program_scores,
        obs=obs,
        section=section,
        cna=cna,
        benchmark_df=benchmark_df,
        baseline_counterfactuals=baselines,
        knn_indices=knn,
        artifact_manifest_path=tmp_path / "stage7_manifest.json",
        synthetic_validation_manifest_path=synthetic_manifest,
    )

    assert manifest["synthetic_validation"]["available"] is True
    assert manifest["validation_streams"]["synthetic_validation"] == "executed"
    assert "Synthetic validation" in html_frag


def _print_debug(manifest: dict, benchmark_df: pd.DataFrame, umap_paths: list[str]) -> None:
    print("--- Stage 7 debug ---")
    print("Benchmark rows:", len(benchmark_df))
    print(benchmark_df.head(12).to_string(index=False))
    print("Figure paths:")
    for p in umap_paths:
        print(" ", p)
    print("Report paths:", manifest.get("report_paths"))
    print("Metrics paths:", manifest.get("metrics_artifact_paths"))
    gaps = manifest.get("unresolved_validation_gaps") or []
    if gaps:
        print("Missing-validation / gap warnings:")
        for g in gaps:
            print(" ", g)
    else:
        print("Missing-validation warnings: none")
    print("Log:", LOG_PATH)
    print("Manifest:", ARTIFACT_JSON_PATH)


if __name__ == "__main__":
    d = STAGE7_PERSIST_DIR
    d.mkdir(parents=True, exist_ok=True)
    expr, ctx, transported, perturb, program_scores, obs, section, cna, knn = synthetic_pipeline_bundle()
    labels = obs["marginal_label"].to_numpy()
    benchmark_df, baselines = run_benchmarks_and_baselines(
        expr,
        transported,
        labels,
        section,
        cna,
        spatial_methods={
            "SpatialBridge_linear": np.clip(expr * 0.95 + 0.03, 0.0, None),
            "SpatialBridge_neural": np.clip(expr * 0.90 + 0.02, 0.0, None),
        },
    )
    benchmark_df.to_csv(d / "benchmark_metrics.csv", index=False)
    cfg = PipelineConfig()
    cfg.programs.umap_random_state = 42
    manifest, stage7_html = run_stage7_reporting(
        repo_root=REPO_ROOT,
        out_dir=d,
        cfg=cfg,
        expr=expr,
        context=ctx,
        transported=transported,
        perturb=perturb,
        program_scores=program_scores,
        obs=obs,
        section=section,
        cna=cna,
        benchmark_df=benchmark_df,
        baseline_counterfactuals=baselines,
        knn_indices=knn,
    )
    stub = pd.DataFrame([{"note": "Stage 7 persistence run (synthetic bundle); see Stage 7 HTML block for tables."}])
    prog_for_report = program_scores.copy()
    prog_for_report.index = [f"gene_{i}" for i in range(len(prog_for_report))]
    write_reports(
        out_dir=d,
        run_name="stage7_persistence",
        readiness=stub,
        qc_summary=stub,
        benchmark=benchmark_df,
        programs=prog_for_report,
        loadings=prog_for_report,
        figures=list(manifest.get("umap_figure_paths") or []),
        malignancy_summary=stub,
        malignancy_counts=stub,
        stage7_html=stage7_html,
    )
    _print_debug(manifest, benchmark_df, list(manifest.get("umap_figure_paths") or []))
    print("Manual run output dir:", d)
