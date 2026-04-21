from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
(REPO_ROOT / ".mpl_cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mpl_cache"))

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

import manuscript_writer  # noqa: E402
import synthetic_validation as synthetic_runner  # noqa: E402
from omega_spatial.config import PipelineConfig  # noqa: E402
from omega_spatial.synthetic_validation import (  # noqa: E402
    build_toy_synthetic_validation_data,
    run_synthetic_validation,
)


def test_toy_bundle_has_expected_corner_gradient() -> None:
    synthetic = build_toy_synthetic_validation_data(grid_shape=(10, 12), n_genes=24, seed=13)
    assert synthetic.bundle.expr.shape == (120, 24)
    assert synthetic.bundle.obs["x"].min() == 0.0
    assert synthetic.bundle.obs["y"].min() == 0.0
    assert synthetic.malignancy_field.max() <= 1.0 + 1e-8
    assert synthetic.malignancy_field.min() >= 0.0

    top_left_idx = 0
    bottom_right_idx = synthetic.bundle.expr.shape[0] - 1
    assert synthetic.malignancy_field[top_left_idx] > synthetic.malignancy_field[bottom_right_idx]
    assert (
        synthetic.bundle.obs["marginal_label"].astype(str).isin(["normal", "intermediate", "tumor"]).all()
    )


def test_run_synthetic_validation_writes_outputs(tmp_path: Path) -> None:
    out_dir = tmp_path / "synthetic_validation"
    cfg = PipelineConfig()
    cfg.bridge.backend = "linear"
    manifest = run_synthetic_validation(
        repo_root=REPO_ROOT,
        out_dir=out_dir,
        cfg=cfg,
        grid_shape=(12, 12),
        n_genes=24,
        seed=7,
    )
    manifest_path = out_dir / "synthetic_validation_manifest.json"
    assert manifest_path.is_file()
    assert manifest["status"] == "executed"
    # Spatial pass produces 5 canonical figures; ablation adds one more.
    assert len(manifest["figure_paths"]) >= 5
    for rel_path in manifest["figure_paths"].values():
        assert Path(REPO_ROOT / rel_path).is_file() or Path(rel_path).is_file()

    summary = pd.read_csv(out_dir / "synthetic_validation_summary.csv")
    assert not summary.empty
    assert float(summary["mean_delta_toward_healthy"].iloc[0]) > 0.0
    assert float(summary["fraction_positive_inward_direction"].iloc[0]) >= 0.5


def test_synthetic_validation_ablation_beats_nonspatial(tmp_path: Path) -> None:
    """The spatial bridge should recover the planted perturbation with lower
    gene-space L2 error than a context-free (non-spatial) bridge when the
    observed expression is noisy — which is the regime where spatial
    conditioning's denoising-over-neighbors delivers real value.

    This is the numerical test for the scientific claim that spatial
    conditioning is doing meaningful work.  We deliberately test at a
    Visium-like per-spot noise level (sigma=1.5) because on a noiseless
    radial field the per-spot expression already determines the target and
    neighborhood averaging has nothing to denoise.  Under realistic noise,
    the spatial bridge's L2 error is roughly half that of the non-spatial
    bridge on this toy field, and its neighborhood coherence and recovery
    metrics are also recorded in the manifest for diagnostics.
    """
    out_dir = tmp_path / "ablation"
    cfg = PipelineConfig()
    cfg.bridge.backend = "linear"
    manifest = run_synthetic_validation(
        repo_root=REPO_ROOT,
        out_dir=out_dir,
        cfg=cfg,
        grid_shape=(14, 14),
        n_genes=32,
        seed=11,
        compare_ablation=True,
        noise_scale=1.5,
    )
    assert "ablation" in manifest
    spatial_err = float(manifest["ablation"]["spatial_mean_gene_l2_error"])
    nonspatial_err = float(manifest["ablation"]["nonspatial_mean_gene_l2_error"])
    assert spatial_err < nonspatial_err, (
        f"Spatial bridge (L2={spatial_err:.4f}) should beat non-spatial "
        f"bridge (L2={nonspatial_err:.4f}) on the noisy synthetic corner-seeded field"
    )
    assert manifest["ablation"]["spatial_better_l2"] is True
    # Coherence metrics should be present for diagnostics even if we don't
    # assert a specific ordering on them.
    assert "spatial_neighborhood_coherence" in manifest["ablation"]
    assert "nonspatial_neighborhood_coherence" in manifest["ablation"]
    ablation_csv = out_dir / "synthetic_validation_ablation_summary.csv"
    assert ablation_csv.is_file()
    ablation_df = pd.read_csv(ablation_csv)
    assert set(ablation_df["method_label"].unique()) == {"spatial", "nonspatial"}
    assert "neighborhood_coherence" in ablation_df.columns


def test_standalone_runner_main_writes_manifest(tmp_path: Path) -> None:
    rc = synthetic_runner.main(
        [
            "--output",
            str(tmp_path / "runner_out"),
            "--grid-rows",
            "10",
            "--grid-cols",
            "10",
            "--genes",
            "20",
            "--seed",
            "5",
        ]
    )
    assert rc == 0
    assert (tmp_path / "runner_out" / "synthetic_validation_manifest.json").is_file()


def test_manuscript_sections_include_synthetic_validation(tmp_path: Path) -> None:
    out_dir = tmp_path / "manuscript_out"
    out_dir.mkdir()
    synthetic_summary = pd.DataFrame(
        [
            {
                "mean_gene_cosine_similarity": 0.91,
                "fraction_positive_inward_direction": 0.97,
                "mean_delta_toward_healthy": 2.4,
                "recovery_strength_correlation": 0.89,
            }
        ]
    )
    fig_summary = out_dir / "synthetic_metric_summary.png"
    fig_appendix = out_dir / "synthetic_recovery_analysis.png"
    fig_summary.write_bytes(b"fake")
    fig_appendix.write_bytes(b"fake")

    results_text = manuscript_writer.write_results(
        bench=pd.DataFrame(),
        heldout=pd.DataFrame(),
        coherence=pd.DataFrame(),
        bio=pd.DataFrame(),
        factor_cmp=pd.DataFrame(),
        best_terms=pd.DataFrame(),
        synthetic_summary=synthetic_summary,
        out_dir=out_dir,
        generated_figs={},
        pipeline_figs={
            "synthetic_summary": fig_summary,
            "synthetic_appendix": fig_appendix,
        },
    )
    appendix_text = manuscript_writer.write_appendix(
        synthetic_summary=synthetic_summary,
        out_dir=out_dir,
        pipeline_figs={"synthetic_appendix": fig_appendix},
    )
    assert "Synthetic validation" in results_text
    assert "fig:synthetic_summary" in results_text
    assert "Synthetic Validation Appendix" in appendix_text
    assert "fig:synthetic_appendix" in appendix_text
