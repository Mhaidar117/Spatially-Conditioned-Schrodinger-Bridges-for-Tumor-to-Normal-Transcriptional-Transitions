"""Tests for the hyperparameter sensitivity sweep."""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
(REPO_ROOT / ".mpl_cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mpl_cache"))

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from omega_spatial.config import PipelineConfig  # noqa: E402
from omega_spatial.sensitivity import run_sensitivity_sweep  # noqa: E402


def test_sensitivity_sweep_writes_artifacts_and_baseline(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    n_spots, n_genes = 48, 14
    expr = rng.normal(6.0, 1.0, size=(n_spots, n_genes))
    context = expr + rng.normal(0.0, 0.3, size=expr.shape)
    labels = np.array(
        ["normal"] * (n_spots // 3)
        + ["intermediate"] * (n_spots // 3)
        + ["tumor"] * (n_spots - 2 * (n_spots // 3))
    )

    out_dir = tmp_path / "sweep"
    result = run_sensitivity_sweep(
        expr,
        context,
        labels,
        out_dir=out_dir,
        base_cfg=PipelineConfig(),
        ridge_lambdas=(1e-3, 1e-2),
        reverse_step_sizes=(0.1, 0.2),
        alphas=(0.1, 0.3),
    )
    # Table has 2x2x2 = 8 rows, all required columns present.
    assert result.table.shape[0] == 8
    for col in (
        "ridge_lambda",
        "reverse_step_size",
        "spatial_smoothing_alpha",
        "mean_perturbation_norm",
        "mean_delta_toward_healthy",
        "mean_cosine_vs_baseline",
    ):
        assert col in result.table.columns

    # Baseline row's cosine-vs-baseline is exactly 1.0 — comparing a run to
    # itself is the identity check that cosine-similarity is wired correctly.
    assert float(result.baseline_row["mean_cosine_vs_baseline"]) > 0.999

    assert result.table_csv.is_file()
    assert result.heatmap_png.is_file()
    # Heatmap PNG is non-trivial.
    assert result.heatmap_png.stat().st_size > 500


def test_sensitivity_sweep_perturbation_norms_are_finite(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    expr = rng.normal(5.0, 0.8, size=(32, 10))
    context = expr + rng.normal(0.0, 0.2, size=expr.shape)
    labels = np.array(["normal"] * 10 + ["intermediate"] * 10 + ["tumor"] * 12)

    result = run_sensitivity_sweep(
        expr,
        context,
        labels,
        out_dir=tmp_path / "sweep",
        ridge_lambdas=(1e-3,),
        reverse_step_sizes=(0.1, 0.2),
        alphas=(0.1,),
    )
    vals = result.table["mean_perturbation_norm"].to_numpy(dtype=float)
    assert np.all(np.isfinite(vals))
    assert np.all(vals >= 0.0)
