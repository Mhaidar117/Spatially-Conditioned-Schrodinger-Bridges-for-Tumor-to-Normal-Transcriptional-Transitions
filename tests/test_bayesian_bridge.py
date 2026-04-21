"""Tests for the Bayesian ridge bridge backend and uncertainty propagation."""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
(REPO_ROOT / ".mpl_cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mpl_cache"))

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from omega_spatial.config import PipelineConfig  # noqa: E402
from omega_spatial.model import (  # noqa: E402
    BayesianRidgeBridgeModel,
    SpatialBridgeModel,
    train_transport_backend,
)
from omega_spatial.perturbations import extract_perturbations  # noqa: E402


def _toy_bundle(n_spots: int = 60, n_genes: int = 16, seed: int = 3):
    rng = np.random.default_rng(seed)
    expr = rng.normal(6.0, 1.0, size=(n_spots, n_genes))
    context = rng.normal(6.0, 1.0, size=(n_spots, n_genes))
    labels = np.array(
        ["normal"] * (n_spots // 3)
        + ["intermediate"] * (n_spots // 3)
        + ["tumor"] * (n_spots - 2 * (n_spots // 3))
    )
    obs = pd.DataFrame(
        {
            "spot_id": [f"s{i}" for i in range(n_spots)],
            "marginal_label": labels,
        }
    )
    var_names = [f"g{j}" for j in range(n_genes)]
    return expr, context, labels, obs, var_names


def test_bayesian_backend_matches_linear_in_mean_prediction() -> None:
    """Bayesian ridge must return the same mean drift as the deterministic ridge.

    Both fits solve exactly the same regularized least-squares problem, so the
    posterior mean equals the deterministic point estimate.
    """
    expr, context, labels, _, _ = _toy_bundle()
    cfg_linear = PipelineConfig()
    cfg_linear.bridge.backend = "linear"
    cfg_bayes = PipelineConfig()
    cfg_bayes.bridge.backend = "bayesian_linear"

    linear = train_transport_backend(expr, context, labels, cfg_linear)
    bayes = train_transport_backend(expr, context, labels, cfg_bayes)

    assert isinstance(linear, SpatialBridgeModel)
    assert isinstance(bayes, BayesianRidgeBridgeModel)
    assert bayes.backend == "bayesian_linear"

    # Same weights modulo floating-point — both solve the same ridge system.
    np.testing.assert_allclose(bayes.w_expr, linear.w_expr, atol=1e-8)
    np.testing.assert_allclose(bayes.w_ctx, linear.w_ctx, atol=1e-8)
    np.testing.assert_allclose(bayes.bias, linear.bias, atol=1e-8)

    # Same scoring function, so transport-step outputs coincide.
    s_lin = linear.score(expr, context)
    s_bay = bayes.score(expr, context)
    np.testing.assert_allclose(s_lin, s_bay, atol=1e-8)


def test_bayesian_predictive_std_is_positive_and_shape_correct() -> None:
    expr, context, labels, _, _ = _toy_bundle()
    cfg = PipelineConfig()
    cfg.bridge.backend = "bayesian_linear"
    model = train_transport_backend(expr, context, labels, cfg)

    drift_std = model.predictive_drift_std(expr, context)
    assert drift_std.shape == (expr.shape[0],)
    assert np.all(drift_std >= 0.0)
    assert np.all(np.isfinite(drift_std))
    assert float(drift_std.mean()) > 0.0  # Non-degenerate noise estimate.

    pert_std = model.predictive_perturbation_std(expr, context)
    assert pert_std.shape == (expr.shape[0],)
    assert np.all(pert_std >= 0.0)
    # Perturbation std grows with step_size * n_steps.
    assert float(pert_std.mean()) >= float(drift_std.mean()) * abs(
        model.reverse_step_size
    )


def test_extract_perturbations_surfaces_uncertainty_column() -> None:
    expr, context, labels, obs, var_names = _toy_bundle()
    cfg = PipelineConfig()
    cfg.bridge.backend = "bayesian_linear"
    model = train_transport_backend(expr, context, labels, cfg)
    result = extract_perturbations(model, expr, context, obs, var_names)

    assert "perturbation_norm_std" in result.obs.columns
    std_col = result.obs["perturbation_norm_std"].to_numpy(dtype=float)
    assert std_col.shape == (expr.shape[0],)
    assert np.all(std_col >= 0.0)
    assert "perturbation_norm_std_stats" in result.diagnostics


def test_linear_backend_has_no_uncertainty_column() -> None:
    """Smoke: linear backend should not write perturbation_norm_std (no-op path)."""
    expr, context, labels, obs, var_names = _toy_bundle()
    cfg = PipelineConfig()
    cfg.bridge.backend = "linear"
    model = train_transport_backend(expr, context, labels, cfg)
    result = extract_perturbations(model, expr, context, obs, var_names)
    assert "perturbation_norm_std" not in result.obs.columns
