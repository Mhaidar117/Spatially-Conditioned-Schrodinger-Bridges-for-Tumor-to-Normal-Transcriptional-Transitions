"""Hyperparameter sensitivity sweep for the linear spatial bridge.

Sweeps the grid ``ridge_lambda x reverse_step_size x spatial_smoothing_alpha``
and, at each combination, refits the bridge, runs transport, and records
headline perturbation statistics.  The purpose is not to search for a better
configuration but to show that the downstream conclusions are robust — i.e.,
that the mean perturbation norm and the mean movement toward healthy are
stable under reasonable perturbations of the three hyperparameters.

The output artifacts (CSV table, heatmap PNG) are wired into
``manuscript_writer.py`` so the manuscript's methods section can cite a
sensitivity appendix.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import PipelineConfig
from .model import (
    generate_counterfactuals,
    train_transport_backend,
    transport_sanity_metrics,
)
from .perturbations import compute_perturbation_matrix
from .utils import repo_relative, safe_rowwise_cosine


@dataclass
class SensitivityResult:
    """Container for sensitivity sweep outputs."""

    table: pd.DataFrame
    table_csv: Path
    heatmap_png: Path
    baseline_row: pd.Series


def _deepcopy_cfg(cfg: PipelineConfig) -> PipelineConfig:
    return copy.deepcopy(cfg)


def _fit_and_score_once(
    expr: np.ndarray,
    context: np.ndarray,
    labels: np.ndarray,
    cfg: PipelineConfig,
    *,
    baseline_perturbation: np.ndarray | None = None,
) -> dict[str, float]:
    """Train backend, run transport, return a flat dict of headline metrics."""
    cfg.bridge.backend = "linear"
    model = train_transport_backend(expr, context, labels, cfg)
    transported = generate_counterfactuals(model, expr, context)
    perturbation = compute_perturbation_matrix(expr, transported)

    perturbation_norm = np.linalg.norm(perturbation, axis=1)
    sanity = transport_sanity_metrics(expr, transported, labels, model.normal_reference)
    all_sp = sanity.get("all_spots", {})
    tumor_norm = sanity.get("tumor_vs_normal_movement", {})

    row: dict[str, float] = {
        "mean_perturbation_norm": float(perturbation_norm.mean()),
        "median_perturbation_norm": float(np.median(perturbation_norm)),
        "mean_delta_toward_healthy": float(all_sp.get("mean_delta_toward_ref", float("nan"))),
        "tumor_to_normal_movement_ratio": float(
            tumor_norm.get("tumor_to_normal_movement_ratio", float("nan"))
        ),
        "fraction_near_reference_post": float(
            sanity.get("collapse_guard", {}).get("fraction_near_reference_post", float("nan"))
        ),
    }
    # Cosine-stability of the perturbation direction vs the baseline sweep point,
    # answering: "do my perturbations point in the same direction as at the
    # reference hyperparameters?"  Averaged across spots.
    if baseline_perturbation is not None:
        cos = safe_rowwise_cosine(perturbation, baseline_perturbation)
        row["mean_cosine_vs_baseline"] = float(np.nanmean(cos))
    return row


def run_sensitivity_sweep(
    expr: np.ndarray,
    context: np.ndarray,
    labels: np.ndarray,
    *,
    out_dir: Path,
    base_cfg: PipelineConfig | None = None,
    ridge_lambdas: Sequence[float] = (1e-4, 1e-3, 1e-2, 1e-1),
    reverse_step_sizes: Sequence[float] = (0.1, 0.2, 0.4),
    alphas: Sequence[float] = (0.1, 0.3, 0.5),
    repo_root: Path | None = None,
    logger: logging.Logger | None = None,
) -> SensitivityResult:
    """Sweep the linear bridge hyperparameters and write table + heatmap.

    Parameters
    ----------
    expr, context, labels
        Stage-3 outputs (expression matrix, neighborhood-context matrix,
        marginal labels).  Non-spatial ablations can pass ``context = expr``.
    out_dir
        Directory into which ``sensitivity_table.csv`` and
        ``sensitivity_heatmap.png`` are written.
    base_cfg
        Configuration used as the starting point for each sweep combination.
        The three swept fields are overridden per combination; all other
        fields (e.g. ``transport_n_steps``) are inherited from the base.
    """
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log = logger or logging.getLogger("omega_spatial.sensitivity")

    base_cfg = base_cfg or PipelineConfig()

    # Use the geometric center of the grid as the baseline reference: results
    # from other cells are compared to this one in cosine-stability terms.
    base_lambda = float(sorted(ridge_lambdas)[len(ridge_lambdas) // 2])
    base_step = float(sorted(reverse_step_sizes)[len(reverse_step_sizes) // 2])
    base_alpha = float(sorted(alphas)[len(alphas) // 2])

    # First pass: fit the baseline and cache its perturbation for comparison.
    baseline_cfg = _deepcopy_cfg(base_cfg)
    baseline_cfg.bridge.ridge_lambda = base_lambda
    baseline_cfg.bridge.reverse_step_size = base_step
    baseline_cfg.spatial.spatial_smoothing_alpha = base_alpha
    baseline_cfg.bridge.backend = "linear"
    baseline_model = train_transport_backend(expr, context, labels, baseline_cfg)
    baseline_transported = generate_counterfactuals(baseline_model, expr, context)
    baseline_perturbation = compute_perturbation_matrix(expr, baseline_transported)

    rows: list[dict[str, float]] = []
    for lam in ridge_lambdas:
        for ss in reverse_step_sizes:
            for alpha in alphas:
                cfg = _deepcopy_cfg(base_cfg)
                cfg.bridge.ridge_lambda = float(lam)
                cfg.bridge.reverse_step_size = float(ss)
                cfg.spatial.spatial_smoothing_alpha = float(alpha)
                log.info(
                    "sensitivity: lam=%.1e step=%.2f alpha=%.2f", lam, ss, alpha
                )
                metrics = _fit_and_score_once(
                    expr,
                    context,
                    labels,
                    cfg,
                    baseline_perturbation=baseline_perturbation,
                )
                row = {
                    "ridge_lambda": float(lam),
                    "reverse_step_size": float(ss),
                    "spatial_smoothing_alpha": float(alpha),
                    **metrics,
                }
                rows.append(row)

    table = pd.DataFrame(rows)
    table_csv = out_dir / "sensitivity_table.csv"
    table.to_csv(table_csv, index=False)

    heatmap_png = out_dir / "sensitivity_heatmap.png"
    _plot_sensitivity_heatmap(
        table,
        heatmap_png,
        alphas=list(alphas),
        metric="mean_perturbation_norm",
    )

    baseline_row = table[
        (table["ridge_lambda"] == base_lambda)
        & (table["reverse_step_size"] == base_step)
        & (table["spatial_smoothing_alpha"] == base_alpha)
    ].iloc[0]

    if repo_root is not None:
        log.info(
            "wrote sensitivity table to %s", repo_relative(repo_root, table_csv)
        )
    return SensitivityResult(
        table=table,
        table_csv=table_csv,
        heatmap_png=heatmap_png,
        baseline_row=baseline_row,
    )


def _plot_sensitivity_heatmap(
    table: pd.DataFrame,
    out_path: Path,
    *,
    alphas: Iterable[float],
    metric: str = "mean_perturbation_norm",
) -> None:
    """One heatmap per alpha (columns), with ridge_lambda on y and step_size on x."""
    alphas_sorted = sorted({float(a) for a in alphas})
    n_panels = len(alphas_sorted)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(4.2 * n_panels, 3.8), dpi=180, squeeze=False
    )
    axes_row = axes[0]

    all_vals = table[metric].to_numpy(dtype=float)
    vmin = float(np.nanmin(all_vals))
    vmax = float(np.nanmax(all_vals))

    for ax, alpha in zip(axes_row, alphas_sorted, strict=False):
        sub = table[np.isclose(table["spatial_smoothing_alpha"], alpha)]
        pivot = sub.pivot_table(
            index="ridge_lambda",
            columns="reverse_step_size",
            values=metric,
            aggfunc="mean",
        )
        pivot = pivot.sort_index(axis=0).sort_index(axis=1)
        img = ax.imshow(
            pivot.to_numpy(dtype=float),
            aspect="auto",
            origin="lower",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xticks(range(pivot.columns.size))
        ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns])
        ax.set_yticks(range(pivot.index.size))
        ax.set_yticklabels([f"{r:.1e}" for r in pivot.index])
        ax.set_xlabel("reverse_step_size")
        ax.set_ylabel("ridge_lambda")
        ax.set_title(f"alpha={alpha:.2f}", fontsize=9)
        plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Hyperparameter sensitivity: {metric}\n"
        "(stable if values vary mildly across cells)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
