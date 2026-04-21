from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

# Ensure non-interactive plotting with writable cache in restricted envs.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "omega_mplconfig"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Template
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec

from .benchmarks import biological_plausibility_summary, spatial_coherence_summary
from .config import PipelineConfig, ReportConfig
from .perturbations import (
    MARGINAL_COLORS,
    _embedding_for_umap,
    _prepare_expression_for_embedding,
)
from .programs import infer_program_display_names
from .synthetic_validation import load_synthetic_validation_artifacts
from .utils import repo_relative

try:
    from scipy.stats import hypergeom
except Exception:  # noqa: BLE001
    hypergeom = None

STAGE7_LOG_NAME = "omega_spatial.stage7"

# Backward-compatible alias for manifest-writing helpers.
_repo_relative = repo_relative


def _scatter_marginal_on_ax(
    ax: Axes,
    emb: np.ndarray,
    labels: np.ndarray,
    title: str,
    basis_note: str,
) -> None:
    labs = pd.Series(labels).astype(str)
    for cat in ("normal", "intermediate", "tumor"):
        m = labs == cat
        if not m.any():
            continue
        color = MARGINAL_COLORS.get(cat, "#bcbd22")
        ax.scatter(emb[m, 0], emb[m, 1], s=10, alpha=0.85, c=color, label=cat, edgecolors="none")
    other = ~labs.isin(list(MARGINAL_COLORS.keys()))
    if other.any():
        ax.scatter(emb[other, 0], emb[other, 1], s=8, alpha=0.55, c="#cccccc", label="other")
    ax.set_xlabel("UMAP 1" if "UMAP" in basis_note else "Component 1")
    ax.set_ylabel("UMAP 2" if "UMAP" in basis_note else "Component 2")
    ax.set_title(f"{title}\n{basis_note}", fontsize=9)
    ax.legend(title="marginal_label", fontsize=7, loc="best")


def write_stage7_umap_stage_progression(
    expr: np.ndarray,
    context: np.ndarray,
    transported: np.ndarray,
    perturb: np.ndarray,
    program_scores: pd.DataFrame,
    obs: pd.DataFrame,
    fig_dir: Path,
    logger: logging.Logger,
    *,
    random_state: int,
) -> list[str]:
    """
    Multi-panel progression: ingestion/QC proxy → context → transported → perturbation → programs.
    Each panel uses its own embedding (different feature spaces); seed is fixed for reproducibility.
    """
    fig_dir = Path(fig_dir).expanduser().resolve()
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    n = expr.shape[0]
    labels = obs["marginal_label"].to_numpy() if "marginal_label" in obs.columns else np.array(["unknown"] * n, dtype=object)
    pnorm = np.linalg.norm(np.asarray(perturb, dtype=float), axis=1)
    w = program_scores.to_numpy(dtype=float)
    dominant = np.argmax(w, axis=1)
    dom_labels = np.array([str(program_scores.columns[j]) for j in dominant], dtype=object)

    fig = plt.figure(figsize=(15, 10), dpi=150)
    gs = GridSpec(2, 3, figure=fig, wspace=0.28, hspace=0.35)
    panels: list[tuple[str, np.ndarray, str]] = [
        (
            "Stages 1–2: observed expression (log-CPM; ingestion / QC proxy)",
            _prepare_expression_for_embedding(expr),
            "marginal",
        ),
        (
            "Stage 3: spatial context (neighbor-mean expression, log-CPM)",
            _prepare_expression_for_embedding(context),
            "marginal",
        ),
        (
            "Stages 4–5: transported state (log-CPM)",
            _prepare_expression_for_embedding(transported),
            "marginal",
        ),
        (
            "Stage 5: perturbation vectors (raw gene space)",
            np.asarray(perturb, dtype=float),
            "perturb_norm",
        ),
        (
            "Stage 6: NMF program scores (per-spot activations)",
            w,
            "program",
        ),
    ]

    for i, (title, xmat, mode) in enumerate(panels):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        emb, note = _embedding_for_umap(xmat, random_state=random_state + i)
        if mode == "marginal":
            _scatter_marginal_on_ax(ax, emb, labels, title, note)
        elif mode == "perturb_norm":
            sc = ax.scatter(emb[:, 0], emb[:, 1], c=pnorm, s=10, alpha=0.88, cmap="viridis")
            fig.colorbar(sc, ax=ax, label="perturbation_norm", fraction=0.046)
            ax.set_xlabel("UMAP 1" if "UMAP" in note else "C1")
            ax.set_ylabel("UMAP 2" if "UMAP" in note else "C2")
            ax.set_title(f"{title}\n{note}\nColormap: viridis (perceptually uniform)", fontsize=9)
        else:
            uniq = sorted(set(dom_labels.tolist()), key=lambda x: str(x))
            tab = colormaps["tab10"]
            denom = max(len(uniq) - 1, 1)
            for ui, u in enumerate(uniq):
                m = dom_labels == u
                ax.scatter(
                    emb[m, 0],
                    emb[m, 1],
                    s=10,
                    alpha=0.88,
                    color=tab(ui / denom),
                    label=u,
                )
            ax.set_xlabel("UMAP 1" if "UMAP" in note else "C1")
            ax.set_ylabel("UMAP 2" if "UMAP" in note else "C2")
            ax.set_title(
                f"{title}\n{note}\nLegend: dominant program (argmax NMF score)",
                fontsize=9,
            )
            ax.legend(title="dominant program", fontsize=6, loc="best")

    ax_leg = fig.add_subplot(gs[1, 2])
    ax_leg.axis("off")
    ax_leg.text(
        0.02,
        0.95,
        "Handoff (coordinator):\n"
        "• marginal_label = weak supervision (not injected into transport).\n"
        "• Separate UMAP per feature space; same random seed family (base+i).\n"
        "• Stages 4–5: transported / perturb from spatial bridge (Stage 4).\n"
        "• Stage 6: NMF on nonnegative-shifted perturbations (see Stage 6 logs).",
        va="top",
        fontsize=9,
        family="sans-serif",
    )

    fig.suptitle(
        "Stage 7 — stage-by-stage UMAP progression (full pipeline summary)\n"
        "Axes: UMAP dimensions per panel. Validation / manuscript-oriented layout.",
        fontsize=12,
        y=0.98,
    )
    out = fig_dir / "stage_7_umap_stage_progression.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(out.resolve()))
    logger.info("Stage 7 UMAP artifact: %s", out)

    return paths


def write_stage7_umap_baseline_comparison(
    baseline_counterfactuals: dict[str, np.ndarray],
    labels: np.ndarray,
    fig_dir: Path,
    logger: logging.Logger,
    *,
    random_state: int,
) -> list[str]:
    """
    Shared embedding of stacked prepared gene-space matrices: observed vs bridge vs baselines.
    """
    fig_dir = Path(fig_dir).expanduser().resolve()
    fig_dir.mkdir(parents=True, exist_ok=True)
    order = [
        ("observed", "Observed (reference)"),
        ("SpatialBridge", "Spatial bridge (transported)"),
        ("SpatialBridge_linear", "Spatial bridge (linear backend)"),
        ("SpatialBridge_neural", "Spatial bridge (neural backend)"),
        ("StaticOT_centroid", "Baseline: static OT / global shift"),
        ("UnconditionalBridge", "Baseline: unconditional bridge"),
        ("DE_shift", "Baseline: DE shift (tumor toward normal mean)"),
        ("LatentNN_normal_blend", "Baseline: latent NN normal blend"),
    ]
    mats: list[np.ndarray] = []
    titles: list[str] = []
    n: int | None = None
    for key, ttitle in order:
        if key not in baseline_counterfactuals:
            logger.warning("Stage 7: missing baseline key %s (omitted from comparison UMAP)", key)
            continue
        x = _prepare_expression_for_embedding(baseline_counterfactuals[key])
        if n is None:
            n = x.shape[0]
        elif x.shape[0] != n:
            logger.warning("Stage 7: row mismatch for %s; omitted from baseline UMAP", key)
            continue
        mats.append(x)
        titles.append(ttitle)

    if len(mats) < 2 or n is None:
        logger.warning("Stage 7 baseline UMAP skipped: insufficient baseline matrices")
        return []

    z = np.vstack(mats)
    emb, note = _embedding_for_umap(z, random_state=random_state)
    lims = (
        float(emb[:, 0].min() - 0.5),
        float(emb[:, 0].max() + 0.5),
        float(emb[:, 1].min() - 0.5),
        float(emb[:, 1].max() + 0.5),
    )

    n_panels = len(mats)
    ncols = 3
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.8 * nrows), dpi=150, squeeze=False)
    lab = np.asarray(labels).astype(str)

    for idx in range(nrows * ncols):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        if idx >= n_panels:
            ax.axis("off")
            continue
        sl = slice(idx * n, (idx + 1) * n)
        e = emb[sl]
        _scatter_marginal_on_ax(ax, e, lab, titles[idx], note)
        ax.set_xlim(lims[0], lims[1])
        ax.set_ylim(lims[2], lims[3])

    fig.suptitle(
        "Stage 7 — baseline vs spatial bridge (shared UMAP embedding)\n"
        f"{note}\n"
        "Same axis limits across panels; legend: marginal_label (weak supervision). "
        "Internal validation — not held-out section performance.",
        fontsize=11,
    )
    fig.tight_layout()
    out = fig_dir / "stage_7_umap_baseline_comparison.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths = [str(out.resolve())]
    logger.info("Stage 7 UMAP artifact: %s", out)
    return paths


def write_stage7_umap_benchmark_variables(
    expr: np.ndarray,
    perturb: np.ndarray,
    obs: pd.DataFrame,
    fig_dir: Path,
    logger: logging.Logger,
    *,
    random_state: int,
) -> tuple[list[str], list[str]]:
    """
    UMAP on observed expression; panels for marginal_label, perturbation_norm, optional evaluation columns.
    Returns (paths, warnings).
    """
    fig_dir = Path(fig_dir).expanduser().resolve()
    fig_dir.mkdir(parents=True, exist_ok=True)
    warns: list[str] = []
    x = _prepare_expression_for_embedding(expr)
    emb, note = _embedding_for_umap(x, random_state=random_state)
    pnorm = np.linalg.norm(np.asarray(perturb, dtype=float), axis=1)
    n = len(emb)
    if len(obs) != n:
        warns.append(f"obs rows ({len(obs)}) != embedding rows ({n}); benchmark-variable UMAP skipped")
        logger.warning(warns[-1])
        return [], warns

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=150)
    ax00, ax01, ax10, ax11 = axes.ravel()

    if "marginal_label" in obs.columns:
        _scatter_marginal_on_ax(ax00, emb, obs["marginal_label"].to_numpy(), "marginal_label (weak supervision)", note)
    else:
        ax00.text(0.5, 0.5, "marginal_label missing", ha="center")
        warns.append("marginal_label column missing")
    sc = ax01.scatter(emb[:, 0], emb[:, 1], c=pnorm, s=12, alpha=0.88, cmap="viridis")
    fig.colorbar(sc, ax=ax01, label="perturbation_norm")
    ax01.set_xlabel("UMAP 1" if "UMAP" in note else "C1")
    ax01.set_ylabel("UMAP 2" if "UMAP" in note else "C2")
    ax01.set_title(f"perturbation_norm (L2 ||u||)\n{note}")

    for ax, col, name in ((ax10, "layer", "layer"), (ax11, "mp", "mp")):
        if col in obs.columns:
            s = obs[col].astype(str).to_numpy()
            uniq = sorted(set(s.tolist()), key=lambda x: str(x))
            cmap = colormaps["tab20"]
            den = max(len(uniq) - 1, 1)
            for ui, u in enumerate(uniq):
                m = s == u
                ax.scatter(emb[m, 0], emb[m, 1], s=10, alpha=0.85, color=cmap(ui / den), label=u)
            ax.set_xlabel("UMAP 1" if "UMAP" in note else "C1")
            ax.set_ylabel("UMAP 2" if "UMAP" in note else "C2")
            ax.set_title(
                f"{name} (evaluation / interpretation only — not used in transport)\n{note}",
                fontsize=9,
            )
            ax.legend(fontsize=6, loc="best", title=name)
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{col} not in obs\n(panel skipped)", ha="center", va="center")
            warns.append(f"{col} column missing; evaluation panel omitted")

    fig.suptitle(
        "Stage 7 — UMAP colored by benchmark-relevant variables\n"
        "Shared embedding from observed expression (log-CPM). See per-panel titles for supervision vs evaluation.",
        fontsize=11,
    )
    fig.tight_layout()
    out = fig_dir / "stage_7_umap_benchmark_variables.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths = [str(out.resolve())]
    logger.info("Stage 7 UMAP artifact: %s", out)
    return paths, warns


def write_stage7_summary_figures(
    benchmark_df: pd.DataFrame,
    coherence: pd.DataFrame,
    plausibility: pd.DataFrame,
    fig_dir: Path,
    logger: logging.Logger,
) -> list[str]:
    """
    Manuscript-facing Stage 7 summary panel covering baselines, coherence, and plausibility.
    """
    fig_dir = Path(fig_dir).expanduser().resolve()
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=150)
    ax0, ax1, ax2 = axes

    sub = benchmark_df.copy()
    if {"metric", "split_scope"}.issubset(sub.columns):
        filtered = sub[
            (sub["metric"].astype(str) == "distance_to_normal_mean_l2")
            & (sub["split_scope"].astype(str) == "spot_level_internal")
        ]
        if not filtered.empty:
            sub = filtered
    if {"method", "value"}.issubset(sub.columns) and not sub.empty:
        vals = sub[["method", "value"]].dropna().sort_values("value", ascending=True)
        ax0.barh(vals["method"].astype(str), vals["value"].astype(float), color="steelblue")
        ax0.set_title("Distance to normal benchmark")
        ax0.set_xlabel("distance_to_normal_mean_l2")
    else:
        ax0.text(0.5, 0.5, "Benchmark data unavailable", ha="center", va="center")
        ax0.set_axis_off()

    coh = coherence.copy()
    if {"metric", "value"}.issubset(coh.columns) and not coh.empty:
        coh_vals = coh[["metric", "value"]].dropna().head(6)
        ax1.barh(coh_vals["metric"].astype(str), coh_vals["value"].astype(float), color="darkslateblue")
        ax1.set_title("Spatial coherence summary")
        ax1.set_xlabel("value")
    else:
        ax1.text(0.5, 0.5, "Coherence summary unavailable", ha="center", va="center")
        ax1.set_axis_off()

    bio = plausibility.copy()
    if {"stratum", "mean_perturbation_norm"}.issubset(bio.columns) and not bio.empty:
        plot_df = bio[bio["stratum"].astype(str).str.startswith("marginal_label:", na=False)].copy()
        if not plot_df.empty:
            plot_df["label_name"] = plot_df["stratum"].astype(str).str.split(":", n=1).str[-1]
            ax2.bar(
                plot_df["label_name"].astype(str),
                plot_df["mean_perturbation_norm"].astype(float),
                color=["#d62728" if x == "tumor" else "#7f7f7f" if x == "intermediate" else "#1f77b4" for x in plot_df["label_name"]],
            )
            ax2.set_title("Perturbation burden by marginal")
            ax2.set_ylabel("mean_perturbation_norm")
        else:
            ax2.text(0.5, 0.5, "Plausibility summary unavailable", ha="center", va="center")
            ax2.set_axis_off()
    else:
        ax2.text(0.5, 0.5, "Plausibility summary unavailable", ha="center", va="center")
        ax2.set_axis_off()

    fig.suptitle(
        "Stage 7 — validation summary\n"
        "Internal benchmark, spatial coherence, and evaluation-only biological plausibility.",
        fontsize=11,
    )
    fig.tight_layout()
    out = fig_dir / "stage_7_validation_summary.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Stage 7 summary figure: %s", out)
    return [str(out.resolve())]


def _distance_to_reference(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(x, dtype=float) - ref.reshape(1, -1), axis=1)


def write_stage7_backend_comparison_figure(
    *,
    expr: np.ndarray,
    baseline_counterfactuals: dict[str, np.ndarray],
    obs: pd.DataFrame,
    fig_dir: Path,
    logger: logging.Logger,
) -> list[str]:
    """Bar plot for mean movement toward normal reference across transport methods."""
    fig_dir = Path(fig_dir).expanduser().resolve()
    fig_dir.mkdir(parents=True, exist_ok=True)
    x = np.asarray(expr, dtype=float)
    if "marginal_label" in obs.columns and len(obs) == x.shape[0]:
        labels = obs["marginal_label"].astype(str).to_numpy()
        normal_idx = np.where(labels == "normal")[0]
    else:
        normal_idx = np.array([], dtype=int)
    if normal_idx.size == 0:
        normal_idx = np.arange(x.shape[0])
    ref = x[normal_idx].mean(axis=0)
    pre = _distance_to_reference(x, ref)

    methods: list[str] = []
    vals: list[float] = []
    for name in (
        "SpatialBridge_linear",
        "SpatialBridge_neural",
        "SpatialBridge",
        "StaticOT_centroid",
        "UnconditionalBridge",
        "DE_shift",
        "LatentNN_normal_blend",
    ):
        arr = baseline_counterfactuals.get(name)
        if arr is None:
            continue
        post = _distance_to_reference(np.asarray(arr, dtype=float), ref)
        methods.append(name)
        vals.append(float(np.mean(pre - post)))
    if not methods:
        return []

    fig, ax = plt.subplots(figsize=(8.8, 4.5), dpi=160)
    ax.bar(methods, vals, color="#4c78a8")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_ylabel("mean_delta_toward_reference")
    ax.set_title("Stage 7 — transport comparison by movement toward normal reference")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    out = fig_dir / "stage_7_stage4b_transport_comparison.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Stage 7 summary figure: %s", out)
    return [str(out.resolve())]


def write_stage7_program_gain_attribution(
    *,
    expr: np.ndarray,
    baseline_counterfactuals: dict[str, np.ndarray],
    obs: pd.DataFrame,
    program_scores: pd.DataFrame,
    program_loadings: pd.DataFrame | None,
    out_dir: Path,
    fig_dir: Path,
    logger: logging.Logger,
) -> dict[str, str]:
    """
    Program-level attribution for Stage 4b gains.
    Gain target = (delta_toward_ref_neural - delta_toward_ref_linear) per spot.
    """
    out_dir = Path(out_dir).expanduser().resolve()
    fig_dir = Path(fig_dir).expanduser().resolve()
    linear = baseline_counterfactuals.get("SpatialBridge_linear")
    neural = baseline_counterfactuals.get("SpatialBridge_neural")
    if linear is None or neural is None or program_scores.empty:
        return {}

    x = np.asarray(expr, dtype=float)
    lin = np.asarray(linear, dtype=float)
    neu = np.asarray(neural, dtype=float)
    if x.shape != lin.shape or x.shape != neu.shape or len(program_scores) != x.shape[0]:
        return {}

    if "marginal_label" in obs.columns and len(obs) == x.shape[0]:
        labels = obs["marginal_label"].astype(str).to_numpy()
        normal_idx = np.where(labels == "normal")[0]
    else:
        normal_idx = np.array([], dtype=int)
    if normal_idx.size == 0:
        normal_idx = np.arange(x.shape[0])
    ref = x[normal_idx].mean(axis=0)
    pre = _distance_to_reference(x, ref)
    gain = (pre - _distance_to_reference(neu, ref)) - (pre - _distance_to_reference(lin, ref))

    rows: list[dict[str, Any]] = []
    for col in program_scores.columns:
        vals = pd.to_numeric(program_scores[col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(vals) & np.isfinite(gain)
        if int(mask.sum()) < 4:
            corr = float("nan")
        else:
            corr = float(np.corrcoef(vals[mask], gain[mask])[0, 1])
        rows.append(
            {
                "program": str(col),
                "gain_corr": corr,
                "gain_mean_top_decile": float(np.nanmean(gain[vals >= np.nanquantile(vals, 0.9)])),
                "gain_mean_bottom_decile": float(np.nanmean(gain[vals <= np.nanquantile(vals, 0.1)])),
            }
        )
    df = pd.DataFrame(rows)
    df["abs_gain_corr"] = df["gain_corr"].abs()
    if program_loadings is not None and not program_loadings.empty:
        try:
            name_map = infer_program_display_names(program_loadings, top_k=12)
            df["program_name"] = df["program"].map(lambda p: name_map.get(str(p), str(p)))
        except Exception:
            df["program_name"] = df["program"].astype(str)
    else:
        df["program_name"] = df["program"].astype(str)
    df = df.sort_values("abs_gain_corr", ascending=False).reset_index(drop=True)

    csv_path = out_dir / "stage_7_stage4b_program_gain_attribution.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Stage 7 attribution table: %s", csv_path)

    top = df.head(min(12, len(df)))
    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=170)
    ax.barh(top["program_name"], top["gain_corr"], color="#54a24b")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("corr(program_score, stage4b_gain_vs_linear)")
    ax.set_title("Stage 7 — program attribution of Stage 4b gains")
    fig.tight_layout()
    fig_path = fig_dir / "stage_7_stage4b_program_gain_attribution.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Stage 7 attribution figure: %s", fig_path)
    return {
        "program_gain_attribution_csv": str(csv_path.resolve()),
        "program_gain_attribution_png": str(fig_path.resolve()),
    }


def run_stage7_reporting(
    repo_root: Path,
    out_dir: Path,
    cfg: PipelineConfig,
    expr: np.ndarray,
    context: np.ndarray,
    transported: np.ndarray,
    perturb: np.ndarray,
    program_scores: pd.DataFrame,
    obs: pd.DataFrame,
    section: np.ndarray,
    cna: np.ndarray,
    benchmark_df: pd.DataFrame,
    baseline_counterfactuals: dict[str, np.ndarray],
    knn_indices: np.ndarray | None,
    transport_backend: str = "linear",
    linear_transported: np.ndarray | None = None,
    neural_transported: np.ndarray | None = None,
    program_loadings: pd.DataFrame | None = None,
    *,
    artifact_manifest_path: Path | None = None,
    synthetic_validation_manifest_path: Path | None = None,
) -> tuple[dict[str, Any], str]:
    """
    Stage 7: benchmark tables on disk, coherence / plausibility summaries, UMAP validation figures,
    repo-level log + artifact manifest for the coordinator.

    If ``artifact_manifest_path`` is set (e.g. by tests), the JSON manifest is written there instead of
    ``logs/stage_7_artifacts.json`` so integration tests do not clobber the repo manifest.

    Returns (manifest dict, HTML fragment for report.html).
    """
    repo_root = Path(repo_root).resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    logs_dir = repo_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "stage_7_reporting_benchmarking.log"
    fig_dir = out_dir / "figures"

    logger = logging.getLogger(STAGE7_LOG_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)

    rs = cfg.programs.umap_random_state
    unresolved: list[str] = []
    validation_streams: dict[str, str] = {
        "synthetic_validation": "not_executed_in_stage7_pipeline_run",
        "internal_biological_stratification": "attempted_via_marginal_label_and_cna_tables",
        "spatial_coherence": "attempted_neighbor_correlation_of_perturbation_norm",
        "baseline_comparison": "executed_distance_metrics_and_shared_umap",
        "held_out_section_patient_cohort": "not_executed_spot_and_section_macro_metrics_are_internal_only",
        "st_codex_cross_modal": "optional_not_run",
        "stage4b_pathway_gain_attribution": "attempted_when_linear_and_neural_outputs_available",
    }

    logger.info("Stage 7 validation streams: %s", json.dumps(validation_streams, indent=2))
    logger.info("Stage 7 selected transport backend: %s", transport_backend)
    logger.info(
        "Baselines evaluated (gene-space counterfactuals): %s",
        [k for k in baseline_counterfactuals if k != "observed"],
    )

    pnorm = np.linalg.norm(np.asarray(perturb, dtype=float), axis=1)
    coh = spatial_coherence_summary(pnorm, knn_indices)
    coh_path = out_dir / "stage_7_spatial_coherence.csv"
    coh.to_csv(coh_path, index=False)
    logger.info("Wrote spatial coherence summary: %s", coh_path)

    bio = biological_plausibility_summary(obs, pnorm, cna, cna_column=cfg.cna.canonical_column)
    bio_path = out_dir / "stage_7_biological_plausibility.csv"
    if bio.empty:
        unresolved.append("biological_plausibility_summary_empty_check_marginal_label_and_cna_alignment")
        logger.warning("Biological plausibility table empty or minimal")
        pd.DataFrame(
            columns=["stratum", "label", "n_spots", "mean_perturbation_norm", "mean_cna"]
        ).to_csv(bio_path, index=False)
    else:
        bio.to_csv(bio_path, index=False)
    logger.info("Wrote biological plausibility summary: %s", bio_path)

    bench_path = out_dir / "benchmark_metrics.csv"
    logger.info("Benchmark metrics path (pipeline): %s", bench_path)

    synthetic_manifest, synthetic_summary_df, synthetic_spotwise_df = load_synthetic_validation_artifacts(
        repo_root,
        manifest_path=synthetic_validation_manifest_path,
    )
    synthetic_figure_paths_for_html: dict[str, str] = {}
    if synthetic_manifest is not None:
        validation_streams["synthetic_validation"] = str(synthetic_manifest.get("status", "executed"))
        for key, rel_path in (synthetic_manifest.get("figure_paths") or {}).items():
            abs_path = repo_root / str(rel_path)
            if abs_path.is_file():
                synthetic_figure_paths_for_html[key] = os.path.relpath(abs_path, out_dir).replace("\\", "/")
        logger.info(
            "Loaded synthetic validation artifacts: %s",
            json.dumps(
                {
                    "manifest": str(synthetic_validation_manifest_path or synthetic_manifest.get("artifact_manifest_path", "")),
                    "summary_rows": int(len(synthetic_summary_df)),
                    "spotwise_rows": int(len(synthetic_spotwise_df)),
                },
                indent=2,
            ),
        )

    umap_paths: list[str] = []
    summary_paths: list[str] = []
    try:
        umap_paths.extend(
            write_stage7_umap_stage_progression(
                expr,
                context,
                transported,
                perturb,
                program_scores,
                obs,
                fig_dir,
                logger,
                random_state=rs,
            )
        )
    except Exception as ex:  # noqa: BLE001
        unresolved.append(f"stage_progression_umap_failed:{ex}")
        logger.exception("Stage 7 progression UMAP failed: %s", ex)

    try:
        umap_paths.extend(
            write_stage7_umap_baseline_comparison(
                baseline_counterfactuals,
                obs["marginal_label"].to_numpy() if "marginal_label" in obs.columns else np.array(["unknown"] * len(obs)),
                fig_dir,
                logger,
                random_state=rs + 50,
            )
        )
    except Exception as ex:  # noqa: BLE001
        unresolved.append(f"baseline_comparison_umap_failed:{ex}")
        logger.exception("Stage 7 baseline UMAP failed: %s", ex)

    try:
        extra_paths, bench_warns = write_stage7_umap_benchmark_variables(
            expr, perturb, obs, fig_dir, logger, random_state=rs + 30
        )
        umap_paths.extend(extra_paths)
        for w in bench_warns:
            logger.warning("Benchmark-variable UMAP note: %s", w)
            unresolved.append(w)
    except Exception as ex:  # noqa: BLE001
        unresolved.append(f"benchmark_variables_umap_failed:{ex}")
        logger.exception("Stage 7 benchmark-variable UMAP failed: %s", ex)

    try:
        summary_paths.extend(write_stage7_summary_figures(benchmark_df, coh, bio, fig_dir, logger))
    except Exception as ex:  # noqa: BLE001
        unresolved.append(f"stage7_summary_figure_failed:{ex}")
        logger.exception("Stage 7 summary figure failed: %s", ex)

    backend_cmp_paths: list[str] = []
    try:
        backend_cmp_paths = write_stage7_backend_comparison_figure(
            expr=expr,
            baseline_counterfactuals=baseline_counterfactuals,
            obs=obs,
            fig_dir=fig_dir,
            logger=logger,
        )
        summary_paths.extend(backend_cmp_paths)
    except Exception as ex:  # noqa: BLE001
        unresolved.append(f"stage4b_backend_comparison_figure_failed:{ex}")
        logger.exception("Stage 7 backend comparison figure failed: %s", ex)

    program_attr_paths: dict[str, str] = {}
    try:
        program_attr_paths = write_stage7_program_gain_attribution(
            expr=expr,
            baseline_counterfactuals=baseline_counterfactuals,
            obs=obs,
            program_scores=program_scores,
            program_loadings=program_loadings,
            out_dir=out_dir,
            fig_dir=fig_dir,
            logger=logger,
        )
    except Exception as ex:  # noqa: BLE001
        unresolved.append(f"stage4b_program_gain_attribution_failed:{ex}")
        logger.exception("Stage 7 program gain attribution failed: %s", ex)

    report_html = out_dir / "report.html"
    report_pdf = out_dir / "report.pdf"
    for p in umap_paths:
        logger.info("Stage 7 UMAP output path: %s", p)

    test_script_path = _repo_relative(repo_root, repo_root / "tests" / "test_stage_7_reporting_benchmarking.py")
    manifest: dict[str, Any] = {
        "metrics_artifact_paths": {
            "benchmark_metrics_csv": _repo_relative(repo_root, bench_path),
            "spatial_coherence_csv": _repo_relative(repo_root, coh_path),
            "biological_plausibility_csv": _repo_relative(repo_root, bio_path),
        },
        "figures_directory": _repo_relative(repo_root, fig_dir),
        "report_paths": {
            "report_html": _repo_relative(repo_root, report_html),
            "report_pdf": _repo_relative(repo_root, report_pdf),
        },
        "umap_figure_paths": [_repo_relative(repo_root, p) for p in umap_paths],
        "summary_figure_paths": [_repo_relative(repo_root, p) for p in summary_paths],
        "test_script_path": test_script_path,
        "unresolved_validation_gaps": unresolved,
        "baselines_executed": [k for k in baseline_counterfactuals if k != "observed"],
        "transport_backend_selected": transport_backend,
        "transport_backend_comparison_available": bool(
            ("SpatialBridge_linear" in baseline_counterfactuals) and ("SpatialBridge_neural" in baseline_counterfactuals)
        ),
        "validation_streams": validation_streams,
        "synthetic_validation": {
            "available": bool(synthetic_manifest is not None),
            "artifact_manifest_path": (
                _repo_relative(repo_root, synthetic_validation_manifest_path)
                if synthetic_validation_manifest_path is not None
                else _repo_relative(repo_root, repo_root / str(synthetic_manifest.get("artifact_manifest_path", "")))
                if synthetic_manifest is not None and synthetic_manifest.get("artifact_manifest_path")
                else ""
            ),
            "summary_metrics": synthetic_manifest.get("summary_metrics", {}) if synthetic_manifest is not None else {},
            "figure_paths": synthetic_manifest.get("figure_paths", {}) if synthetic_manifest is not None else {},
        },
        "stage4b_gain_attribution_paths": {
            k: _repo_relative(repo_root, v) for k, v in program_attr_paths.items()
        },
        "coordinator_handoff": {
            "baselines": [
                "SpatialBridge (model transport)",
                "StaticOT_centroid (global mean re-centering toward normal reference)",
                "UnconditionalBridge (fixed blend toward normal)",
                "DE_shift (tumor bulk shift toward normal mean)",
                "LatentNN_normal_blend (expression kNN among normal spots)",
            ],
            "caveats": [
                "Section-macro metrics aggregate within-run sections; they do not constitute held-out generalization.",
                "layer, mp, ivygap, org1, org2, cc, CODEX are evaluation-only when present; do not treat as training signals.",
                "Distance-to-normal is an internal plausibility probe, not a clinical endpoint.",
                "Stage 4b gain-attribution analyses are associative and should not be interpreted as causal pathway interventions.",
            ],
            "key_outputs": [
                _repo_relative(repo_root, bench_path),
                _repo_relative(repo_root, coh_path),
                _repo_relative(repo_root, bio_path),
            ],
        },
    }
    manifest_path = Path(artifact_manifest_path) if artifact_manifest_path is not None else logs_dir / "stage_7_artifacts.json"
    manifest["artifact_manifest_path"] = _repo_relative(repo_root, manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote artifact manifest: %s", manifest_path)
    logger.info("Stage 7 complete; unresolved gaps: %s", unresolved if unresolved else "none")

    html_fragment = render_stage7_html_block(
        coh,
        bio,
        benchmark_df,
        manifest,
        synthetic_summary=synthetic_summary_df,
        synthetic_spotwise=synthetic_spotwise_df,
        synthetic_figure_paths=synthetic_figure_paths_for_html,
    )

    fh.flush()
    fh.close()
    logger.removeHandler(fh)

    return manifest, html_fragment


def render_stage7_html_block(
    coherence: pd.DataFrame,
    plausibility: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    manifest: dict[str, Any],
    *,
    synthetic_summary: pd.DataFrame | None = None,
    synthetic_spotwise: pd.DataFrame | None = None,
    synthetic_figure_paths: dict[str, str] | None = None,
) -> str:
    """HTML fragment for report.html (validation separation, handoff)."""
    gaps = manifest.get("unresolved_validation_gaps") or []
    streams = manifest.get("validation_streams") or {}
    synthetic_summary = synthetic_summary if synthetic_summary is not None else pd.DataFrame()
    synthetic_spotwise = synthetic_spotwise if synthetic_spotwise is not None else pd.DataFrame()
    synthetic_figure_paths = synthetic_figure_paths or {}
    synthetic_html = "<p>Synthetic validation was not executed.</p>"
    if manifest.get("synthetic_validation", {}).get("available"):
        synthetic_parts: list[str] = ["<h3>Synthetic validation (known-ground-truth recovery)</h3>"]
        if not synthetic_summary.empty:
            synthetic_parts.append(synthetic_summary.fillna("N/A").to_html(index=False, na_rep="N/A"))
        if not synthetic_spotwise.empty:
            cols = [
                c
                for c in [
                    "true_malignancy",
                    "true_recovery_strength",
                    "inferred_recovery_strength",
                    "gene_cosine_similarity",
                    "gene_l2_error",
                    "delta_toward_healthy",
                ]
                if c in synthetic_spotwise.columns
            ]
            if cols:
                synthetic_parts.append("<p><strong>Spotwise synthetic metrics (first 10 rows):</strong></p>")
                synthetic_parts.append(
                    synthetic_spotwise.loc[:, cols].head(10).fillna("N/A").to_html(index=False, na_rep="N/A")
                )
        for fig_key in (
            "synthetic_recovery_analysis_png",
            "synthetic_metric_summary_png",
        ):
            fig_path = synthetic_figure_paths.get(fig_key)
            if fig_path:
                synthetic_parts.append(f'<div><img src="{fig_path}" width="700"></div>')
        synthetic_html = "".join(synthetic_parts)
    return f"""
<h2>Stage 7 — Reporting, benchmarking, and validation notes</h2>
<p><strong>Validation streams (status labels):</strong></p>
<pre style="background:#f6f8fa;padding:10px;">{json.dumps(streams, indent=2)}</pre>
{synthetic_html}
<h3>Spatial coherence (evaluation)</h3>
{coherence.fillna("N/A").to_html(index=False, na_rep="N/A")}
<h3>Biological plausibility summaries (evaluation / stratification)</h3>
{plausibility.fillna("N/A").to_html(index=False, na_rep="N/A") if not plausibility.empty else "<p>No stratified table (check marginal_label / CNA alignment).</p>"}
<h3>Benchmark metrics (extended)</h3>
{benchmark_df.fillna("N/A").to_html(index=False, na_rep="N/A")}
<h3>Unresolved validation gaps</h3>
<ul>{"".join(f"<li>{g}</li>" for g in gaps) or "<li>None recorded</li>"}</ul>
<h3>Coordinator handoff</h3>
<pre style="background:#f6f8fa;padding:10px;">{json.dumps(manifest.get("coordinator_handoff", {}), indent=2)}</pre>
"""


def make_figures(
    out_dir: Path,
    obs: pd.DataFrame,
    perturb_mag: np.ndarray,
    program_scores: pd.DataFrame,
    program_loadings: pd.DataFrame | None = None,
) -> list[str]:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    x_col = "x" if "x" in obs.columns else obs.columns[0]
    y_col = "y" if "y" in obs.columns else obs.columns[1]
    plt.figure(figsize=(6, 5))
    plt.scatter(obs[x_col], obs[y_col], c=perturb_mag, s=8, cmap="viridis")
    plt.title("Spatial perturbation magnitude")
    plt.colorbar(label="||u||")
    path1 = fig_dir / "spatial_perturbation_map.png"
    plt.tight_layout()
    plt.savefig(path1, dpi=150)
    plt.close()
    paths.append(str(path1))

    if program_scores.empty:
        return paths
    top_cols = (
        program_scores.mean(axis=0)
        .sort_values(ascending=False)
        .index[: min(6, program_scores.shape[1])]
        .tolist()
    )
    top = program_scores.loc[:, top_cols]
    name_map: dict[str, str] = {}
    if program_loadings is not None and not program_loadings.empty:
        try:
            name_map = infer_program_display_names(program_loadings, top_k=12)
        except Exception:
            name_map = {}

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(12, 4.8),
        dpi=180,
        gridspec_kw={"width_ratios": [2.2, 1.0]},
        layout="constrained",
    )
    display_labels = [name_map.get(str(c), str(c)) for c in top.columns]

    # Left panel: score distributions are easier to compare than overlaid traces.
    vals = [pd.to_numeric(top[c], errors="coerce").fillna(0.0).to_numpy(dtype=float) for c in top.columns]
    vp = ax_left.violinplot(vals, showmeans=False, showmedians=True, widths=0.85)
    cmap = colormaps["tab10"]
    denom = max(len(vals) - 1, 1)
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(cmap(i / denom))
        body.set_alpha(0.35)
        body.set_edgecolor("#333333")
        body.set_linewidth(0.6)
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        if key in vp:
            vp[key].set_color("#333333")
            vp[key].set_linewidth(1.0)
    ax_left.set_xticks(np.arange(1, len(display_labels) + 1))
    ax_left.set_xticklabels(display_labels, rotation=25, ha="right", fontsize=8)
    ax_left.set_ylabel("Program score")
    ax_left.set_title("Program score distributions across spots")
    ax_left.grid(axis="y", alpha=0.2, linewidth=0.6)

    # Right panel: dominant-program prevalence keeps weaker programs visible.
    dominant = np.argmax(program_scores.to_numpy(dtype=float), axis=1)
    dom_series = pd.Series([str(program_scores.columns[i]) for i in dominant], dtype="object")
    dom_share = dom_series.value_counts(normalize=True).sort_values(ascending=True)
    dom_labels = [name_map.get(str(c), str(c)) for c in dom_share.index.tolist()]
    ax_right.barh(dom_labels, dom_share.to_numpy(), color="#4c78a8", alpha=0.9)
    for yi, frac in enumerate(dom_share.to_numpy()):
        ax_right.text(frac + 0.005, yi, f"{100*frac:.1f}%", va="center", fontsize=8)
    ax_right.set_xlim(0.0, max(0.25, float(dom_share.max()) * 1.2))
    ax_right.set_xlabel("Fraction of spots")
    ax_right.set_title("Dominant program prevalence")
    ax_right.grid(axis="x", alpha=0.2, linewidth=0.6)

    fig.suptitle("Top program scores across spots", fontsize=12)
    path2 = fig_dir / "top_program_scores.png"
    fig.savefig(path2, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(path2))
    return paths


def _benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = min(prev, ranked[i] * n / rank)
        q[i] = val
        prev = val
    out = np.empty(n, dtype=float)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def _load_gmt_gene_sets(gmt_path: Path, min_size: int, max_size: int) -> dict[str, set[str]]:
    sets: dict[str, set[str]] = {}
    if not gmt_path.is_file():
        return sets
    with gmt_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            term = parts[0].strip()
            genes = {"".join(ch for ch in g.upper() if ch.isalnum()) for g in parts[2:] if g.strip()}
            genes.discard("")
            if min_size <= len(genes) <= max_size:
                sets[term] = genes
    return sets


def _ora_pathway_enrichment(
    loadings: pd.DataFrame,
    *,
    gmt_path: Path,
    top_n: int,
    min_size: int,
    max_size: int,
) -> pd.DataFrame:
    if hypergeom is None:
        return pd.DataFrame()
    gene_sets = _load_gmt_gene_sets(gmt_path, min_size=min_size, max_size=max_size)
    if not gene_sets:
        return pd.DataFrame()
    name_map = infer_program_display_names(loadings, top_k=12)
    genes_raw = [str(g) for g in loadings.index.tolist()]
    genes_norm = ["".join(ch for ch in g.upper() if ch.isalnum()) for g in genes_raw]
    universe = {g for g in genes_norm if g}
    m = len(universe)
    if m == 0:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for col in loadings.columns:
        s = pd.to_numeric(loadings[col], errors="coerce").fillna(0.0)
        rank_idx = s.sort_values(ascending=False).index.tolist()
        top_idx = rank_idx[: min(top_n, len(rank_idx))]
        top_norm = {
            "".join(ch for ch in str(i).upper() if ch.isalnum())
            for i in top_idx
            if str(i).strip()
        }
        top_norm = {g for g in top_norm if g in universe}
        n = len(top_norm)
        if n == 0:
            continue
        for term, members in gene_sets.items():
            kset = members & universe
            k = len(kset)
            if k == 0:
                continue
            overlap_genes = sorted(top_norm & kset)
            x = len(overlap_genes)
            if x == 0:
                continue
            pval = float(hypergeom.sf(x - 1, m, k, n))
            expected = (n * k) / float(m)
            rows.append(
                {
                    "program": str(col),
                    "program_name": name_map.get(str(col), str(col)),
                    "pathway": term,
                    "overlap": x,
                    "set_size": k,
                    "query_size": n,
                    "universe_size": m,
                    "expected_overlap": expected,
                    "fold_enrichment": float(x / expected) if expected > 0 else float("nan"),
                    "p_value": pval,
                    "leading_edge_genes": ";".join(overlap_genes[:50]),
                }
            )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["fdr_bh"] = _benjamini_hochberg(df["p_value"].to_numpy(dtype=float))
    return df.sort_values(["fdr_bh", "p_value", "fold_enrichment"], ascending=[True, True, False]).reset_index(drop=True)


def _simple_pathway_enrichment(loadings: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    name_map = infer_program_display_names(loadings, top_k=min(top_n, 12))
    rows = []
    for col in loadings.columns:
        top_genes = loadings[col].sort_values(ascending=False).head(top_n).index.tolist()
        rows.append(
            {
                "program": col,
                "program_name": name_map.get(str(col), str(col)),
                "pathway": "TopWeightedGenes",
                "genes": ";".join(top_genes),
            }
        )
    return pd.DataFrame(rows)


def _best_term_per_program(pathway: pd.DataFrame) -> pd.DataFrame:
    if pathway.empty or "program" not in pathway.columns:
        return pd.DataFrame(columns=["program", "program_name", "best_term", "source"])
    rows: list[dict[str, Any]] = []
    for program_id, sub in pathway.groupby("program", dropna=False):
        sub = sub.copy()
        program_name = str(sub.get("program_name", pd.Series(dtype=str)).iloc[0]) if "program_name" in sub.columns else str(program_id)
        source = "fallback_top_weighted_genes"
        best_term = "TopWeightedGenes"
        p_value = float("nan")
        fdr_bh = float("nan")
        fold_enrichment = float("nan")
        overlap = float("nan")

        if "fdr_bh" in sub.columns and "pathway" in sub.columns:
            ranked = sub.sort_values(["fdr_bh", "p_value", "fold_enrichment"], ascending=[True, True, False], na_position="last")
            top = ranked.iloc[0]
            source = "ora_hypergeometric"
            best_term = str(top.get("pathway", "NA"))
            p_value = float(top.get("p_value", np.nan))
            fdr_bh = float(top.get("fdr_bh", np.nan))
            fold_enrichment = float(top.get("fold_enrichment", np.nan))
            overlap = float(top.get("overlap", np.nan))
        elif "pathway" in sub.columns:
            top = sub.iloc[0]
            best_term = str(top.get("pathway", "TopWeightedGenes"))

        rows.append(
            {
                "program": str(program_id),
                "program_name": program_name,
                "best_term": best_term,
                "source": source,
                "p_value": p_value,
                "fdr_bh": fdr_bh,
                "fold_enrichment": fold_enrichment,
                "overlap": overlap,
            }
        )
    return pd.DataFrame(rows).sort_values("program").reset_index(drop=True)


def _write_stage4b_pathway_gain_artifacts(out_dir: Path, best_terms: pd.DataFrame) -> dict[str, str]:
    """
    Join Stage 7 program gain attribution with per-program pathway labels.
    Emits the requested Stage 4b pathway gain CSV + figure if inputs exist.
    """
    out_dir = Path(out_dir).expanduser().resolve()
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    prog_path = out_dir / "stage_7_stage4b_program_gain_attribution.csv"
    if not prog_path.is_file() or best_terms.empty:
        return {}
    prog = pd.read_csv(prog_path)
    if prog.empty:
        return {}
    join = prog.merge(best_terms[["program", "best_term", "source"]], on="program", how="left")
    join["best_term"] = join["best_term"].fillna("TopWeightedGenes")
    # Use absolute correlation rank as a lightweight significance proxy when only one association statistic is present.
    join["rank_by_abs_effect"] = join["abs_gain_corr"].rank(method="dense", ascending=False)
    join["fdr_proxy"] = (join["rank_by_abs_effect"] / max(1, len(join))).clip(upper=1.0)
    out_csv = out_dir / "stage_7_stage4b_pathway_gain_attribution.csv"
    join.to_csv(out_csv, index=False)

    top = join.sort_values("abs_gain_corr", ascending=False).head(min(12, len(join))).copy()
    fig, ax = plt.subplots(figsize=(9.0, 5.5), dpi=170)
    labels = top.apply(lambda r: f"{r['program']} | {r['best_term']}", axis=1)
    ax.barh(labels, top["gain_corr"], color="#54a24b")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("stage4b_vs_linear_gain_correlation")
    ax.set_title("Stage 7 — pathway/program effects linked to Stage 4b gains")
    fig.tight_layout()
    out_fig = fig_dir / "stage_7_stage4b_vs_linear_pathway_effects.png"
    fig.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {
        "stage4b_pathway_gain_attribution_csv": str(out_csv.resolve()),
        "stage4b_pathway_effects_png": str(out_fig.resolve()),
    }


def write_reports(
    out_dir: Path,
    run_name: str,
    readiness: pd.DataFrame,
    qc_summary: pd.DataFrame,
    benchmark: pd.DataFrame,
    programs: pd.DataFrame,
    loadings: pd.DataFrame,
    figures: list[str],
    malignancy_summary: pd.DataFrame,
    malignancy_counts: pd.DataFrame,
    *,
    stage7_html: str = "",
    report_cfg: ReportConfig | None = None,
) -> None:
    report_cfg = report_cfg or ReportConfig()
    gmt_path = Path(report_cfg.enrichment_gmt_path).expanduser()
    pathway = pd.DataFrame()
    if report_cfg.enrichment_gmt_path and gmt_path.is_file():
        pathway = _ora_pathway_enrichment(
            loadings,
            gmt_path=gmt_path,
            top_n=max(20, int(report_cfg.enrichment_top_genes)),
            min_size=max(2, int(report_cfg.enrichment_min_set_size)),
            max_size=max(10, int(report_cfg.enrichment_max_set_size)),
        )
    if pathway.empty:
        pathway = _simple_pathway_enrichment(loadings)
    pathway.to_csv(out_dir / "pathway_enrichment.csv", index=False)
    best_terms = _best_term_per_program(pathway)
    best_terms.to_csv(out_dir / "best_term_per_program.csv", index=False)
    stage4b_pathway_gain_paths = _write_stage4b_pathway_gain_artifacts(out_dir, best_terms)

    template = Template(
        """
<html>
<head><title>{{ run_name }}</title></head>
<body>
<h1>{{ run_name }} - Biologist Interpretation Report</h1>
{% if cna_warning %}
<div style="border:1px solid #cc9900;padding:10px;margin:10px 0;background:#fff9e6;">
<strong>Data warning:</strong> CNA/malignancy scores were not detected in this run.
Tumor/normal marginals cannot be strictly defined by CNA without supplying a CNA column.
</div>
{% endif %}
<h2>Dataset readiness</h2>
{{ readiness_html }}
<h2>QC summary</h2>
{{ qc_html }}
<h2>Benchmark summary</h2>
{{ bench_html }}
<h2>Malignancy scoring summary</h2>
{{ malignancy_summary_html }}
<h2>Malignancy counts by section</h2>
{{ malignancy_counts_html }}
<h2>Top gene programs</h2>
{{ programs_html }}
<h2>Candidate intervention modules</h2>
<p>Programs with highest aggregate perturbation loadings are candidate intervention modules.</p>
{% if stage4b_pathway_gain_note %}
<h2>Stage 4b pathway gain attribution</h2>
<p>{{ stage4b_pathway_gain_note }}</p>
{% endif %}
{% if stage7_html %}
{{ stage7_html | safe }}
{% endif %}
<h2>Figures</h2>
{% for fig in figures %}
<div><img src="{{ fig }}" width="700"></div>
{% endfor %}
</body>
</html>
"""
    )
    html = template.render(
        run_name=run_name,
        cna_warning=(
            bool(readiness.get("recommendations", pd.Series(dtype=str)).astype(str).str.contains("Missing CNA", regex=False).any())
            or bool(readiness.get("cna_column", pd.Series(dtype=str)).fillna("N/A").astype(str).str.contains("N/A", regex=False).any())
        ),
        readiness_html=readiness.fillna("N/A").to_html(index=False, na_rep="N/A"),
        qc_html=qc_summary.fillna("N/A").to_html(index=False, na_rep="N/A"),
        bench_html=benchmark.fillna("N/A").to_html(index=False, na_rep="N/A"),
        malignancy_summary_html=malignancy_summary.fillna("N/A").to_html(index=False, na_rep="N/A"),
        malignancy_counts_html=malignancy_counts.fillna("N/A").to_html(index=False, na_rep="N/A"),
        programs_html=programs.head(20).rename_axis("gene").reset_index().fillna("N/A").to_html(index=False, na_rep="N/A"),
        figures=[f"figures/{Path(f).name}" if Path(f).exists() else f for f in figures],
        stage7_html=stage7_html,
        stage4b_pathway_gain_note=(
            "Pathway-attribution artifacts were generated for Stage 4b vs linear gain analysis."
            if stage4b_pathway_gain_paths
            else ""
        ),
    )
    (out_dir / "report.html").write_text(html, encoding="utf-8")

    # PDF fallback with minimal content if no html->pdf backend.
    pdf_path = out_dir / "report.pdf"
    try:
        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(pdf_path) as pdf:
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.05, 0.95, f"{run_name}\nBiologist Interpretation Summary", va="top", fontsize=14)
            fig.text(0.05, 0.86, "See report.html for full interactive details.", fontsize=10)
            fig.text(0.05, 0.80, benchmark.to_string(index=False)[:2800], family="monospace", fontsize=7)
            pdf.savefig(fig)
            plt.close(fig)
    except Exception:
        pdf_path.write_text("PDF generation fallback. Please open report.html.", encoding="utf-8")
