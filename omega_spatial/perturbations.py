"""
Stage 5: Transported states, perturbation vectors (transported - observed), and artifact I/O.

Downstream handoff: perturbation rows align 1:1 with ``obs`` row order and gene columns with ``var_names``.
Perturbations are computed for **all spots**; filter using ``marginal_label`` in ``obs`` when needed.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .model import (
    TransportModel,
    per_spot_l2_distance_to_reference,
    transport_sanity_metrics,
    transport_states,
)

UMAP_RANDOM_STATE_DEFAULT = 42

# visualization_standards.md: tumor warm, normal cool, intermediate neutral
MARGINAL_COLORS: dict[str, str] = {
    "tumor": "#d62728",
    "normal": "#1f77b4",
    "intermediate": "#7f7f7f",
}


@dataclass
class PerturbationExtractionResult:
    """Container for Stage 5 outputs (observed kept for provenance; may share memory with input expr)."""

    observed: np.ndarray
    transported: np.ndarray
    perturbation: np.ndarray
    perturbation_norm: np.ndarray
    obs: pd.DataFrame
    var_names: list[str]
    inference_entry_point: str
    perturbations_for_all_spots: bool = True
    n_steps: int | None = None
    step_size: float | None = None
    transport_sanity: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_summary_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("observed", None)
        d.pop("transported", None)
        d.pop("perturbation", None)
        d["observed_shape"] = list(self.observed.shape)
        d["transported_shape"] = list(self.transported.shape)
        d["perturbation_shape"] = list(self.perturbation.shape)
        d["n_obs_rows"] = int(len(self.obs))
        d["n_genes"] = len(self.var_names)
        return d


def compute_perturbation_matrix(observed: np.ndarray, transported: np.ndarray) -> np.ndarray:
    """``perturbation = transported - observed`` with shape checks."""
    x = np.asarray(observed, dtype=float)
    t = np.asarray(transported, dtype=float)
    if x.shape != t.shape:
        raise ValueError(f"observed/transported shape mismatch: {x.shape} vs {t.shape}")
    return t - x


def _norm_stats(perturbation: np.ndarray) -> dict[str, Any]:
    norms = np.linalg.norm(perturbation, axis=1)
    return {
        "mean": float(norms.mean()),
        "std": float(norms.std()),
        "min": float(norms.min()),
        "max": float(norms.max()),
        "median": float(np.median(norms)),
        "nan_count": int(np.isnan(perturbation).sum()),
        "inf_count": int(np.isinf(perturbation).sum()),
    }


def _class_norm_summaries(perturbation: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    lab = np.asarray(labels).astype(str)
    norms = np.linalg.norm(perturbation, axis=1)
    out: dict[str, Any] = {}
    for name in ("tumor", "normal", "intermediate"):
        m = lab == name
        if not np.any(m):
            continue
        out[name] = {
            "n": int(m.sum()),
            "mean_perturbation_norm": float(norms[m].mean()),
            "std_perturbation_norm": float(norms[m].std()),
        }
    return out


def extract_perturbations(
    model: TransportModel,
    expr: np.ndarray,
    context: np.ndarray,
    obs: pd.DataFrame,
    var_names: list[str],
    *,
    n_steps: int | None = None,
    step_size: float | None = None,
    inference_entry_point: str = "omega_spatial.model.transport_states",
) -> PerturbationExtractionResult:
    """
    Run Stage 4 transport on all spots and derive perturbation = transported - observed.

    Does not use evaluation-only labels inside the transport map; labels are only used post-hoc in diagnostics.
    """
    x = np.asarray(expr, dtype=float)
    c = np.asarray(context, dtype=float)
    if x.shape != c.shape:
        raise ValueError(f"expr and context must have the same shape; got {x.shape} vs {c.shape}")
    if x.shape[0] != len(obs):
        raise ValueError(f"expr rows ({x.shape[0]}) must match len(obs) ({len(obs)})")
    if x.shape[1] != len(var_names):
        raise ValueError(f"expr columns ({x.shape[1]}) must match len(var_names) ({len(var_names)})")

    transported = transport_states(model, x, c, n_steps=n_steps, step_size=step_size)
    perturbation = compute_perturbation_matrix(x, transported)
    if not np.allclose(perturbation, transported - x):
        raise RuntimeError("internal error: perturbation identity violated")

    perturbation_norm = np.linalg.norm(perturbation, axis=1)
    labels = obs["marginal_label"].to_numpy() if "marginal_label" in obs.columns else np.array(["unknown"] * len(obs))

    sanity = transport_sanity_metrics(x, transported, labels, model.normal_reference)
    diag: dict[str, Any] = {
        "perturbation_norm_stats": _norm_stats(perturbation),
        "class_perturbation_norm_summaries": _class_norm_summaries(perturbation, labels),
    }
    warn_list: list[str] = []
    if "marginal_label" not in obs.columns:
        warn_list.append("obs missing 'marginal_label'; class summaries use placeholder 'unknown'")
    if warn_list:
        diag["warnings"] = warn_list

    obs_out = obs.copy()
    obs_out["perturbation_norm"] = perturbation_norm

    ns = n_steps if n_steps is not None else model.default_n_steps
    ss = step_size if step_size is not None else model.reverse_step_size

    # If the backend exposes closed-form predictive uncertainty (Bayesian ridge),
    # surface a per-spot std estimate on the obs frame alongside the point
    # estimate.  Silent no-op for models without uncertainty (linear/neural).
    if hasattr(model, "predictive_perturbation_std"):
        try:
            perturbation_norm_std = model.predictive_perturbation_std(
                x, c, n_steps=ns, step_size=ss
            )
            obs_out["perturbation_norm_std"] = np.asarray(perturbation_norm_std, dtype=float)
            diag["perturbation_norm_std_stats"] = {
                "mean": float(np.nanmean(perturbation_norm_std)),
                "median": float(np.nanmedian(perturbation_norm_std)),
                "max": float(np.nanmax(perturbation_norm_std)),
            }
        except Exception as exc:  # noqa: BLE001
            diag.setdefault("warnings", []).append(
                f"predictive_perturbation_std unavailable: {exc!r}"
            )

    return PerturbationExtractionResult(
        observed=x,
        transported=transported,
        perturbation=perturbation,
        perturbation_norm=perturbation_norm,
        obs=obs_out,
        var_names=list(var_names),
        inference_entry_point=inference_entry_point,
        perturbations_for_all_spots=True,
        n_steps=int(ns),
        step_size=float(ss),
        transport_sanity=sanity,
        diagnostics=diag,
    )


def save_perturbation_artifacts(out_dir: Path, result: PerturbationExtractionResult) -> dict[str, str]:
    """Write matrix artifacts and gene names; returns resolved path strings."""
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}

    npz_path = out_dir / "stage_5_expression_bundle.npz"
    np.savez_compressed(
        npz_path,
        observed=result.observed.astype(np.float64),
        transported=result.transported.astype(np.float64),
        perturbation=result.perturbation.astype(np.float64),
    )
    paths["expression_bundle_npz"] = str(npz_path.resolve())

    genes_path = out_dir / "stage_5_gene_names.json"
    genes_path.write_text(json.dumps({"var_names": result.var_names}, indent=2), encoding="utf-8")
    paths["gene_names_json"] = str(genes_path.resolve())

    summary_cols = ["perturbation_norm"]
    if "marginal_label" in result.obs.columns:
        summary_cols.insert(0, "marginal_label")
    for c in ("section_id", "sample_id", "barcode"):
        if c in result.obs.columns and c not in summary_cols:
            summary_cols.insert(0, c)
    spot_path = out_dir / "stage_5_spot_summary.csv"
    result.obs[summary_cols].copy().to_csv(spot_path, index=True)
    paths["spot_summary_csv"] = str(spot_path.resolve())

    # Tabular gene-space exports (same convention as legacy perturbation_vectors.csv)
    pd.DataFrame(result.transported, columns=result.var_names).to_csv(out_dir / "transported_expression.csv", index=False)
    paths["transported_expression_csv"] = str((out_dir / "transported_expression.csv").resolve())
    pd.DataFrame(result.perturbation, columns=result.var_names).to_csv(out_dir / "perturbation_matrix.csv", index=False)
    paths["perturbation_matrix_csv"] = str((out_dir / "perturbation_matrix.csv").resolve())

    return paths


def _prepare_expression_for_embedding(expr: np.ndarray, max_genes: int | None = 1500) -> np.ndarray:
    x = np.maximum(np.asarray(expr, dtype=float), 0.0)
    if max_genes is not None:
        g = min(max_genes, x.shape[1])
        x = x[:, :g]
    lib = x.sum(axis=1, keepdims=True) + 1e-8
    return np.log1p(x / lib * 1e4)


def _embedding_for_umap(X: np.ndarray, random_state: int) -> tuple[np.ndarray, str]:
    from .utils import umap_or_pca_2d

    return umap_or_pca_2d(X, random_state, label="expression")


def _shared_embedding_obs_transport(
    x_obs: np.ndarray,
    x_tr: np.ndarray,
    random_state: int,
) -> tuple[np.ndarray, str]:
    z = np.vstack([x_obs, x_tr])
    return _embedding_for_umap(z, random_state=random_state)


def _plot_umap_marginal(
    emb: np.ndarray,
    labels: np.ndarray,
    title: str,
    y_note: str,
    out_path: Path,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    labs = pd.Series(labels).astype(str)
    for cat in ("normal", "intermediate", "tumor"):
        m = labs == cat
        if not m.any():
            continue
        color = MARGINAL_COLORS.get(cat, "#bcbd22")
        ax.scatter(emb[m, 0], emb[m, 1], s=12, alpha=0.85, c=color, label=cat, edgecolors="none")
    other = ~labs.isin(list(MARGINAL_COLORS.keys()))
    if other.any():
        ax.scatter(emb[other, 0], emb[other, 1], s=10, alpha=0.6, c="#cccccc", label="other")
    ax.set_xlabel("UMAP 1" if "UMAP" in y_note else "Component 1")
    ax.set_ylabel("UMAP 2" if "UMAP" in y_note else "Component 2")
    ax.set_title(
        f"{title}\n{y_note}\nAxes: UMAP dimensions (validation). Legend: marginal_label "
        f"(weak supervision; not used inside transport)."
    )
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(title="marginal_label", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
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
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=values, s=12, alpha=0.88, cmap="viridis")
    ax.set_xlabel("UMAP 1" if "UMAP" in y_note else "Component 1")
    ax.set_ylabel("UMAP 2" if "UMAP" in y_note else "Component 2")
    ax.set_title(
        f"{title}\n{y_note}\nColormap: viridis (perceptually uniform). "
        f"Variable: {cbar_label} (spot-level perturbation summary)."
    )
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    fig.colorbar(sc, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_bridge_malignant_trajectories(
    emb_obs: np.ndarray,
    emb_tr: np.ndarray,
    malignant_mask: np.ndarray,
    cna_scores: np.ndarray | None,
    out_path: Path,
    *,
    random_state: int,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(random_state)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=150)
    ax.scatter(emb_obs[:, 0], emb_obs[:, 1], s=3, alpha=0.15, c="#b0b0b0", label="all spots")

    idx = np.flatnonzero(malignant_mask)
    if idx.size == 0:
        idx = np.arange(min(emb_obs.shape[0], 300), dtype=int)
    max_arrows = min(500, idx.size)
    if idx.size > max_arrows:
        idx = rng.choice(idx, size=max_arrows, replace=False)

    dx = emb_tr[idx, 0] - emb_obs[idx, 0]
    dy = emb_tr[idx, 1] - emb_obs[idx, 1]
    if cna_scores is not None and cna_scores.shape[0] == emb_obs.shape[0]:
        cvals = cna_scores[idx]
        q = ax.quiver(
            emb_obs[idx, 0],
            emb_obs[idx, 1],
            dx,
            dy,
            cvals,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.0022,
            alpha=0.88,
            cmap="magma",
        )
        fig.colorbar(q, ax=ax, label="cna_score (high-malignancy subset)")
    else:
        ax.quiver(
            emb_obs[idx, 0],
            emb_obs[idx, 1],
            dx,
            dy,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.0022,
            alpha=0.88,
            color="#d62728",
        )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(
        "Stage 4 — bridge trajectory vectors for high-malignancy spots\n"
        "Arrows start at observed coordinates and point to transported coordinates."
    )
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_delta_by_malignancy_quantile(
    cna_scores: np.ndarray,
    delta_toward_ref: np.ndarray,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cna = np.asarray(cna_scores, dtype=float)
    delta = np.asarray(delta_toward_ref, dtype=float)
    finite = np.isfinite(cna) & np.isfinite(delta)
    cna = cna[finite]
    delta = delta[finite]
    if cna.size == 0:
        return

    qs = np.quantile(cna, [0.2, 0.4, 0.6, 0.8])
    bins = np.digitize(cna, qs, right=True)
    groups = [delta[bins == i] for i in range(5)]
    labels = ["Q1 (low CNA)", "Q2", "Q3", "Q4", "Q5 (high CNA)"]

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150)
    bp = ax.boxplot(groups, labels=labels, patch_artist=True, showfliers=False)
    cmap = plt.colormaps["RdYlBu_r"]
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(cmap(i / 4))
        patch.set_alpha(0.75)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_ylabel("distance_toward_reference_delta")
    ax.set_title(
        "Stage 4 — movement toward normal reference by pre-bridge CNA quantile\n"
        "Positive values indicate reduced distance to the normal reference."
    )
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_stage5_umap_figures(
    expr: np.ndarray,
    transported: np.ndarray,
    perturbation: np.ndarray,
    labels: np.ndarray,
    fig_dir: Path,
    logger: logging.Logger,
    *,
    random_state: int = UMAP_RANDOM_STATE_DEFAULT,
    normal_reference: np.ndarray | None = None,
) -> list[str]:
    """
    Required Stage 5 UMAP panels: observed (marginal), transported (shared embedding), perturbation_norm.
    """
    x_obs = _prepare_expression_for_embedding(expr)
    x_tr = _prepare_expression_for_embedding(transported)
    emb_full, note = _shared_embedding_obs_transport(x_obs, x_tr, random_state=random_state)
    n = x_obs.shape[0]
    emb_obs = emb_full[:n]
    emb_tr = emb_full[n:]
    xlim = float(emb_full[:, 0].min() - 0.5), float(emb_full[:, 0].max() + 0.5)
    ylim = float(emb_full[:, 1].min() - 0.5), float(emb_full[:, 1].max() + 0.5)

    pnorm = np.linalg.norm(perturbation, axis=1)
    paths: list[str] = []

    p1 = Path(fig_dir) / "stage_5_umap_observed_marginal_labels.png"
    _plot_umap_marginal(
        emb_obs,
        labels,
        "Stage 5 — observed expression (shared UMAP with transported states)",
        note,
        p1,
        xlim=xlim,
        ylim=ylim,
    )
    paths.append(str(p1.resolve()))
    logger.info("UMAP artifact: %s", p1)

    p2 = Path(fig_dir) / "stage_5_umap_transported_marginal_labels.png"
    _plot_umap_marginal(
        emb_tr,
        labels,
        "Stage 5 — transported state (same UMAP embedding as observed)",
        note,
        p2,
        xlim=xlim,
        ylim=ylim,
    )
    paths.append(str(p2.resolve()))
    logger.info("UMAP artifact: %s", p2)

    p3 = Path(fig_dir) / "stage_5_umap_perturbation_norm.png"
    _plot_umap_continuous(
        emb_obs,
        pnorm,
        "Stage 5 — L2 perturbation norm (observed positions in shared UMAP)",
        note,
        p3,
        cbar_label="perturbation_norm",
        xlim=xlim,
        ylim=ylim,
    )
    paths.append(str(p3.resolve()))
    logger.info("UMAP artifact: %s", p3)

    if normal_reference is not None:
        delta_toward_ref = (
            per_spot_l2_distance_to_reference(expr, normal_reference)
            - per_spot_l2_distance_to_reference(transported, normal_reference)
        )
        p4 = Path(fig_dir) / "stage_5_umap_delta_distance_toward_reference.png"
        _plot_umap_continuous(
            emb_obs,
            delta_toward_ref,
            "Stage 5 — movement toward normal reference",
            note,
            p4,
            cbar_label="distance_toward_reference_delta",
            xlim=xlim,
            ylim=ylim,
        )
        paths.append(str(p4.resolve()))
        logger.info("UMAP artifact: %s", p4)

    return paths


def write_stage4_umap_figures(
    expr: np.ndarray,
    transported: np.ndarray,
    labels: np.ndarray,
    normal_reference: np.ndarray,
    fig_dir: Path,
    logger: logging.Logger,
    *,
    cna_scores: np.ndarray | None = None,
    random_state: int = UMAP_RANDOM_STATE_DEFAULT,
) -> list[str]:
    """
    Stage 4 transport figures on a shared embedding: observed, transported, and movement toward reference.
    """
    x_obs = _prepare_expression_for_embedding(expr)
    x_tr = _prepare_expression_for_embedding(transported)
    emb_full, note = _shared_embedding_obs_transport(x_obs, x_tr, random_state=random_state)
    n = x_obs.shape[0]
    emb_obs = emb_full[:n]
    emb_tr = emb_full[n:]
    xlim = float(emb_full[:, 0].min() - 0.5), float(emb_full[:, 0].max() + 0.5)
    ylim = float(emb_full[:, 1].min() - 0.5), float(emb_full[:, 1].max() + 0.5)
    delta_toward_ref = (
        per_spot_l2_distance_to_reference(expr, normal_reference)
        - per_spot_l2_distance_to_reference(transported, normal_reference)
    )

    paths: list[str] = []

    p1 = Path(fig_dir) / "stage_4_umap_observed_marginal_labels.png"
    _plot_umap_marginal(
        emb_obs,
        labels,
        "Stage 4 — observed expression (shared embedding with transported states)",
        note,
        p1,
        xlim=xlim,
        ylim=ylim,
    )
    paths.append(str(p1.resolve()))
    logger.info("UMAP artifact: %s", p1)

    p2 = Path(fig_dir) / "stage_4_umap_transported_marginal_labels.png"
    _plot_umap_marginal(
        emb_tr,
        labels,
        "Stage 4 — transported state (same embedding as observed)",
        note,
        p2,
        xlim=xlim,
        ylim=ylim,
    )
    paths.append(str(p2.resolve()))
    logger.info("UMAP artifact: %s", p2)

    p3 = Path(fig_dir) / "stage_4_umap_delta_distance_toward_reference.png"
    _plot_umap_continuous(
        emb_obs,
        delta_toward_ref,
        "Stage 4 — movement toward normal reference",
        note,
        p3,
        cbar_label="distance_toward_reference_delta",
        xlim=xlim,
        ylim=ylim,
    )
    paths.append(str(p3.resolve()))
    logger.info("UMAP artifact: %s", p3)

    if cna_scores is not None and len(cna_scores) == n:
        cna = np.asarray(cna_scores, dtype=float)
        p4 = Path(fig_dir) / "stage_4_umap_cna_score_observed.png"
        _plot_umap_continuous(
            emb_obs,
            cna,
            "Stage 4 — pre-bridge CNA scores (observed state)",
            note,
            p4,
            cbar_label="cna_score",
            xlim=xlim,
            ylim=ylim,
        )
        paths.append(str(p4.resolve()))
        logger.info("UMAP artifact: %s", p4)

        p5 = Path(fig_dir) / "stage_4_umap_cna_score_transported.png"
        _plot_umap_continuous(
            emb_tr,
            cna,
            "Stage 4 — same CNA scores on transported state coordinates",
            note,
            p5,
            cbar_label="cna_score",
            xlim=xlim,
            ylim=ylim,
        )
        paths.append(str(p5.resolve()))
        logger.info("UMAP artifact: %s", p5)

        hi_thr = float(np.nanquantile(cna[np.isfinite(cna)], 0.8))
        malignant_mask = np.isfinite(cna) & (cna >= hi_thr)
        p6 = Path(fig_dir) / "stage_4_bridge_malignant_trajectories.png"
        _plot_bridge_malignant_trajectories(
            emb_obs,
            emb_tr,
            malignant_mask,
            cna,
            p6,
            random_state=random_state + 77,
            xlim=xlim,
            ylim=ylim,
        )
        paths.append(str(p6.resolve()))
        logger.info("UMAP artifact: %s", p6)

        p7 = Path(fig_dir) / "stage_4_bridge_delta_by_cna_quantile.png"
        _plot_delta_by_malignancy_quantile(cna, delta_toward_ref, p7)
        if p7.is_file():
            paths.append(str(p7.resolve()))
            logger.info("UMAP artifact: %s", p7)

    return paths


def write_stage5_summary_figures(
    obs: pd.DataFrame,
    perturbation_norm: np.ndarray,
    fig_dir: Path,
    logger: logging.Logger,
) -> list[str]:
    """
    Stage 5 summary figures for spatial perturbation burden and marginal-level burden stratification.
    """
    import matplotlib.pyplot as plt

    fig_dir = Path(fig_dir).expanduser().resolve()
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    x_col = "x" if "x" in obs.columns else None
    y_col = "y" if "y" in obs.columns else None
    if x_col is not None and y_col is not None:
        p1 = fig_dir / "stage_5_spatial_perturbation_map.png"
        fig1, ax1 = plt.subplots(figsize=(6, 5), dpi=150)
        sc = ax1.scatter(obs[x_col], obs[y_col], c=perturbation_norm, s=8, cmap="viridis")
        ax1.set_title("Stage 5 — spatial perturbation burden")
        ax1.set_xlabel(x_col)
        ax1.set_ylabel(y_col)
        fig1.colorbar(sc, ax=ax1, label="perturbation_norm")
        fig1.tight_layout()
        fig1.savefig(p1, dpi=300, bbox_inches="tight")
        plt.close(fig1)
        paths.append(str(p1.resolve()))
        logger.info("Stage 5 summary figure: %s", p1)

    if "marginal_label" in obs.columns and len(obs) == len(perturbation_norm):
        p2 = fig_dir / "stage_5_perturbation_norm_by_marginal_label.png"
        fig2, ax2 = plt.subplots(figsize=(6.5, 4.5), dpi=150)
        labels = obs["marginal_label"].astype(str)
        order = [lab for lab in ("tumor", "intermediate", "normal") if (labels == lab).any()]
        data = [perturbation_norm[labels.to_numpy() == lab] for lab in order]
        bp = ax2.boxplot(data, labels=order, patch_artist=True)
        for patch, lab in zip(bp["boxes"], order):
            patch.set_facecolor(MARGINAL_COLORS.get(lab, "#cccccc"))
            patch.set_alpha(0.7)
        ax2.set_title("Stage 5 — perturbation burden by marginal label")
        ax2.set_ylabel("perturbation_norm")
        fig2.tight_layout()
        fig2.savefig(p2, dpi=300, bbox_inches="tight")
        plt.close(fig2)
        paths.append(str(p2.resolve()))
        logger.info("Stage 5 summary figure: %s", p2)

    return paths


def stage5_handoff_for_downstream(output_dir: str | Path) -> dict[str, Any]:
    """Static + path documentation for Stages 6–7 (paths are under the pipeline or artifact ``output_dir``)."""
    od = str(Path(output_dir).expanduser().resolve())
    return {
        "perturbation_matrix_paths": {
            "npz_bundle": f"{od}/stage_5_expression_bundle.npz (arrays: observed, transported, perturbation)",
            "csv_wide": f"{od}/perturbation_matrix.csv",
            "legacy_alias": "perturbation_vectors.csv (same matrix as perturbation_matrix.csv in pipeline runs)",
        },
        "transported_state_paths": {
            "csv_wide": f"{od}/transported_expression.csv",
            "npz": f"{od}/stage_5_expression_bundle.npz key 'transported'",
        },
        "annotated_outputs": {
            "spot_summary": f"{od}/stage_5_spot_summary.csv (perturbation_norm + marginal_label when present)",
            "annotated_output_obs_csv": f"{od}/annotated_output_obs.csv (from run_pipeline, same directory)",
            "annotated_output_h5ad": f"{od}/annotated_output.h5ad (from run_pipeline when anndata is available)",
        },
        "row_alignment": (
            "Row i of perturbation_matrix.csv / bundle perturbation aligns with row i of bundle.obs "
            "at extraction time (same order as observed expression). Gene j aligns with var_names[j]."
        ),
        "spots_included": (
            "Perturbations are computed for **all spots**. Use obs['marginal_label'] "
            "(tumor / normal / intermediate) to restrict analyses; do not inject evaluation-only columns into transport."
        ),
        "safe_summary_columns_for_reporting": [
            "perturbation_norm",
            "marginal_label",
            "section_id",
            "cna_score",
            "malignancy_score",
        ],
    }


def write_stage5_artifact_manifest(
    manifest_path: Path,
    out_dir: Path,
    *,
    perturbation_paths: dict[str, str],
    annotated_paths: dict[str, str],
    umap_paths: list[str],
    summary_figure_paths: list[str],
    test_script_path: str,
    known_limitations: list[str],
    extra: dict[str, Any] | None = None,
) -> None:
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    od = Path(out_dir).expanduser().resolve()
    body: dict[str, Any] = {
        "output_directory": str(od),
        "transported_state_artifact_paths": {
            "transported_expression_csv": perturbation_paths.get("transported_expression_csv", ""),
            "expression_bundle_npz": perturbation_paths.get("expression_bundle_npz", ""),
            "gene_names_json": perturbation_paths.get("gene_names_json", ""),
        },
        "perturbation_artifact_paths": {
            "perturbation_matrix_csv": perturbation_paths.get("perturbation_matrix_csv", ""),
            "expression_bundle_npz": perturbation_paths.get("expression_bundle_npz", ""),
        },
        "annotated_output_paths": annotated_paths,
        "umap_figure_paths": umap_paths,
        "summary_figure_paths": summary_figure_paths,
        "test_script_path": test_script_path,
        "known_limitations": known_limitations,
        "handoff_contract": stage5_handoff_for_downstream(od),
    }
    if extra:
        body.update(extra)
    manifest_path.write_text(json.dumps(body, indent=2), encoding="utf-8")
