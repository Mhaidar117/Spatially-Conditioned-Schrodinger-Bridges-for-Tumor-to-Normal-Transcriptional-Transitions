"""
Stage 6: Program discovery on perturbation vectors (NMF primary; PCA / ICA baselines).

Consumes Stage 5-style perturbation matrices: rows = spots, columns = genes (``var_names`` order).
Downstream handoff: ``stage_6_nmf_scores.csv`` rows align with spot order; loadings rows align with ``var_names``.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA, NMF, PCA
from sklearn.metrics import normalized_mutual_info_score

from .config import PipelineConfig

# visualization_standards.md (aligned with Stage 5)
_MARGINAL_COLORS: dict[str, str] = {
    "tumor": "#d62728",
    "normal": "#1f77b4",
    "intermediate": "#7f7f7f",
}


@dataclass
class ProgramResults:
    scores: pd.DataFrame
    loadings: pd.DataFrame
    model_selection: pd.DataFrame
    chosen_method: str


@dataclass
class ProgramDiscoveryResult:
    """Full Stage 6 outputs including baseline factorizations for comparison UMAPs."""

    nmf_scores: pd.DataFrame
    nmf_loadings: pd.DataFrame
    pca_scores: pd.DataFrame
    ica_scores: pd.DataFrame
    model_selection: pd.DataFrame
    chosen_method: str
    n_components: int
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _nonnegative(a: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(a, dtype=float), a_min=0.0, a_max=None)


def _effective_n_components(perturb: np.ndarray, chosen: int) -> int:
    max_rank = int(min(perturb.shape[0], perturb.shape[1]))
    return max(1, min(int(chosen), max_rank))


def _relative_frobenius_reconstruction(x: np.ndarray, w: np.ndarray, h: np.ndarray) -> float:
    rec = w @ h
    denom = float(np.linalg.norm(x))
    if denom < 1e-12:
        return float("nan")
    return float(np.linalg.norm(x - rec) / denom)


def _sparsity(arr: np.ndarray) -> float:
    return float(np.mean(np.abs(arr) < 1e-3))


def _nmf_stability_cosine(
    x_nonneg: np.ndarray,
    n_components: int,
    seed_a: int,
    seed_b: int,
    max_iter: int,
) -> dict[str, Any]:
    """Two NMF fits; greedy match components by cosine similarity of gene loadings (H rows)."""
    if n_components < 2 or x_nonneg.shape[0] < 5:
        return {"skipped": True, "reason": "insufficient rank or sample size"}

    def fit(seed: int) -> tuple[np.ndarray, np.ndarray]:
        m = NMF(
            n_components=n_components,
            init="nndsvda",
            random_state=seed,
            max_iter=max_iter,
        )
        w = m.fit_transform(x_nonneg)
        h = m.components_
        return w, h

    _, h_a = fit(seed_a)
    _, h_b = fit(seed_b)
    h_a_n = h_a / (np.linalg.norm(h_a, axis=1, keepdims=True) + 1e-12)
    h_b_n = h_b / (np.linalg.norm(h_b, axis=1, keepdims=True) + 1e-12)
    sim = h_a_n @ h_b_n.T
    used_b: set[int] = set()
    cosines: list[float] = []
    for i in range(n_components):
        row = sim[i].copy()
        for j in used_b:
            row[j] = -1.0
        j = int(np.argmax(row))
        used_b.add(j)
        cosines.append(float(np.clip(row[j], -1.0, 1.0)))
    return {
        "skipped": False,
        "mean_matched_loading_cosine": float(np.mean(cosines)),
        "per_component_cosine": cosines,
        "seed_a": seed_a,
        "seed_b": seed_b,
    }


def compare_factorizations(perturb: np.ndarray, cfg: PipelineConfig) -> pd.DataFrame:
    """Backward-compatible comparison table (fits NMF, PCA, ICA)."""
    res = run_program_discovery(perturb, [], cfg)
    return res.model_selection


def run_program_discovery(
    perturb: np.ndarray,
    var_names: list[str],
    cfg: PipelineConfig,
) -> ProgramDiscoveryResult:
    """
    Fit NMF on nonnegative-shifted perturbations; fit PCA and ICA on raw perturbations for baselines.

    ``var_names`` is only validated for length when non-empty (pipeline passes real names).
    """
    x = np.asarray(perturb, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"perturbation matrix must be 2D; got shape {x.shape}")
    if x.size == 0:
        raise ValueError("perturbation matrix is empty")
    if var_names and len(var_names) != x.shape[1]:
        raise ValueError(f"len(var_names) ({len(var_names)}) != n_genes ({x.shape[1]})")

    warn: list[str] = []
    if not np.all(np.isfinite(x)):
        warn.append("non-finite values in perturbation matrix; replacing NaN/Inf with 0 for fitting")
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    n = _effective_n_components(x, cfg.programs.chosen_components)
    if n < cfg.programs.chosen_components:
        warn.append(
            f"chosen_components capped to min(n_spots, n_genes): using n_components={n} "
            f"(requested {cfg.programs.chosen_components})"
        )

    x_nonneg = _nonnegative(x - x.min())
    max_iter = cfg.programs.nmf_max_iter

    nmf = NMF(n_components=n, init="nndsvda", random_state=cfg.programs.random_seed, max_iter=max_iter)
    w_nmf = nmf.fit_transform(x_nonneg)
    h_nmf = nmf.components_
    rec_nmf = _relative_frobenius_reconstruction(x_nonneg, w_nmf, h_nmf)
    nmf_iters = int(getattr(nmf, "n_iter_", 0) or 0)
    if nmf_iters >= max_iter:
        warn.append(f"NMF reached max_iter={max_iter} without convergence flag")

    pca = PCA(n_components=n, random_state=cfg.programs.random_seed)
    w_pca = pca.fit_transform(x)
    h_pca = pca.components_
    rec_pca = _relative_frobenius_reconstruction(x, w_pca, h_pca)

    w_ica = np.zeros((x.shape[0], n))
    h_ica = np.zeros((n, x.shape[1]))
    rec_ica = float("nan")
    ica_note = "ok"
    try:
        ica = FastICA(n_components=n, random_state=cfg.programs.random_seed, max_iter=max_iter)
        w_ica = ica.fit_transform(x)
        h_ica = ica.components_
        rec_ica = _relative_frobenius_reconstruction(x, w_ica, h_ica)
    except Exception as ex:  # noqa: BLE001
        ica_note = f"ICA failed: {ex}"
        warn.append(ica_note)

    stability = _nmf_stability_cosine(
        x_nonneg,
        n,
        cfg.programs.random_seed,
        cfg.programs.random_seed + 1000,
        max_iter,
    )

    colnames = [f"program_{i}" for i in range(n)]
    nmf_scores = pd.DataFrame(w_nmf, columns=colnames)
    genes = var_names if var_names else [f"gene_{j}" for j in range(x.shape[1])]
    nmf_loadings = pd.DataFrame(h_nmf.T, index=genes, columns=colnames)
    pca_scores = pd.DataFrame(w_pca, columns=colnames)
    ica_scores = pd.DataFrame(w_ica, columns=colnames)

    model_selection = pd.DataFrame(
        [
            {
                "method": "NMF",
                "target": "nonnegative_shifted_perturbation",
                "n_components": n,
                "reconstruction_error": rec_nmf,
                "interpretability_score": float(np.mean(h_nmf >= 0.0)),
                "sparsity_loadings": _sparsity(h_nmf),
                "n_iter": nmf_iters,
            },
            {
                "method": "PCA",
                "target": "raw_perturbation",
                "n_components": n,
                "reconstruction_error": rec_pca,
                "interpretability_score": 0.0,
                "sparsity_loadings": _sparsity(h_pca),
                "n_iter": int(getattr(pca, "n_iter_", 0) or 0),
            },
            {
                "method": "ICA",
                "target": "raw_perturbation",
                "n_components": n,
                "reconstruction_error": rec_ica,
                "interpretability_score": 0.0,
                "sparsity_loadings": _sparsity(h_ica),
                "n_iter": -1,
                "note": ica_note,
            },
        ]
    )

    diagnostics: dict[str, Any] = {
        "perturbation_shape": list(x.shape),
        "n_components": n,
        "nmf_reconstruction_relative_frobenius": rec_nmf,
        "pca_reconstruction_relative_frobenius": rec_pca,
        "ica_reconstruction_relative_frobenius": rec_ica,
        "nmf_stability": stability,
    }

    return ProgramDiscoveryResult(
        nmf_scores=nmf_scores,
        nmf_loadings=nmf_loadings,
        pca_scores=pca_scores,
        ica_scores=ica_scores,
        model_selection=model_selection,
        chosen_method="NMF",
        n_components=n,
        diagnostics=diagnostics,
        warnings=warn,
    )


def run_nmf_programs(
    perturb: np.ndarray,
    var_names: list[str],
    cfg: PipelineConfig,
) -> ProgramResults:
    """Pipeline entry: NMF scores/loadings plus method comparison table."""
    res = run_program_discovery(perturb, var_names, cfg)
    return ProgramResults(
        scores=res.nmf_scores,
        loadings=res.nmf_loadings,
        model_selection=res.model_selection,
        chosen_method=res.chosen_method,
    )


def top_genes_per_program(loadings: pd.DataFrame, top_k: int) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for col in loadings.columns:
        s = loadings[col].sort_values(ascending=False)
        out[str(col)] = [str(i) for i in s.index[:top_k].tolist()]
    return out


def infer_program_display_names(loadings: pd.DataFrame, top_k: int = 12) -> dict[str, str]:
    """
    Infer human-readable labels from top-loading genes.
    Falls back to ``program_x (GENE1/GENE2/GENE3)`` when no marker family matches.
    """
    marker_sets: list[tuple[str, set[str]]] = [
        ("Reactive_Glial", {"GFAP", "AQP4", "GJA1", "VIM", "CHI3L1", "CD44", "C3", "APOE"}),
        ("Myelin_Oligo", {"MBP", "PLP1", "MOBP", "MAG", "MOG", "CNP"}),
        ("Neuronal_Plasticity", {"STMN2", "GAP43", "SNAP25", "MAP1B", "UCHL1", "VSNL1", "RTN1"}),
        ("Mito_Metabolic", {"MTND1", "MTND2", "MTND3", "MTCO3", "NDUFC2", "ATP5IF1", "MTRNR2L12"}),
        ("Stromal_Matrix", {"COL1A1", "COL6A1", "FN1", "SPARC", "AQP1", "VWF", "AEBP1"}),
        ("Proliferative", {"MKI67", "TOP2A", "PCNA", "STMN1", "UBE2C", "CENPF"}),
    ]

    def _norm(g: str) -> str:
        return "".join(ch for ch in str(g).upper() if ch.isalnum())

    out: dict[str, str] = {}
    top = top_genes_per_program(loadings, top_k=top_k)
    for col, genes in top.items():
        gene_norm = {_norm(g) for g in genes}
        best_label = ""
        best_hits = -1
        for label, markers in marker_sets:
            hits = sum(1 for m in markers if m in gene_norm)
            if hits > best_hits:
                best_hits = hits
                best_label = label
        anchor = [g for g in genes[:3] if g]
        if best_hits <= 0:
            out[col] = f"{col} ({'/'.join(anchor)})"
        else:
            out[col] = f"{best_label} ({'/'.join(anchor)})"
    return out


def summarize_programs_by_column(
    scores: pd.DataFrame,
    obs: pd.DataFrame,
    column: str,
) -> pd.DataFrame | None:
    if column not in obs.columns or len(scores) != len(obs):
        return None
    g = scores.copy()
    g[column] = obs[column].astype(str).values
    return g.groupby(column, dropna=False).mean()


def _program_annotation_associations(
    scores: pd.DataFrame,
    obs: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Program/annotation agreement using dominant-program NMI against held-out labels."""
    if len(scores) != len(obs):
        return pd.DataFrame(columns=["annotation", "n_spots", "n_unique_labels", "dominant_program_nmi"])
    dominant = np.argmax(scores.to_numpy(dtype=float), axis=1)
    rows: list[dict[str, Any]] = []
    for col in columns:
        if col not in obs.columns:
            continue
        lab = obs[col].astype(str).fillna("NA").to_numpy()
        valid = lab != "nan"
        if int(valid.sum()) < 5:
            continue
        rows.append(
            {
                "annotation": col,
                "n_spots": int(valid.sum()),
                "n_unique_labels": int(pd.Series(lab[valid]).nunique()),
                "dominant_program_nmi": float(normalized_mutual_info_score(lab[valid], dominant[valid])),
            }
        )
    return pd.DataFrame(rows)


def _program_spatial_localization(scores: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    if len(scores) != len(obs) or "x" not in obs.columns or "y" not in obs.columns:
        return pd.DataFrame(columns=["program", "corr_x", "corr_y", "spatial_localization_score"])
    x = pd.to_numeric(obs["x"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(obs["y"], errors="coerce").to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for col in scores.columns:
        s = pd.to_numeric(scores[col], errors="coerce").to_numpy(dtype=float)
        mx = np.isfinite(x) & np.isfinite(s)
        my = np.isfinite(y) & np.isfinite(s)
        cx = float(np.corrcoef(x[mx], s[mx])[0, 1]) if int(mx.sum()) >= 4 else float("nan")
        cy = float(np.corrcoef(y[my], s[my])[0, 1]) if int(my.sum()) >= 4 else float("nan")
        rows.append(
            {
                "program": str(col),
                "corr_x": cx,
                "corr_y": cy,
                "spatial_localization_score": float(np.nanmax(np.abs([cx, cy]))),
            }
        )
    return pd.DataFrame(rows)


def save_program_artifacts(
    out_dir: str | Path,
    discovery: ProgramDiscoveryResult,
    *,
    obs: pd.DataFrame | None = None,
    spot_id_column: str = "spot_id",
) -> dict[str, str]:
    """Write tabular Stage 6 artifacts; returns absolute path strings."""
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    scores = discovery.nmf_scores.copy()
    if obs is not None and spot_id_column in obs.columns:
        scores.index = obs[spot_id_column].astype(str).values
    else:
        scores.index = [f"spot_{i}" for i in range(len(scores))]
    scores.index.name = "spot_id"

    p_scores = out / "stage_6_nmf_scores.csv"
    scores.to_csv(p_scores)
    paths["nmf_scores_csv"] = str(p_scores.resolve())

    p_load = out / "stage_6_nmf_loadings.csv"
    discovery.nmf_loadings.to_csv(p_load)
    paths["nmf_loadings_csv"] = str(p_load.resolve())

    name_map = infer_program_display_names(discovery.nmf_loadings, top_k=12)
    p_names = out / "stage_6_program_names.csv"
    pd.DataFrame(
        [{"program_id": pid, "program_name": pname} for pid, pname in name_map.items()]
    ).to_csv(p_names, index=False)
    paths["program_names_csv"] = str(p_names.resolve())

    p_cmp = out / "stage_6_factorization_comparison.csv"
    discovery.model_selection.to_csv(p_cmp, index=False)
    paths["factorization_comparison_csv"] = str(p_cmp.resolve())

    p_pca = out / "stage_6_pca_scores.csv"
    discovery.pca_scores.to_csv(p_pca, index=False)
    paths["pca_scores_csv"] = str(p_pca.resolve())

    p_ica = out / "stage_6_ica_scores.csv"
    discovery.ica_scores.to_csv(p_ica, index=False)
    paths["ica_scores_csv"] = str(p_ica.resolve())

    if obs is not None:
        strat = summarize_programs_by_column(discovery.nmf_scores, obs, "marginal_label")
        if strat is not None:
            p_st = out / "stage_6_stratified_nmf_means_by_marginal_label.csv"
            strat.to_csv(p_st)
            paths["stratified_marginal_csv"] = str(p_st.resolve())

        assoc = _program_annotation_associations(discovery.nmf_scores, obs, ["mp", "layer", "ivygap", "org1", "org2"])
        if not assoc.empty:
            p_assoc = out / "stage_6_program_annotation_associations.csv"
            assoc.to_csv(p_assoc, index=False)
            paths["program_annotation_associations_csv"] = str(p_assoc.resolve())

        spatial_loc = _program_spatial_localization(discovery.nmf_scores, obs)
        if not spatial_loc.empty:
            p_spatial = out / "stage_6_program_spatial_localization.csv"
            spatial_loc.to_csv(p_spatial, index=False)
            paths["program_spatial_localization_csv"] = str(p_spatial.resolve())

    diag_path = out / "stage_6_program_diagnostics.json"
    diag_path.write_text(json.dumps(discovery.diagnostics, indent=2, default=str), encoding="utf-8")
    paths["diagnostics_json"] = str(diag_path.resolve())

    return paths


def _embedding_for_umap(X: np.ndarray, random_state: int) -> tuple[np.ndarray, str]:
    from .utils import umap_or_pca_2d

    return umap_or_pca_2d(X, random_state, label="perturbation vectors")


def write_stage6_umap_figures(
    perturb: np.ndarray,
    discovery: ProgramDiscoveryResult,
    fig_dir: str | Path,
    logger: logging.Logger,
    cfg: PipelineConfig,
    *,
    obs: pd.DataFrame | None = None,
) -> list[str]:
    """
    Required views: dominant NMF program; panels of NMF program scores; PCA partition comparison.
    Embedding is fit once on raw perturbation vectors (shared across panels).
    """
    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    fig_dir = Path(fig_dir).expanduser().resolve()
    fig_dir.mkdir(parents=True, exist_ok=True)
    dpi = cfg.programs.figure_dpi
    rs = cfg.programs.umap_random_state

    x = np.asarray(perturb, dtype=float)
    emb, basis_note = _embedding_for_umap(x, random_state=rs)
    logger.info("Stage 6 perturbation embedding basis: %s", basis_note)
    paths: list[str] = []
    display_names = infer_program_display_names(discovery.nmf_loadings, top_k=12)

    n = discovery.n_components
    w_nmf = discovery.nmf_scores.to_numpy()
    dominant = np.argmax(w_nmf, axis=1)
    labels = np.array(
        [display_names.get(f"program_{int(j)}", f"program_{int(j)}") for j in dominant],
        dtype=object,
    )

    # 1) Dominant NMF program
    fig1, ax1 = plt.subplots(figsize=(7.0, 6.0), layout="constrained")
    uniq = sorted(set(labels.tolist()))
    tab_cmap = colormaps["tab10"]
    denom = max(len(uniq) - 1, 1)
    for i, u in enumerate(uniq):
        m = labels == u
        ax1.scatter(
            emb[m, 0],
            emb[m, 1],
            s=12,
            alpha=0.88,
            color=tab_cmap(i / denom),
            label=u,
        )
    ax1.set_xlabel("UMAP 1" if "UMAP" in basis_note else "Component 1")
    ax1.set_ylabel("UMAP 2" if "UMAP" in basis_note else "Component 2")
    ax1.set_title(
        "Stage 6 — dominant NMF program (perturbation space)\n"
        f"{basis_note}\n"
        "Legend: argmax NMF score per spot; axes = embedding of perturbation vectors."
    )
    ax1.legend(title="dominant NMF", loc="best", fontsize=8)
    p1 = fig_dir / "stage_6_umap_nmf_dominant_program.png"
    fig1.savefig(p1, dpi=dpi)
    plt.close(fig1)
    paths.append(str(p1.resolve()))
    logger.info("UMAP artifact: %s", p1)

    # 2) NMF score panels (up to 4 components)
    n_panel = min(4, n)
    if n_panel > 0:
        fig2, axes = plt.subplots(2, 2, figsize=(9.5, 8.5), layout="constrained")
        axes_flat = axes.ravel()
        for k in range(4):
            ax = axes_flat[k]
            if k < n_panel:
                vals = w_nmf[:, k]
                sc2 = ax.scatter(emb[:, 0], emb[:, 1], c=vals, cmap="viridis", s=10, alpha=0.9)
                fig2.colorbar(sc2, ax=ax, label="NMF score")
                pid = str(discovery.nmf_scores.columns[k])
                ax.set_title(f"NMF {display_names.get(pid, pid)}")
            else:
                ax.axis("off")
            ax.set_xlabel("UMAP 1" if "UMAP" in basis_note else "C1")
            ax.set_ylabel("UMAP 2" if "UMAP" in basis_note else "C2")
        fig2.suptitle(
            "Stage 6 — NMF program scores on shared perturbation UMAP\n"
            f"{basis_note}\n"
            "Continuous color = NMF activation; same embedding for all panels."
        )
        p2 = fig_dir / "stage_6_umap_nmf_program_score_panels.png"
        fig2.savefig(p2, dpi=dpi)
        plt.close(fig2)
        paths.append(str(p2.resolve()))
        logger.info("UMAP artifact: %s", p2)

    # 3) PCA dominant axis (partition of perturbation landscape)
    w_pca = discovery.pca_scores.to_numpy()
    p_dom = np.argmax(np.abs(w_pca), axis=1)
    p_labels = np.array([f"PCA_axis_{int(j)}" for j in p_dom], dtype=object)
    fig3, ax3 = plt.subplots(figsize=(7.0, 6.0), layout="constrained")
    uniq_p = sorted(set(p_labels.tolist()))
    set3 = colormaps["Set3"]
    denp = max(len(uniq_p) - 1, 1)
    for i, u in enumerate(uniq_p):
        m = p_labels == u
        ax3.scatter(
            emb[m, 0],
            emb[m, 1],
            s=12,
            alpha=0.88,
            color=set3(i / denp),
            label=u,
        )
    ax3.set_xlabel("UMAP 1" if "UMAP" in basis_note else "Component 1")
    ax3.set_ylabel("UMAP 2" if "UMAP" in basis_note else "Component 2")
    ax3.set_title(
        "Stage 6 — PCA partition (same embedding as NMF figures)\n"
        f"{basis_note}\n"
        "Legend: axis with largest |PCA score| per spot; comparison baseline to NMF dominant plot."
    )
    ax3.legend(title="PCA partition", loc="best", fontsize=7)
    p3 = fig_dir / "stage_6_umap_pca_dominant_partition.png"
    fig3.savefig(p3, dpi=dpi)
    plt.close(fig3)
    paths.append(str(p3.resolve()))
    logger.info("UMAP artifact: %s", p3)

    # 4) ICA partition when ICA scores are non-degenerate
    w_ica = discovery.ica_scores.to_numpy()
    if np.isfinite(w_ica).all() and float(np.std(w_ica)) > 1e-8:
        i_dom = np.argmax(np.abs(w_ica), axis=1)
        i_labels = np.array([f"ICA_axis_{int(j)}" for j in i_dom], dtype=object)
        fig5, ax5 = plt.subplots(figsize=(7.0, 6.0), layout="constrained")
        uniq_i = sorted(set(i_labels.tolist()))
        accent = colormaps["Accent"]
        deni = max(len(uniq_i) - 1, 1)
        for ii, u in enumerate(uniq_i):
            m = i_labels == u
            ax5.scatter(
                emb[m, 0],
                emb[m, 1],
                s=12,
                alpha=0.88,
                color=accent(ii / deni),
                label=u,
            )
        ax5.set_xlabel("UMAP 1" if "UMAP" in basis_note else "Component 1")
        ax5.set_ylabel("UMAP 2" if "UMAP" in basis_note else "Component 2")
        ax5.set_title(
            "Stage 6 — ICA partition (same perturbation embedding)\n"
            f"{basis_note}\n"
            "Legend: axis with largest |ICA score| per spot; alternate baseline to NMF."
        )
        ax5.legend(title="ICA partition", loc="best", fontsize=7)
        p5 = fig_dir / "stage_6_umap_ica_dominant_partition.png"
        fig5.savefig(p5, dpi=dpi)
        plt.close(fig5)
        paths.append(str(p5.resolve()))
        logger.info("UMAP artifact: %s", p5)

    # Optional: marginal_label on same embedding (evaluation / interpretation)
    if obs is not None and "marginal_label" in obs.columns and len(obs) == len(emb):
        fig4, ax4 = plt.subplots(figsize=(7.0, 6.0), layout="constrained")
        labs = obs["marginal_label"].astype(str).to_numpy()
        colors = [_MARGINAL_COLORS.get(str(lab), "#bcbd22") for lab in labs]
        ax4.scatter(emb[:, 0], emb[:, 1], c=colors, s=12, alpha=0.88)
        ax4.set_xlabel("UMAP 1" if "UMAP" in basis_note else "Component 1")
        ax4.set_ylabel("UMAP 2" if "UMAP" in basis_note else "Component 2")
        ax4.set_title(
            "Stage 6 — marginal_label on perturbation UMAP (evaluation / interpretation only)\n"
            f"{basis_note}"
        )
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=_MARGINAL_COLORS[k], markersize=8, label=k)
            for k in ("tumor", "normal", "intermediate")
            if k in set(labs.tolist())
        ]
        if handles:
            ax4.legend(handles=handles, title="marginal_label", loc="best")
        p4 = fig_dir / "stage_6_umap_marginal_labels_on_perturbation.png"
        fig4.savefig(p4, dpi=dpi)
        plt.close(fig4)
        paths.append(str(p4.resolve()))
        logger.info("UMAP artifact: %s", p4)

    return paths


def write_stage6_summary_figures(
    discovery: ProgramDiscoveryResult,
    fig_dir: str | Path,
    logger: logging.Logger,
    *,
    obs: pd.DataFrame | None = None,
    top_genes: int = 12,
) -> list[str]:
    """
    Stage 6 summary figures for manuscript-facing program interpretation beyond UMAP panels.
    """
    import matplotlib.pyplot as plt

    fig_dir = Path(fig_dir).expanduser().resolve()
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    loadings = discovery.nmf_loadings.copy()
    prog_cols = [c for c in loadings.columns if str(c).startswith("program_")]
    display_names = infer_program_display_names(loadings, top_k=12)
    if prog_cols:
        selected_rows: list[str] = []
        for col in prog_cols:
            selected_rows.extend(loadings[col].sort_values(ascending=False).head(top_genes).index.astype(str).tolist())
        selected_rows = list(dict.fromkeys(selected_rows))
        sub = loadings.loc[selected_rows, prog_cols]
        p1 = fig_dir / "stage_6_program_loadings_heatmap.png"
        fig1, ax1 = plt.subplots(figsize=(1.6 * len(prog_cols) + 2.5, max(5.0, 0.24 * len(selected_rows) + 2.5)), dpi=150)
        im = ax1.imshow(sub.to_numpy(), aspect="auto", cmap="magma")
        ax1.set_xticks(np.arange(len(prog_cols)))
        ax1.set_xticklabels([display_names.get(str(c), str(c)) for c in prog_cols], rotation=45, ha="right")
        ax1.set_yticks(np.arange(len(selected_rows)))
        ax1.set_yticklabels(selected_rows, fontsize=8)
        ax1.set_title("Stage 6 — top program loadings heatmap")
        fig1.colorbar(im, ax=ax1, label="gene weight")
        fig1.tight_layout()
        fig1.savefig(p1, dpi=300, bbox_inches="tight")
        plt.close(fig1)
        paths.append(str(p1.resolve()))
        logger.info("Stage 6 summary figure: %s", p1)

    if prog_cols and obs is not None and len(obs) == len(discovery.nmf_scores):
        strat = summarize_programs_by_column(discovery.nmf_scores, obs, "marginal_label")
        if strat is not None and not strat.empty:
            p2 = fig_dir / "stage_6_program_composition_by_marginal_label.png"
            fig2, ax2 = plt.subplots(figsize=(8.5, 4.8), dpi=150)
            strat.loc[:, prog_cols].plot(kind="bar", stacked=True, ax=ax2, colormap="tab20")
            ax2.set_title("Stage 6 — mean NMF program composition by marginal label")
            ax2.set_xlabel("marginal_label")
            ax2.set_ylabel("mean program score")
            handles, labels = ax2.get_legend_handles_labels()
            ax2.legend(
                handles,
                [display_names.get(str(lbl), str(lbl)) for lbl in labels],
                title="program",
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                fontsize=8,
            )
            fig2.tight_layout()
            fig2.savefig(p2, dpi=300, bbox_inches="tight")
            plt.close(fig2)
            paths.append(str(p2.resolve()))
            logger.info("Stage 6 summary figure: %s", p2)

    return paths


def write_stage6_artifact_manifest(
    manifest_path: str | Path,
    *,
    artifact_paths: dict[str, str],
    umap_paths: list[str],
    test_script_path: str,
    known_limitations: list[str],
    extra: dict[str, Any] | None = None,
) -> None:
    """Machine-readable Stage 6 manifest for coordinator / Stage 7."""
    payload: dict[str, Any] = {
        "scores_artifact_path": artifact_paths.get("nmf_scores_csv", ""),
        "loadings_artifact_path": artifact_paths.get("nmf_loadings_csv", ""),
        "model_comparison_artifact_path": artifact_paths.get("factorization_comparison_csv", ""),
        "program_annotation_associations_path": artifact_paths.get("program_annotation_associations_csv", ""),
        "program_spatial_localization_path": artifact_paths.get("program_spatial_localization_csv", ""),
        "umap_figure_paths": list(umap_paths),
        "test_script_path": test_script_path,
        "known_limitations": list(known_limitations),
        "stage_7_handoff": stage7_handoff_contract(artifact_paths),
    }
    if extra:
        payload.update(extra)
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def stage7_handoff_contract(artifact_paths: dict[str, str]) -> dict[str, Any]:
    """Static schema summary for Stage 7 reporting (no factorization re-run needed)."""
    return {
        "nmf_scores": {
            "path_key": "nmf_scores_csv",
            "schema": "index: spot_id; columns: program_0 .. program_{K-1} (nonnegative NMF spot scores).",
        },
        "nmf_loadings": {
            "path_key": "nmf_loadings_csv",
            "schema": "index: gene symbol/name (aligned to Stage 5 var_names); columns: program_* (gene weights).",
        },
        "program_names": {
            "path_key": "program_names_csv",
            "schema": "rows map program_id -> inferred human-readable program_name based on top loading genes.",
        },
        "factorization_comparison": {
            "path_key": "factorization_comparison_csv",
            "schema": "rows: NMF, PCA, ICA; columns include method, target, n_components, reconstruction_error, sparsity_loadings.",
        },
        "baseline_score_matrices": {
            "pca_scores_csv": "columns program_* = PCA score dimensions (ordered by variance).",
            "ica_scores_csv": "columns program_* = ICA source axes (when ICA converges; may be zeros if fit failed).",
        },
        "method_names_in_comparison": ["NMF", "PCA", "ICA"],
        "safe_program_summaries_for_reports": [
            "Per-spot NMF scores and dominant program index.",
            "Top genes per program from loadings (post hoc).",
            "Stratified mean NMF scores by marginal_label (if obs provided).",
            "Reconstruction errors and sparsity from factorization_comparison.csv.",
        ],
        "convergence_warnings": "See stage_6_program_discovery.log and ProgramDiscoveryResult.warnings; NMF max_iter and ICA failures are logged.",
        "resolved_paths": artifact_paths,
    }

