from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import PipelineConfig
from .types import DatasetBundle

# Provenance codes for obs["malignancy_provenance"] and UMAPs (Stages 3–4 handoff).
# Downstream: if fallback_expression_program or inferred_chromosomal_cna, surface that training marginals are not true CNA.
PROV_CANONICAL_SCORE = "canonical_cna_score_column"
PROV_ALIAS_PRECOMPUTED = "precomputed_malignancy_alias"
PROV_INFERRED_CHROMOSOMAL = "inferred_chromosomal_cna_from_expression"
PROV_FALLBACK_EXPRESSION_PROGRAM = "fallback_expression_program_hvg_proxy"

# Do not treat evaluation-style or categorical bins as continuous malignancy (Stage 2 anti-leakage).
_SCORE_COLUMN_BLOCKLIST = frozenset(
    {
        "cna_bin",
        "layer",
        "ivygap",
        "org1",
        "org2",
        "mp",
    }
)


@dataclass
class CNAStageResult:
    bundle: DatasetBundle
    source: str
    warnings: list[str]
    summary: pd.DataFrame
    section_counts: pd.DataFrame
    malignancy_provenance: str = ""
    decision_path: list[str] = field(default_factory=list)
    umap_paths: list[str] = field(default_factory=list)


def find_existing_cna_score(obs: pd.DataFrame, cfg: PipelineConfig) -> tuple[pd.Series | None, str | None]:
    canonical = cfg.cna.canonical_column
    aliases = [canonical] + [a for a in cfg.cna.aliases if a != canonical]
    lowered = {str(c).lower(): c for c in obs.columns}
    for alias in aliases:
        if alias in obs.columns:
            num = pd.to_numeric(obs[alias], errors="coerce")
            if num.notna().sum() > 0:
                return num, alias
        if alias.lower() in lowered:
            col = lowered[alias.lower()]
            num = pd.to_numeric(obs[col], errors="coerce")
            if num.notna().sum() > 0:
                return num, str(col)
    if cfg.cna.require_true_score:
        return None, None
    for c in obs.columns:
        lc = str(c).lower()
        if lc in _SCORE_COLUMN_BLOCKLIST:
            continue
        if "cna" in lc or "malignan" in lc:
            num = pd.to_numeric(obs[c], errors="coerce")
            if num.notna().sum() == 0:
                continue
            return num, str(c)
    return None, None


def gene_annotation_configured(cfg: PipelineConfig) -> bool:
    path = Path(cfg.cna.gene_annotation_path).expanduser()
    return bool(cfg.cna.gene_annotation_path and path.is_file())


def _load_gene_annotation(cfg: PipelineConfig) -> pd.DataFrame:
    path = Path(cfg.cna.gene_annotation_path).expanduser()
    if not path.exists():
        raise ValueError(
            "CNA inference requires genomic annotation, but gene_annotation_path is missing or invalid. "
            "Provide a table with gene_id, chromosome, and genomic position."
        )
    sep = "\t" if path.suffix == ".tsv" else ","
    ann = pd.read_csv(path, sep=sep)
    needed = {cfg.cna.gene_id_column, cfg.cna.chromosome_column, cfg.cna.position_column}
    missing = [c for c in needed if c not in ann.columns]
    if missing:
        raise ValueError(
            f"CNA inference requires columns {sorted(needed)} in gene annotation, missing: {missing}."
        )
    ann = ann[[cfg.cna.gene_id_column, cfg.cna.chromosome_column, cfg.cna.position_column]].copy()
    ann.columns = ["gene_id", "chromosome", "position"]
    ann["gene_id"] = ann["gene_id"].astype(str)
    ann["chromosome"] = ann["chromosome"].astype(str).str.replace("^chr", "", regex=True)
    ann["position"] = pd.to_numeric(ann["position"], errors="coerce")
    ann = ann.dropna(subset=["position"])
    return ann


def _window_smooth(x: np.ndarray, window: int) -> np.ndarray:
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / window
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, x)


def _reference_profile(expr_sorted: np.ndarray, mapped: pd.DataFrame, cfg: PipelineConfig) -> np.ndarray:
    ref_path = Path(cfg.cna.reference_normal_path).expanduser() if cfg.cna.reference_normal_path else None
    if ref_path and ref_path.exists():
        sep = "\t" if ref_path.suffix == ".tsv" else ","
        ref = pd.read_csv(ref_path, sep=sep)
        if {"gene_id", "baseline_expr"}.issubset(ref.columns):
            ref = ref[["gene_id", "baseline_expr"]].copy()
            ref["gene_id"] = ref["gene_id"].astype(str)
            m = mapped[["gene_id"]].merge(ref, on="gene_id", how="left")
            vals = pd.to_numeric(m["baseline_expr"], errors="coerce").to_numpy()
            if np.isfinite(vals).sum() > 0:
                med = np.nanmedian(vals[np.isfinite(vals)])
                vals = np.where(np.isfinite(vals), vals, med)
                return vals
    return np.median(expr_sorted, axis=0)


def infer_cna_score_from_expression(bundle: DatasetBundle, cfg: PipelineConfig) -> tuple[np.ndarray, dict[str, float]]:
    ann = _load_gene_annotation(cfg)
    genes = pd.DataFrame({"gene_id": [str(g) for g in bundle.var_names], "idx": np.arange(len(bundle.var_names))})
    mapped = genes.merge(ann, on="gene_id", how="inner")
    if mapped.shape[0] < cfg.cna.min_mapped_genes:
        raise ValueError(
            f"CNA inference requires at least {cfg.cna.min_mapped_genes} mapped genes, found {mapped.shape[0]}."
        )

    mapped = mapped.sort_values(["chromosome", "position"]).reset_index(drop=True)
    expr_sorted = bundle.expr[:, mapped["idx"].to_numpy()]
    smooth = _window_smooth(expr_sorted, cfg.cna.smoothing_window)
    baseline = _reference_profile(smooth, mapped, cfg)
    deviations = smooth - baseline[None, :]

    raw_score = np.mean(np.abs(deviations), axis=1)
    order = np.argsort(raw_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, len(raw_score), endpoint=True)
    score = ranks
    return score, {"mapped_genes": float(mapped.shape[0]), "score_std": float(np.std(score))}


def infer_fallback_expression_program_score(bundle: DatasetBundle, cfg: PipelineConfig) -> tuple[np.ndarray, dict[str, float]]:
    """
    Notebook-style last resort: rank-normalized HVG activity proxy (not chromosomal CNA, not curated program banks).
    Mirrors the intent of data_exploration.ipynb's expression-derived malignancy when CNA assets are absent.
    """
    x = np.asarray(bundle.expr, dtype=float)
    n_spots, n_genes = x.shape
    if n_spots < 2 or n_genes < 10:
        raise ValueError("Expression-program fallback needs at least 2 spots and 10 genes.")
    lib = x.sum(axis=1, keepdims=True) + 1e-8
    x = np.log1p(x / lib * 1e4)
    gene_var = np.var(x, axis=0)
    k = max(10, min(int(cfg.cna.program_fallback_top_genes), n_genes))
    top_idx = np.argsort(-gene_var)[:k]
    sub = x[:, top_idx]
    mu = sub.mean(axis=0, keepdims=True)
    sig = sub.std(axis=0, keepdims=True) + 1e-8
    z = (sub - mu) / sig
    raw = np.mean(z, axis=1)
    order = np.argsort(raw)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, n_spots, endpoint=True)
    score = ranks
    return score, {"n_hvg": float(k), "score_std": float(np.std(score))}


def resolve_malignancy_scores(
    bundle: DatasetBundle, cfg: PipelineConfig
) -> tuple[pd.DataFrame, str, str, list[str], list[str]]:
    """
    Priority: (1) existing canonical/aliased malignancy columns, (2) chromosomal CNA-from-expression when annotation exists,
    (3) HVG expression-program proxy. Returns obs copy with cfg.cna.canonical_column filled, machine provenance key,
    human-readable source summary, warnings, and ordered decision_path for logging.
    """
    obs = bundle.obs.copy()
    warnings: list[str] = []
    path: list[str] = []

    score, source_col = find_existing_cna_score(obs, cfg)
    if score is not None:
        path.append(f"detected_existing_score:{source_col}")
        obs[cfg.cna.canonical_column] = score.to_numpy(dtype=float)
        if source_col == cfg.cna.canonical_column:
            prov = PROV_CANONICAL_SCORE
            src = f"provided:{source_col}"
        else:
            prov = PROV_ALIAS_PRECOMPUTED
            src = f"provided_alias:{source_col}"
        obs["malignancy_provenance"] = prov
        return obs, prov, src, warnings, path

    if cfg.cna.require_true_score:
        raise ValueError(
            f"True CNA score is required but missing/unusable. Expected numeric '{cfg.cna.canonical_column}' "
            f"(or configured aliases) joined during Stage 1."
        )

    path.append("no_existing_malignancy_column")
    if not cfg.cna.infer_if_missing:
        raise ValueError(
            f"No malignancy score found in input and cna.infer_if_missing=false. "
            f"Provide '{cfg.cna.canonical_column}' or enable inference."
        )

    if gene_annotation_configured(cfg):
        path.append("attempt_chromosomal_cna_inference")
        try:
            inferred, stats = infer_cna_score_from_expression(bundle, cfg)
            if float(np.std(inferred)) <= cfg.cna.constant_score_std_threshold:
                raise ValueError("Inferred chromosomal CNA score is nearly constant.")
            obs[cfg.cna.canonical_column] = inferred
            obs["malignancy_provenance"] = PROV_INFERRED_CHROMOSOMAL
            warnings.append(
                f"Inferred chromosomal CNA-style score from expression (not true DNA copy number); "
                f"mapped_genes={int(stats['mapped_genes'])}."
            )
            return obs, PROV_INFERRED_CHROMOSOMAL, "inferred_chromosomal_cna_from_expression", warnings, path
        except Exception as exc:
            warnings.append(f"Chromosomal CNA inference failed ({type(exc).__name__}: {exc}); trying expression-program fallback.")
            path.append("chromosomal_cna_inference_failed")

    path.append("expression_program_hvg_fallback")
    fb, stats = infer_fallback_expression_program_score(bundle, cfg)
    if float(np.std(fb)) <= cfg.cna.constant_score_std_threshold:
        raise ValueError(
            "Expression-program fallback score is nearly constant across spots and unusable for marginal construction."
        )
    obs[cfg.cna.canonical_column] = fb
    obs["malignancy_provenance"] = PROV_FALLBACK_EXPRESSION_PROGRAM
    warnings.append(
        f"Used HVG expression-program malignancy proxy (data_exploration-style fallback, not curated program banks); "
        f"n_hvg={int(stats['n_hvg'])}."
    )
    return obs, PROV_FALLBACK_EXPRESSION_PROGRAM, "fallback_expression_program_hvg_proxy", warnings, path


def assign_marginals_from_cna_score(obs: pd.DataFrame, cfg: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Within-section quantile cutoffs; deterministic section order; NaN scores -> intermediate with warning."""
    section_col = cfg.state.section_column
    obs = obs.copy()
    if section_col not in obs.columns:
        obs[section_col] = "section_0"
    score_col = cfg.cna.canonical_column
    warnings: list[str] = []
    rows: list[dict] = []
    labels = pd.Series(index=obs.index, dtype=object)
    low_s = pd.Series(index=obs.index, dtype=float)
    high_s = pd.Series(index=obs.index, dtype=float)

    sections = sorted(obs[section_col].astype(str).unique())
    for sec in sections:
        sub_mask = obs[section_col].astype(str) == sec
        sub_idx = obs.index[sub_mask]
        cna = pd.to_numeric(obs.loc[sub_idx, score_col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(cna)
        if valid.sum() == 0:
            warnings.append(f"Section '{sec}': no finite malignancy scores; all spots labeled intermediate.")
            labels.loc[sub_idx] = "intermediate"
            low_s.loc[sub_idx] = np.nan
            high_s.loc[sub_idx] = np.nan
            rows.append(
                {
                    "section_id": sec,
                    "low_threshold": float("nan"),
                    "high_threshold": float("nan"),
                    "tumor_spots": 0,
                    "normal_spots": 0,
                    "intermediate_spots": int(len(cna)),
                    "low_confidence": True,
                }
            )
            continue

        lo = float(np.quantile(cna[valid], cfg.state.low_quantile))
        hi = float(np.quantile(cna[valid], cfg.state.high_quantile))
        sec_labels = np.full(len(cna), "intermediate", dtype=object)
        sec_labels[valid & (cna >= hi)] = "tumor"
        sec_labels[valid & (cna <= lo)] = "normal"
        sec_labels[valid & (cna > lo) & (cna < hi)] = "intermediate"
        if (~valid).any():
            warnings.append(f"Section '{sec}': {(~valid).sum()} spots with non-finite score -> intermediate.")
        labels.loc[sub_idx] = sec_labels
        low_s.loc[sub_idx] = lo
        high_s.loc[sub_idx] = hi

        tumor_n = int(np.sum(sec_labels == "tumor"))
        normal_n = int(np.sum(sec_labels == "normal"))
        intermediate_n = int(np.sum(sec_labels == "intermediate"))
        low_conf = tumor_n < cfg.cna.min_spots_per_group or normal_n < cfg.cna.min_spots_per_group
        if low_conf:
            warnings.append(
                f"Section '{sec}' has low-confidence marginal construction: tumor={tumor_n}, normal={normal_n}."
            )
        rows.append(
            {
                "section_id": sec,
                "low_threshold": lo,
                "high_threshold": hi,
                "tumor_spots": tumor_n,
                "normal_spots": normal_n,
                "intermediate_spots": intermediate_n,
                "low_confidence": bool(low_conf),
            }
        )

    obs["marginal_label"] = labels.values
    obs["cna_low_threshold"] = low_s.values
    obs["cna_high_threshold"] = high_s.values
    obs["is_pseudo_paired_within_section"] = True
    obs["spot_representation"] = "pseudo_cell_mixture"
    return obs, pd.DataFrame(rows), warnings


def assign_within_section_marginals(obs: pd.DataFrame, cfg: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Alias for Stage 2 marginal assignment (shared with states.assign_marginals)."""
    return assign_marginals_from_cna_score(obs, cfg)


def compute_umap_embedding(
    expr: np.ndarray, random_state: int = 42, max_genes: int = 1500
) -> tuple[np.ndarray, str]:
    x = np.asarray(expr, dtype=float)
    g = min(max_genes, x.shape[1])
    x = x[:, :g]
    lib = x.sum(axis=1, keepdims=True) + 1e-8
    x = np.log1p(x / lib * 1e4)
    try:
        import umap  # type: ignore

        nn = max(2, min(15, x.shape[0] - 1))
        emb = umap.UMAP(
            n_neighbors=nn,
            min_dist=0.1,
            random_state=random_state,
            metric="euclidean",
        ).fit_transform(x)
        return emb, "UMAP on log1p-normalized expression (observed counts); axes = UMAP dimensions."
    except Exception:
        from sklearn.decomposition import PCA

        emb = PCA(n_components=2, random_state=random_state).fit_transform(x)
        return emb, "PCA 2D fallback (install umap-learn for UMAP); axes = principal components."


def write_stage2_umap_figures(
    bundle: DatasetBundle,
    cfg: PipelineConfig,
    fig_dir: Path,
    random_state: int = 42,
) -> list[str]:
    """
    Stage 2 validation UMAPs per visualization_standards.md (tumor=warm, normal=cool, intermediate=neutral).
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    obs = bundle.obs
    emb, basis_note = compute_umap_embedding(bundle.expr, random_state=random_state)
    score_col = cfg.cna.canonical_column
    scores = pd.to_numeric(obs[score_col], errors="coerce").to_numpy()
    paths: list[str] = []

    # Continuous malignancy score
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    sc = ax.scatter(
        emb[:, 0],
        emb[:, 1],
        c=scores,
        cmap="plasma",
        s=10,
        alpha=0.85,
        edgecolors="none",
    )
    plt.colorbar(sc, ax=ax, label=score_col)
    ax.set_xlabel("UMAP 1" if "UMAP" in basis_note else "Component 1")
    ax.set_ylabel("UMAP 2" if "UMAP" in basis_note else "Component 2")
    ax.set_title(f"Stage 2 — {score_col} on expression embedding\n{basis_note}")
    fig.tight_layout()
    p_score = fig_dir / "stage_2_umap_cna_score.png"
    fig.savefig(p_score, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(p_score))

    # Marginal labels
    color_map = {"tumor": "#d62728", "normal": "#1f77b4", "intermediate": "#9e9e9e"}
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    labs = obs["marginal_label"].astype(str)
    for lab in ("tumor", "normal", "intermediate"):
        m = labs == lab
        if m.any():
            ax.scatter(
                emb[m, 0],
                emb[m, 1],
                s=10,
                alpha=0.85,
                c=color_map.get(lab, "#333333"),
                label=lab,
                edgecolors="none",
            )
    ax.set_xlabel("UMAP 1" if "UMAP" in basis_note else "Component 1")
    ax.set_ylabel("UMAP 2" if "UMAP" in basis_note else "Component 2")
    ax.set_title(f"Stage 2 — within-section marginal labels\n{basis_note}")
    ax.legend(title="marginal_label", loc="best")
    fig.tight_layout()
    p_marg = fig_dir / "stage_2_umap_marginal_labels.png"
    fig.savefig(p_marg, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(p_marg))

    # Provenance
    prov = obs["malignancy_provenance"].astype(str) if "malignancy_provenance" in obs.columns else pd.Series(["unknown"] * len(obs))
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    uniq = sorted(prov.unique())
    cmap = plt.colormaps["tab10"]
    colors = cmap(np.linspace(0, 1, max(len(uniq), 1), endpoint=False))
    for i, u in enumerate(uniq):
        m = prov == u
        ax.scatter(
            emb[m, 0],
            emb[m, 1],
            s=10,
            alpha=0.85,
            color=colors[i % len(colors)],
            label=u,
            edgecolors="none",
        )
    ax.set_xlabel("UMAP 1" if "UMAP" in basis_note else "Component 1")
    ax.set_ylabel("UMAP 2" if "UMAP" in basis_note else "Component 2")
    ax.set_title(
        f"Stage 2 — malignancy score provenance\n{basis_note}\n"
        "Training marginals must not use evaluation-only columns (layer, ivygap, CODEX, etc.)."
    )
    ax.legend(title="malignancy_provenance", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    p_prov = fig_dir / "stage_2_umap_malignancy_provenance.png"
    fig.savefig(p_prov, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(p_prov))

    return paths


def run_marginal_definition(bundle: DatasetBundle, cfg: PipelineConfig, out_dir: Path) -> CNAStageResult:
    """
    Stage 2 entry: canonical malignancy column, provenance, within-section tumor/normal/intermediate marginals,
    CSV summaries, histograms, spatial map, and UMAP validation figures.
    """
    obs, prov_key, source, warnings, decision_path = resolve_malignancy_scores(bundle, cfg)
    obs, section_counts, warn2 = assign_marginals_from_cna_score(obs, cfg)
    warnings.extend(warn2)

    score_col = cfg.cna.canonical_column
    score_vals = pd.to_numeric(obs[score_col], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(score_vals)
    score_mean = float(np.mean(score_vals[finite])) if finite.any() else float("nan")
    score_std = float(np.std(score_vals[finite])) if finite.sum() > 1 else float("nan")
    summary = pd.DataFrame(
        [
            {
                "cna_source": source,
                "malignancy_provenance": prov_key,
                "n_spots": int(len(obs)),
                "score_mean": score_mean,
                "score_std": score_std,
                "n_tumor": int((obs["marginal_label"] == "tumor").sum()),
                "n_normal": int((obs["marginal_label"] == "normal").sum()),
                "n_intermediate": int((obs["marginal_label"] == "intermediate").sum()),
                "warnings": " | ".join(warnings) if warnings else "None",
            }
        ]
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "malignancy_scoring_summary.csv", index=False)
    section_counts.to_csv(out_dir / "malignancy_counts_by_section.csv", index=False)

    # Stage 2 validation artifacts: marginal sanity and cna_bin alignment diagnostics.
    sanity_rows: list[dict[str, object]] = []
    if "marginal_label" in obs.columns:
        for sec, sub in obs.groupby(cfg.state.section_column, dropna=False):
            svals = pd.to_numeric(sub[score_col], errors="coerce")
            sanity_rows.append(
                {
                    "section_id": str(sec),
                    "n_spots": int(len(sub)),
                    "score_q10": float(np.nanquantile(svals, 0.1)),
                    "score_q50": float(np.nanquantile(svals, 0.5)),
                    "score_q90": float(np.nanquantile(svals, 0.9)),
                    "tumor_fraction": float(np.mean(sub["marginal_label"].astype(str) == "tumor")),
                    "normal_fraction": float(np.mean(sub["marginal_label"].astype(str) == "normal")),
                    "intermediate_fraction": float(np.mean(sub["marginal_label"].astype(str) == "intermediate")),
                }
            )
    marginal_sanity_df = pd.DataFrame(sanity_rows)
    if not marginal_sanity_df.empty:
        marginal_sanity_df.to_csv(out_dir / "stage_2_marginal_sanity_table.csv", index=False)

    cna_bin_alignment_path = out_dir / "stage_2_cna_bin_alignment_table.csv"
    if "cna_bin" in obs.columns and "marginal_label" in obs.columns:
        ct = pd.crosstab(
            obs["cna_bin"].astype(str),
            obs["marginal_label"].astype(str),
            normalize="index",
        )
        ct = ct.reindex(columns=["tumor", "intermediate", "normal"], fill_value=0.0)
        ct.reset_index().to_csv(cna_bin_alignment_path, index=False)

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 3))
    for sec in sorted(obs[cfg.state.section_column].astype(str).unique()):
        sub = obs[obs[cfg.state.section_column].astype(str) == sec]
        plt.hist(pd.to_numeric(sub[score_col], errors="coerce").dropna().to_numpy(), bins=40, alpha=0.4, label=str(sec))
    plt.title("Malignancy score distribution by section")
    plt.xlabel(score_col)
    plt.ylabel("Spot count")
    if obs[cfg.state.section_column].nunique() <= 8:
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "malignancy_score_distribution_by_section.png", dpi=150)
    plt.close()

    comp = section_counts.copy()
    if {"section_id", "marginal_label", "n_spots"}.issubset(comp.columns):
        pivot = (
            comp.pivot(index="section_id", columns="marginal_label", values="n_spots")
            .fillna(0.0)
            .reindex(columns=["tumor", "intermediate", "normal"], fill_value=0.0)
        )
        plt.figure(figsize=(7.5, 4.5))
        bottom = np.zeros(len(pivot), dtype=float)
        colors = {"tumor": "#d62728", "intermediate": "#7f7f7f", "normal": "#1f77b4"}
        for col in pivot.columns:
            vals = pivot[col].to_numpy(dtype=float)
            plt.bar(pivot.index.astype(str), vals, bottom=bottom, color=colors.get(col, "#cccccc"), label=col)
            bottom += vals
        plt.title("Marginal composition by section")
        plt.xlabel(cfg.state.section_column)
        plt.ylabel("Spot count")
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="marginal_label", fontsize=8)
        plt.tight_layout()
        plt.savefig(fig_dir / "stage_2_marginal_composition_by_section.png", dpi=150)
        plt.close()

    x_col = cfg.spatial.x_column if cfg.spatial.x_column in obs.columns else "x"
    y_col = cfg.spatial.y_column if cfg.spatial.y_column in obs.columns else "y"
    if x_col in obs.columns and y_col in obs.columns:
        plt.figure(figsize=(6, 5))
        plt.scatter(
            obs[x_col],
            obs[y_col],
            c=pd.to_numeric(obs[score_col], errors="coerce"),
            s=8,
            cmap="magma",
        )
        plt.title("Spatial map of malignancy score")
        plt.colorbar(label=score_col)
        plt.tight_layout()
        plt.savefig(fig_dir / "malignancy_spatial_map.png", dpi=150)
        plt.close()

        if "marginal_label" in obs.columns:
            plt.figure(figsize=(6, 5))
            color_map = {"tumor": "#d62728", "normal": "#1f77b4", "intermediate": "#7f7f7f"}
            labs = obs["marginal_label"].astype(str)
            for lab in ("tumor", "intermediate", "normal"):
                m = labs == lab
                if not m.any():
                    continue
                plt.scatter(obs.loc[m, x_col], obs.loc[m, y_col], s=8, c=color_map[lab], label=lab, alpha=0.85)
            plt.title("Spatial map of marginal labels")
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.legend(title="marginal_label", fontsize=8)
            plt.tight_layout()
            plt.savefig(fig_dir / "stage_2_spatial_marginal_labels.png", dpi=150)
            plt.close()

    if not marginal_sanity_df.empty:
        plt.figure(figsize=(8, 4.8))
        x = np.arange(len(marginal_sanity_df))
        w = 0.26
        plt.bar(
            x - w,
            marginal_sanity_df["tumor_fraction"].to_numpy(dtype=float),
            width=w,
            color="#d62728",
            label="tumor",
        )
        plt.bar(
            x,
            marginal_sanity_df["intermediate_fraction"].to_numpy(dtype=float),
            width=w,
            color="#7f7f7f",
            label="intermediate",
        )
        plt.bar(
            x + w,
            marginal_sanity_df["normal_fraction"].to_numpy(dtype=float),
            width=w,
            color="#1f77b4",
            label="normal",
        )
        plt.xticks(x, marginal_sanity_df["section_id"].astype(str), rotation=45, ha="right")
        plt.ylabel("fraction of section spots")
        plt.title("Stage 2 — section-level marginal sanity")
        plt.legend(title="marginal_label")
        plt.tight_layout()
        plt.savefig(fig_dir / "stage_2_section_level_marginal_sanity.png", dpi=150)
        plt.close()

    if cna_bin_alignment_path.is_file():
        ctab = pd.read_csv(cna_bin_alignment_path)
        cols = [c for c in ["tumor", "intermediate", "normal"] if c in ctab.columns]
        if cols:
            plt.figure(figsize=(8, 4.8))
            xpos = np.arange(len(ctab))
            bottom = np.zeros(len(ctab), dtype=float)
            cmap = {"tumor": "#d62728", "intermediate": "#7f7f7f", "normal": "#1f77b4"}
            for c in cols:
                vals = pd.to_numeric(ctab[c], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                plt.bar(xpos, vals, bottom=bottom, color=cmap.get(c, "#cccccc"), label=c)
                bottom += vals
            plt.xticks(xpos, ctab["cna_bin"].astype(str), rotation=35, ha="right")
            plt.ylim(0.0, 1.0)
            plt.ylabel("fraction within cna_bin")
            plt.title("Stage 2 — cna_bin-aligned marginal diagnostics")
            plt.legend(title="marginal_label")
            plt.tight_layout()
            plt.savefig(fig_dir / "stage_2_cna_bin_alignment.png", dpi=150)
            plt.close()

    umap_paths = write_stage2_umap_figures(
        DatasetBundle(
            expr=bundle.expr,
            obs=obs,
            var_names=bundle.var_names,
            source_path=bundle.source_path,
            dataset_kind=bundle.dataset_kind,
        ),
        cfg,
        fig_dir,
        random_state=cfg.train.random_seed,
    )

    out_bundle = DatasetBundle(
        expr=bundle.expr,
        obs=obs,
        var_names=bundle.var_names,
        source_path=bundle.source_path,
        dataset_kind=bundle.dataset_kind,
    )
    return CNAStageResult(
        bundle=out_bundle,
        source=source,
        warnings=warnings,
        summary=summary,
        section_counts=section_counts,
        malignancy_provenance=prov_key,
        decision_path=decision_path,
        umap_paths=umap_paths,
    )


def run_cna_inference_or_scoring(bundle: DatasetBundle, cfg: PipelineConfig, out_dir: Path) -> CNAStageResult:
    """Pipeline-compatible name for Stage 2 marginal definition."""
    return run_marginal_definition(bundle, cfg, out_dir)
