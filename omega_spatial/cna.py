from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import PipelineConfig
from .types import DatasetBundle


@dataclass
class CNAStageResult:
    bundle: DatasetBundle
    source: str
    warnings: list[str]
    summary: pd.DataFrame
    section_counts: pd.DataFrame


def find_existing_cna_score(obs: pd.DataFrame, cfg: PipelineConfig) -> tuple[pd.Series | None, str | None]:
    canonical = cfg.cna.canonical_column
    aliases = [canonical] + [a for a in cfg.cna.aliases if a != canonical]
    lowered = {str(c).lower(): c for c in obs.columns}
    for alias in aliases:
        if alias in obs.columns:
            return pd.to_numeric(obs[alias], errors="coerce"), alias
        if alias.lower() in lowered:
            col = lowered[alias.lower()]
            return pd.to_numeric(obs[col], errors="coerce"), col
    for c in obs.columns:
        lc = str(c).lower()
        if "cna" in lc or "malignan" in lc:
            return pd.to_numeric(obs[c], errors="coerce"), str(c)
    return None, None


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

    # Score is mean absolute chromosome-scale deviation, rank-normalized to [0,1].
    raw_score = np.mean(np.abs(deviations), axis=1)
    order = np.argsort(raw_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, len(raw_score), endpoint=True)
    score = ranks
    return score, {"mapped_genes": float(mapped.shape[0]), "score_std": float(np.std(score))}


def assign_marginals_from_cna_score(obs: pd.DataFrame, cfg: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    section_col = cfg.state.section_column
    if section_col not in obs.columns:
        obs = obs.copy()
        obs[section_col] = "section_0"
    warnings: list[str] = []
    rows = []
    labels = pd.Series(index=obs.index, dtype=object)

    for sec, sub in obs.groupby(section_col):
        cna = pd.to_numeric(sub[cfg.cna.canonical_column], errors="coerce").to_numpy()
        lo = np.quantile(cna, cfg.state.low_quantile)
        hi = np.quantile(cna, cfg.state.high_quantile)
        sec_labels = np.where(cna >= hi, "tumor", np.where(cna <= lo, "normal", "intermediate"))
        labels.loc[sub.index] = sec_labels

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
                "low_threshold": float(lo),
                "high_threshold": float(hi),
                "tumor_spots": tumor_n,
                "normal_spots": normal_n,
                "intermediate_spots": intermediate_n,
                "low_confidence": bool(low_conf),
            }
        )

    obs = obs.copy()
    obs["marginal_label"] = labels.values
    obs["is_pseudo_paired_within_section"] = True
    obs["spot_representation"] = "pseudo_cell_mixture"
    return obs, pd.DataFrame(rows), warnings


def run_cna_inference_or_scoring(bundle: DatasetBundle, cfg: PipelineConfig, out_dir: Path) -> CNAStageResult:
    obs = bundle.obs.copy()
    warnings: list[str] = []

    score, source_col = find_existing_cna_score(obs, cfg)
    source = ""
    if score is not None:
        obs[cfg.cna.canonical_column] = score.to_numpy(dtype=float)
        source = f"provided:{source_col}"
    else:
        if not cfg.cna.infer_if_missing:
            raise ValueError(
                f"No malignancy score found in input and cna.infer_if_missing=false. "
                f"Provide '{cfg.cna.canonical_column}' or enable inference."
            )
        inferred, stats = infer_cna_score_from_expression(bundle, cfg)
        if float(np.std(inferred)) <= cfg.cna.constant_score_std_threshold:
            raise ValueError(
                "Inferred malignancy score is nearly constant across spots and unusable for marginal construction."
            )
        obs[cfg.cna.canonical_column] = inferred
        source = "inferred_from_expression"
        warnings.append(f"Inferred malignancy score from expression; mapped_genes={int(stats['mapped_genes'])}.")

    obs, section_counts, warn2 = assign_marginals_from_cna_score(obs, cfg)
    warnings.extend(warn2)

    summary = pd.DataFrame(
        [
            {
                "cna_source": source,
                "n_spots": int(len(obs)),
                "score_mean": float(np.mean(obs[cfg.cna.canonical_column])),
                "score_std": float(np.std(obs[cfg.cna.canonical_column])),
                "n_tumor": int((obs["marginal_label"] == "tumor").sum()),
                "n_normal": int((obs["marginal_label"] == "normal").sum()),
                "n_intermediate": int((obs["marginal_label"] == "intermediate").sum()),
                "warnings": " | ".join(warnings) if warnings else "None",
            }
        ]
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "malignancy_scoring_summary.csv", index=False)
    section_counts.to_csv(out_dir / "malignancy_counts_by_section.csv", index=False)

    # Required plots.
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 3))
    for sec, sub in obs.groupby(cfg.state.section_column):
        plt.hist(sub[cfg.cna.canonical_column].to_numpy(), bins=40, alpha=0.4, label=str(sec))
    plt.title("Malignancy score distribution by section")
    plt.xlabel(cfg.cna.canonical_column)
    plt.ylabel("Spot count")
    if obs[cfg.state.section_column].nunique() <= 8:
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "malignancy_score_distribution_by_section.png", dpi=150)
    plt.close()

    x_col = cfg.spatial.x_column if cfg.spatial.x_column in obs.columns else "x"
    y_col = cfg.spatial.y_column if cfg.spatial.y_column in obs.columns else "y"
    if x_col in obs.columns and y_col in obs.columns:
        plt.figure(figsize=(6, 5))
        plt.scatter(obs[x_col], obs[y_col], c=obs[cfg.cna.canonical_column], s=8, cmap="magma")
        plt.title("Spatial map of inferred malignancy score")
        plt.colorbar(label=cfg.cna.canonical_column)
        plt.tight_layout()
        plt.savefig(fig_dir / "malignancy_spatial_map.png", dpi=150)
        plt.close()

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
    )
