from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from .config import PipelineConfig
from .types import DatasetBundle, IngestionDiagnostics, ReadinessReport


def _find_spatial_columns(obs: pd.DataFrame, cfg: PipelineConfig) -> tuple[str, str] | None:
    x_candidates = [cfg.spatial.x_column, "x", "array_col", "pxl_col_in_fullres"]
    y_candidates = [cfg.spatial.y_column, "y", "array_row", "pxl_row_in_fullres"]
    x_col = next((c for c in x_candidates if c in obs.columns), None)
    y_col = next((c for c in y_candidates if c in obs.columns), None)
    if x_col and y_col:
        return x_col, y_col
    return None


def get_ingestion_diagnostics(bundle: DatasetBundle) -> IngestionDiagnostics | None:
    """Parse Stage 1 ingestion summary stored on ``obs.attrs['omega_ingestion']``."""
    raw = bundle.obs.attrs.get("omega_ingestion") if hasattr(bundle.obs, "attrs") else None
    if not raw or not isinstance(raw, dict):
        return None
    try:
        return IngestionDiagnostics(**raw)
    except TypeError:
        return None


def diagnose_stage1_ingestion(bundle: DatasetBundle) -> dict[str, Any]:
    """
    Human- and machine-readable Stage 1 report: schema, join coverage, evaluation columns.
    Intended for logs and ``stage_1_artifacts.json``.
    """
    obs = bundle.obs
    n_spots, n_genes = int(bundle.expr.shape[0]), int(bundle.expr.shape[1])
    required = ["section_id", "x", "y"]
    missing_required = [c for c in required if c not in obs.columns]
    eval_present = [c for c in ("mp", "layer", "ivygap", "org1", "org2", "cc", "cna_bin") if c in obs.columns]
    malig_present = [c for c in ("cna_score", "malignancy_score", "cna_bin") if c in obs.columns]
    join_rate = None
    if "metadata_joined" in obs.columns:
        join_rate = float(obs["metadata_joined"].astype(bool).mean())
    ing = get_ingestion_diagnostics(bundle)
    return {
        "dataset_kind": bundle.dataset_kind,
        "source_path": str(bundle.source_path),
        "n_spots": n_spots,
        "n_genes": n_genes,
        "n_sections": int(obs["section_id"].nunique()) if "section_id" in obs.columns else 0,
        "missing_canonical_columns": missing_required,
        "evaluation_columns_present": eval_present,
        "malignancy_related_columns_present": malig_present,
        "cna_bin_available": "cna_bin" in obs.columns,
        "metadata_join_coverage": join_rate,
        "spot_id_in_obs": "spot_id" in obs.columns,
        "barcode_in_obs": "barcode" in obs.columns,
        "ingestion_diagnostics": asdict(ing) if ing is not None else None,
    }


def _find_cna_column(obs: pd.DataFrame, cfg: PipelineConfig) -> str | None:
    blocklisted = {"cna_bin", "layer", "ivygap", "org1", "org2", "mp"}
    if cfg.state.cna_column in obs.columns:
        vals = pd.to_numeric(obs[cfg.state.cna_column], errors="coerce")
        if vals.notna().sum() > 0:
            return cfg.state.cna_column
    for col in obs.columns:
        lower = str(col).lower()
        if lower in blocklisted:
            continue
        if "cna" in lower or "malignan" in lower:
            vals = pd.to_numeric(obs[col], errors="coerce")
            if vals.notna().sum() > 0:
                return str(col)
    return None


def validate_schema(bundle: DatasetBundle, cfg: PipelineConfig) -> ReadinessReport:
    issues: list[str] = []
    recs: list[str] = []
    spatial = _find_spatial_columns(bundle.obs, cfg)
    cna = _find_cna_column(bundle.obs, cfg)
    cna_source = "provided" if cna is not None else "missing"
    will_run_cna_inference = cna is None and cfg.cna.infer_if_missing
    reference_available = bool(cfg.cna.reference_normal_path and Path(cfg.cna.reference_normal_path).expanduser().exists())

    if spatial is None:
        issues.append("Missing spatial coordinates (x/y).")
        recs.append("Add x and y columns (or spatial obsm for h5ad).")
    if cna is None and cfg.cna.require_true_score:
        issues.append(
            f"Missing true CNA score '{cfg.cna.canonical_column}'. Stage 2 is configured to hard-fail without precomputed true CNA."
        )
    elif cna is None and not cfg.cna.infer_if_missing:
        issues.append(
            f"Missing malignancy score and cna.infer_if_missing=false. Provide '{cfg.cna.canonical_column}' or enable inference."
        )
    if cna is None and cfg.cna.infer_if_missing:
        ann_ok = bool(cfg.cna.gene_annotation_path and Path(cfg.cna.gene_annotation_path).expanduser().exists())
        if not ann_ok:
            issues.append(
                "CNAInference_or_MalignancyScoring requires gene genomic annotation (gene_id/chromosome/position). "
                "Set cna.gene_annotation_path to a valid annotation file."
            )
        else:
            recs.append("No precomputed malignancy score found; CNA inference will run from expression.")
    if bundle.expr.ndim != 2 or bundle.expr.shape[0] == 0 or bundle.expr.shape[1] == 0:
        issues.append("Expression matrix must be 2D and non-empty.")
    if cfg.state.section_column not in bundle.obs.columns:
        recs.append(f"Section column '{cfg.state.section_column}' missing; defaulting to single section.")

    sufficient_for_pseudopairing = len(issues) == 0
    n_sections = bundle.obs[cfg.state.section_column].nunique() if cfg.state.section_column in bundle.obs.columns else 1
    return ReadinessReport(
        is_ready=len(issues) == 0,
        dataset_kind=bundle.dataset_kind,
        n_sections=int(n_sections),
        n_spots=int(bundle.expr.shape[0]),
        n_genes=int(bundle.expr.shape[1]),
        spatial_columns=spatial,
        cna_column=cna,
        cna_source=cna_source,
        will_run_cna_inference=will_run_cna_inference,
        reference_normal_available=reference_available,
        sufficient_for_pseudopairing=sufficient_for_pseudopairing,
        issues=issues,
        recommendations=recs,
    )


def write_readiness_report(report: ReadinessReport, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    row = asdict(report)
    row["spatial_columns"] = ",".join(report.spatial_columns) if report.spatial_columns else "N/A"
    row["cna_column"] = report.cna_column if report.cna_column else "N/A"
    row["issues"] = " | ".join(report.issues) if report.issues else "No issues"
    row["recommendations"] = " | ".join(report.recommendations) if report.recommendations else "None"
    pd.DataFrame([row]).to_csv(output_dir / "dataset_readiness.csv", index=False)
