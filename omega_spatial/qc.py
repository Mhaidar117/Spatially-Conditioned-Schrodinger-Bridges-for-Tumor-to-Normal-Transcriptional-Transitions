from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .types import DatasetBundle


def run_qc(bundle: DatasetBundle, cfg: PipelineConfig) -> DatasetBundle:
    expr = np.asarray(bundle.expr, dtype=float).copy()
    obs = bundle.obs.copy()

    spot_counts = expr.sum(axis=1)
    spot_genes = (expr > 0).sum(axis=1)
    keep = (spot_counts >= cfg.qc.min_counts_per_spot) & (spot_genes >= cfg.qc.min_genes_per_spot)
    if keep.sum() == 0:
        keep = np.ones(expr.shape[0], dtype=bool)

    expr = expr[keep]
    obs = obs.loc[keep].copy()
    obs["qc_total_counts"] = spot_counts[keep]
    obs["qc_n_genes_by_counts"] = spot_genes[keep]

    libsize = expr.sum(axis=1, keepdims=True) + 1e-8
    expr = expr / libsize * 1e4
    if cfg.qc.log1p:
        expr = np.log1p(expr)

    # HVG-style simple variance filter.
    var = np.var(expr, axis=0)
    top = min(cfg.qc.top_hvg, expr.shape[1])
    idx = np.argsort(var)[::-1][:top]
    expr = expr[:, idx]
    genes = [bundle.var_names[i] for i in idx]
    return DatasetBundle(expr=expr, obs=obs, var_names=genes, source_path=bundle.source_path, dataset_kind=bundle.dataset_kind)


def write_qc_summary(bundle: DatasetBundle, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(
        [
            {
                "n_spots_after_qc": int(bundle.expr.shape[0]),
                "n_genes_after_qc": int(bundle.expr.shape[1]),
                "mean_counts": float(bundle.obs.get("qc_total_counts", pd.Series(dtype=float)).mean()),
                "mean_detected_genes": float(bundle.obs.get("qc_n_genes_by_counts", pd.Series(dtype=float)).mean()),
            }
        ]
    )
    summary.to_csv(out_dir / "qc_summary.csv", index=False)
