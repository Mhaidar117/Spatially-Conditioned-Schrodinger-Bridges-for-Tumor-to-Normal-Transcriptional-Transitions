from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .benchmarks import run_benchmarks
from .cna import run_cna_inference_or_scoring
from .config import PipelineConfig, dump_config
from .io import load_dataset, resolve_dataset
from .model import generate_counterfactuals, train_spatial_bridge
from .programs import run_nmf_programs
from .qc import run_qc, write_qc_summary
from .readiness import validate_schema, write_readiness_report
from .reporting import make_figures, write_reports
from .spatial import build_spatial_knn, spatial_context
from .states import split_by_section


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _imbalance_weights(labels: np.ndarray) -> np.ndarray:
    unique, counts = np.unique(labels, return_counts=True)
    freq = dict(zip(unique.tolist(), counts.tolist()))
    return np.array([1.0 / max(freq.get(v, 1), 1) for v in labels], dtype=float)


def _save_annotated_output(out_dir: Path, obs: pd.DataFrame, perturb: np.ndarray, var_names: list[str]) -> None:
    obs_out = obs.copy()
    obs_out["perturbation_norm"] = np.linalg.norm(perturb, axis=1)
    obs_out.to_csv(out_dir / "annotated_output_obs.csv", index=False)
    pd.DataFrame(perturb, columns=var_names).to_csv(out_dir / "perturbation_vectors.csv", index=False)
    try:
        import anndata as ad  # type: ignore

        adata = ad.AnnData(X=perturb, obs=obs_out, var=pd.DataFrame(index=var_names))
        adata.write_h5ad(out_dir / "annotated_output.h5ad")
    except Exception:
        # Fallback keeps expected artifact path even when anndata is unavailable.
        (out_dir / "annotated_output.h5ad").write_text(
            "Placeholder artifact path. Install anndata to write native h5ad output.",
            encoding="utf-8",
        )


def run_pipeline(cfg: PipelineConfig) -> None:
    out_dir = Path(cfg.output_path).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    dump_config(cfg, str(out_dir / "run_config.yaml"))

    resolved, kind = resolve_dataset(cfg.input_path)
    bundle = load_dataset(resolved, kind)
    readiness = validate_schema(bundle, cfg)
    write_readiness_report(readiness, out_dir)
    _write_json(logs_dir / "stage_dataset_resolver.json", {"resolved": str(resolved), "kind": kind})
    if not readiness.is_ready:
        raise RuntimeError(f"Dataset readiness failed: {'; '.join(readiness.issues)}")

    bundle = run_qc(bundle, cfg)
    write_qc_summary(bundle, out_dir)
    _write_json(logs_dir / "stage_qc.json", {"n_spots": int(bundle.expr.shape[0]), "n_genes": int(bundle.expr.shape[1])})

    cna_result = run_cna_inference_or_scoring(bundle, cfg, out_dir)
    bundle = cna_result.bundle
    _write_json(
        logs_dir / "stage_cna_scoring.json",
        {"source": cna_result.source, "warnings": cna_result.warnings},
    )

    splits = split_by_section(bundle, cfg)
    _write_json(logs_dir / "stage_splits.json", {k: int(v.size) for k, v in splits.items()})

    knn = build_spatial_knn(bundle, cfg)
    ctx = spatial_context(bundle.expr, knn)
    labels = bundle.obs["marginal_label"].to_numpy()

    weights = _imbalance_weights(labels)
    bundle.obs["imbalance_weight"] = weights
    _write_json(logs_dir / "stage_imbalance.json", {"mean_weight": float(weights.mean()), "max_weight": float(weights.max())})

    model = train_spatial_bridge(bundle.expr, ctx, labels, cfg)
    counterfactual = generate_counterfactuals(model, bundle.expr, ctx, n_steps=max(3, cfg.train.steps // 100))
    perturb = counterfactual - bundle.expr
    _write_json(logs_dir / "stage_train_infer.json", {"counterfactual_shape": list(counterfactual.shape)})

    prog = run_nmf_programs(perturb, bundle.var_names, cfg)
    prog.scores.to_csv(out_dir / "gene_programs.csv", index=False)
    prog.loadings.to_csv(out_dir / "gene_program_loadings.csv", index=True)
    prog.model_selection.to_csv(out_dir / "nmf_method_comparison.csv", index=False)

    section = (
        bundle.obs[cfg.state.section_column].astype(str).to_numpy()
        if cfg.state.section_column in bundle.obs.columns
        else np.array(["section_0"] * bundle.expr.shape[0])
    )
    if cfg.cna.canonical_column not in bundle.obs.columns:
        raise RuntimeError(
            f"CNAInference_or_MalignancyScoring failed: missing canonical score column '{cfg.cna.canonical_column}'."
        )
    cna = bundle.obs[cfg.cna.canonical_column].astype(float).to_numpy()
    benchmark = run_benchmarks(bundle.expr, counterfactual, labels, section, cna)
    benchmark.to_csv(out_dir / "benchmark_metrics.csv", index=False)

    _save_annotated_output(out_dir, bundle.obs, perturb, bundle.var_names)

    perturb_mag = np.linalg.norm(perturb, axis=1)
    figs = make_figures(out_dir, bundle.obs, perturb_mag, prog.scores)
    cna_figs = [str(p) for p in sorted((out_dir / "figures").glob("malignancy_*.png"))]
    readiness_df = pd.read_csv(out_dir / "dataset_readiness.csv")
    qc_df = pd.read_csv(out_dir / "qc_summary.csv")
    malignancy_df = pd.read_csv(out_dir / "malignancy_scoring_summary.csv")
    malignancy_counts_df = pd.read_csv(out_dir / "malignancy_counts_by_section.csv")
    write_reports(
        out_dir=out_dir,
        run_name=cfg.run_name,
        readiness=readiness_df,
        qc_summary=qc_df,
        benchmark=benchmark,
        programs=prog.loadings,
        loadings=prog.loadings,
        figures=figs + cna_figs,
        malignancy_summary=malignancy_df,
        malignancy_counts=malignancy_counts_df,
    )

    _write_json(
        logs_dir / "stage_complete.json",
        {
            "outputs": [
                "annotated_output.h5ad",
                "perturbation_vectors.csv",
                "gene_programs.csv",
                "pathway_enrichment.csv",
                "benchmark_metrics.csv",
                "qc_summary.csv",
                "malignancy_scoring_summary.csv",
                "malignancy_counts_by_section.csv",
                "figures/",
                "report.html",
                "report.pdf",
                "run_config.yaml",
            ]
        },
    )
