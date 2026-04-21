from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .benchmarks import run_benchmarks_and_baselines
from .cna import run_cna_inference_or_scoring
from .config import PipelineConfig, dump_config
from .io import load_dataset, resolve_dataset
from .model import save_bridge_checkpoint, train_spatial_bridge, train_transport_backend
from .perturbations import (
    extract_perturbations,
    save_perturbation_artifacts,
    write_stage4_umap_figures,
    write_stage5_summary_figures,
    write_stage5_umap_figures,
    write_stage5_artifact_manifest,
)
from .programs import (
    run_program_discovery,
    save_program_artifacts,
    write_stage6_artifact_manifest,
    write_stage6_summary_figures,
    write_stage6_umap_figures,
)
from .qc import run_qc, write_qc_summary
from .readiness import validate_schema, write_readiness_report
from .reporting import make_figures, run_stage7_reporting, write_reports
from .spatial import (
    build_spatial_neighborhoods,
    write_stage3_artifact_manifest,
    write_stage3_umap_figures,
)
from .states import split_by_section
from .validation import run_stage8_heldout_validation, run_stage9_cross_modal_validation

_REPO_ROOT = Path(__file__).resolve().parents[1]


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
    bundle = load_dataset(resolved, kind, cfg=cfg)
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

    stage3_log = logs_dir / "stage_3_spatial_context.log"
    s3_logger = logging.getLogger("omega_spatial.stage3_pipeline")
    s3_logger.setLevel(logging.INFO)
    s3_logger.handlers.clear()
    s3_fh = logging.FileHandler(stage3_log, encoding="utf-8")
    s3_fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    s3_logger.addHandler(s3_fh)
    neighborhood = build_spatial_neighborhoods(bundle, cfg, log=s3_logger)
    knn = neighborhood.knn_indices
    ctx = neighborhood.context_matrix
    figd = out_dir / "figures"
    stage3_umap_paths = write_stage3_umap_figures(
        bundle,
        neighborhood,
        figd,
        s3_logger,
        random_state=cfg.programs.umap_random_state,
    )
    knn_path = out_dir / "knn_indices.npy"
    dist_path = out_dir / "neighbor_distances.npy"
    ctx_path = out_dir / "context_matrix.npy"
    np.save(knn_path, knn)
    np.save(dist_path, neighborhood.neighbor_distances)
    np.save(ctx_path, ctx)
    write_stage3_artifact_manifest(
        logs_dir / "stage_3_artifacts.json",
        graph_paths={
            "knn_indices_npy": str(knn_path.resolve()),
            "neighbor_distances_npy": str(dist_path.resolve()),
            "context_matrix_npy": str(ctx_path.resolve()),
        },
        umap_paths=stage3_umap_paths,
        diagnostics=neighborhood.diagnostics,
        known_limitations=[
            "UMAPs use separate fits for pre-context vs context-enriched features.",
            "Section-restricted kNN uses observed coordinates and pads neighbors for small sections.",
        ],
        test_script_path=str((_REPO_ROOT / "tests/test_stage_3_spatial_context.py").resolve()),
        extra={
            "neighbor_distance_summary": {
                "mean": float(neighborhood.mean_neighbor_distance.mean()),
                "std": float(neighborhood.mean_neighbor_distance.std()),
                "p50": float(np.median(neighborhood.mean_neighbor_distance)),
                "p90": float(np.quantile(neighborhood.mean_neighbor_distance, 0.9)),
            },
            "section_neighbor_purity_mean": float(neighborhood.section_neighbor_purity.mean()),
        },
    )
    labels = bundle.obs["marginal_label"].to_numpy()

    weights = _imbalance_weights(labels)
    bundle.obs["imbalance_weight"] = weights
    _write_json(logs_dir / "stage_imbalance.json", {"mean_weight": float(weights.mean()), "max_weight": float(weights.max())})

    model = train_transport_backend(bundle.expr, ctx, labels, cfg)
    n_transport_steps = max(3, cfg.train.steps // 100)
    stage5_result = extract_perturbations(
        model,
        bundle.expr,
        ctx,
        bundle.obs,
        bundle.var_names,
        n_steps=n_transport_steps,
    )
    perturb = stage5_result.perturbation
    transported = stage5_result.transported
    selected_backend = getattr(model, "backend", "linear")
    linear_shadow_model = None
    linear_shadow_transported = None
    if selected_backend == "neural":
        linear_shadow_model = train_spatial_bridge(bundle.expr, ctx, labels, cfg)
        linear_shadow_transported = linear_shadow_model.transport(
            bundle.expr,
            ctx,
            n_steps=n_transport_steps,
        )
    stage5_paths = save_perturbation_artifacts(out_dir, stage5_result)
    bridge_ckpt = out_dir / ("bridge_model.pt" if selected_backend == "neural" else "bridge_model.npz")
    save_bridge_checkpoint(model, bridge_ckpt)
    _write_json(
        logs_dir / "stage_train_infer.json",
        {
            "counterfactual_shape": list(transported.shape),
            "stage5_inference_entry": stage5_result.inference_entry_point,
            "stage5_n_steps": stage5_result.n_steps,
            "transport_backend": selected_backend,
        },
    )

    stage4_log = logs_dir / "stage_4_bridge_model.log"
    s4_logger = logging.getLogger("omega_spatial.stage4_pipeline")
    s4_logger.setLevel(logging.INFO)
    s4_logger.handlers.clear()
    s4_fh = logging.FileHandler(stage4_log, encoding="utf-8")
    s4_fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    s4_logger.addHandler(s4_fh)
    s4_logger.info("Bridge training summary: %s", json.dumps(model.training_summary.to_dict()))
    s4_logger.info("Selected transport backend: %s", selected_backend)
    if linear_shadow_model is not None:
        s4_logger.info(
            "Linear shadow bridge summary for comparisons: %s",
            json.dumps(linear_shadow_model.training_summary.to_dict()),
        )
    stage4_umap_paths = write_stage4_umap_figures(
        bundle.expr,
        transported,
        labels,
        model.normal_reference,
        figd,
        s4_logger,
        cna_scores=bundle.obs[cfg.cna.canonical_column].to_numpy(dtype=float),
        random_state=cfg.programs.umap_random_state,
    )
    _write_json(
        logs_dir / "stage_4_artifacts.json",
        {
            "bridge_checkpoint_path": str(bridge_ckpt.resolve()),
            "umap_figure_paths": stage4_umap_paths,
            "transport_sanity": stage5_result.transport_sanity,
            "training_summary": model.training_summary.to_dict(),
            "transport_backend": selected_backend,
            "comparison_available": bool(linear_shadow_transported is not None),
        },
    )

    stage5_log = logs_dir / "stage_5_perturbation_extraction.log"
    s5_logger = logging.getLogger("omega_spatial.stage5_pipeline")
    s5_logger.setLevel(logging.INFO)
    s5_logger.handlers.clear()
    fh = logging.FileHandler(stage5_log, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    s5_logger.addHandler(fh)
    s5_logger.info("Model inference entry point: %s", stage5_result.inference_entry_point)
    s5_logger.info("Input observed shape: %s, transported shape: %s", bundle.expr.shape, transported.shape)
    s5_logger.info("Perturbation norm stats: %s", json.dumps(stage5_result.diagnostics.get("perturbation_norm_stats", {})))
    s5_logger.info("Class summaries: %s", json.dumps(stage5_result.diagnostics.get("class_perturbation_norm_summaries", {})))
    s5_logger.info("Transport sanity: %s", json.dumps(stage5_result.transport_sanity))
    s5_logger.info("Stage 5 matrix artifact paths: %s", json.dumps(stage5_paths, indent=2))
    for w in stage5_result.diagnostics.get("warnings", []) or []:
        s5_logger.warning("%s", w)

    umap_paths: list[str] = []
    stage5_summary_paths: list[str] = []
    try:
        labs = bundle.obs["marginal_label"].to_numpy()
        umap_paths = write_stage5_umap_figures(
            bundle.expr,
            transported,
            perturb,
            labs,
            figd,
            s5_logger,
            normal_reference=model.normal_reference,
        )
        for p in umap_paths:
            s5_logger.info("UMAP artifact path: %s", p)
    except Exception as ex:
        s5_logger.error("Stage 5 UMAP generation failed: %s", ex)
    try:
        stage5_summary_paths = write_stage5_summary_figures(
            stage5_result.obs,
            np.linalg.norm(perturb, axis=1),
            figd,
            s5_logger,
        )
    except Exception as ex:
        s5_logger.error("Stage 5 summary figure generation failed: %s", ex)

    annotated_paths = {
        "annotated_output_obs_csv": str((out_dir / "annotated_output_obs.csv").resolve()),
        "annotated_output_h5ad": str((out_dir / "annotated_output.h5ad").resolve()),
        "perturbation_vectors_csv": str((out_dir / "perturbation_vectors.csv").resolve()),
    }
    write_stage5_artifact_manifest(
        logs_dir / "stage_5_artifacts.json",
        out_dir,
        perturbation_paths=stage5_paths,
        annotated_paths=annotated_paths,
        umap_paths=umap_paths,
        summary_figure_paths=stage5_summary_paths,
        test_script_path=str((_REPO_ROOT / "tests/test_stage_5_perturbation_extraction.py").resolve()),
        known_limitations=[
            "Perturbations are defined for all spots; filter by marginal_label for tumor-focused analyses.",
            "Transport is Euler integration of the selected Stage 4 backend drift field; not a calibrated causal counterfactual.",
            "Large Visium exports (wide CSV) can be heavy; prefer stage_5_expression_bundle.npz for array I/O.",
        ],
        extra={
            "perturbation_identity": "perturbation = transported_state - observed_state",
            "pipeline_run": True,
        },
    )
    s5_logger.info("Wrote artifact manifest: %s", logs_dir / "stage_5_artifacts.json")

    stage6_log = logs_dir / "stage_6_program_discovery.log"
    s6_logger = logging.getLogger("omega_spatial.stage6_pipeline")
    s6_logger.setLevel(logging.INFO)
    s6_logger.handlers.clear()
    s6_fh = logging.FileHandler(stage6_log, encoding="utf-8")
    s6_fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    s6_logger.addHandler(s6_fh)
    discovery = run_program_discovery(perturb, bundle.var_names, cfg)
    program_artifact_paths = save_program_artifacts(out_dir, discovery, obs=stage5_result.obs)
    prog_scores = discovery.nmf_scores
    prog_loadings = discovery.nmf_loadings
    prog_model_selection = discovery.model_selection
    prog_scores.to_csv(out_dir / "gene_programs.csv", index=False)
    prog_loadings.to_csv(out_dir / "gene_program_loadings.csv", index=True)
    prog_model_selection.to_csv(out_dir / "nmf_method_comparison.csv", index=False)
    stage6_umap_paths = write_stage6_umap_figures(
        perturb,
        discovery,
        figd,
        s6_logger,
        cfg,
        obs=stage5_result.obs,
    )
    stage6_summary_paths = write_stage6_summary_figures(
        discovery,
        figd,
        s6_logger,
        obs=stage5_result.obs,
    )
    write_stage6_artifact_manifest(
        logs_dir / "stage_6_artifacts.json",
        artifact_paths=program_artifact_paths,
        umap_paths=stage6_umap_paths + stage6_summary_paths,
        test_script_path=str((_REPO_ROOT / "tests/test_stage_6_program_discovery.py").resolve()),
        known_limitations=[
            "Programs are inferred from perturbation vectors, not observed expression directly.",
            "NMF/PCA/ICA comparisons are internal decomposition diagnostics rather than held-out biology tests.",
        ],
        extra={"pipeline_run": True, "chosen_method": discovery.chosen_method},
    )

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
    spatial_methods: dict[str, np.ndarray] = {}
    if selected_backend == "neural":
        spatial_methods["SpatialBridge_neural"] = transported
        if linear_shadow_transported is not None:
            spatial_methods["SpatialBridge_linear"] = linear_shadow_transported
    else:
        spatial_methods["SpatialBridge_linear"] = transported
    benchmark_df, baseline_counterfactuals = run_benchmarks_and_baselines(
        bundle.expr,
        transported,
        labels,
        section,
        cna,
        spatial_methods=spatial_methods,
    )
    benchmark_df.to_csv(out_dir / "benchmark_metrics.csv", index=False)

    _save_annotated_output(out_dir, stage5_result.obs, perturb, bundle.var_names)

    stage7_manifest, stage7_html = run_stage7_reporting(
        repo_root=_REPO_ROOT,
        out_dir=out_dir,
        cfg=cfg,
        expr=bundle.expr,
        context=ctx,
        transported=transported,
        perturb=perturb,
        program_scores=prog_scores,
        obs=stage5_result.obs,
        section=section,
        cna=cna,
        benchmark_df=benchmark_df,
        baseline_counterfactuals=baseline_counterfactuals,
        knn_indices=knn,
        transport_backend=selected_backend,
        linear_transported=linear_shadow_transported if linear_shadow_transported is not None else transported,
        neural_transported=transported if selected_backend == "neural" else None,
        program_loadings=prog_loadings,
    )

    stage8_manifest = run_stage8_heldout_validation(
        repo_root=_REPO_ROOT,
        out_dir=out_dir,
        expr=bundle.expr,
        transported=transported,
        obs=stage5_result.obs,
        section=section,
        cna=cna,
        split_seed=cfg.train.random_seed,
        backend_transports={
            "SpatialBridge_selected": transported,
            **({"SpatialBridge_linear": linear_shadow_transported} if linear_shadow_transported is not None else {}),
            **({"SpatialBridge_neural": transported} if selected_backend == "neural" else {}),
        },
    )
    stage9_manifest = run_stage9_cross_modal_validation(
        repo_root=_REPO_ROOT,
        out_dir=out_dir,
        obs=stage5_result.obs,
        backend_names=sorted(k for k in baseline_counterfactuals if k.startswith("SpatialBridge")),
    )

    perturb_mag = np.linalg.norm(perturb, axis=1)
    make_figures(out_dir, stage5_result.obs, perturb_mag, prog_scores, prog_loadings)
    report_figs = [str(p.resolve()) for p in sorted((out_dir / "figures").glob("*.png"))]
    readiness_df = pd.read_csv(out_dir / "dataset_readiness.csv")
    qc_df = pd.read_csv(out_dir / "qc_summary.csv")
    malignancy_df = pd.read_csv(out_dir / "malignancy_scoring_summary.csv")
    malignancy_counts_df = pd.read_csv(out_dir / "malignancy_counts_by_section.csv")
    write_reports(
        out_dir=out_dir,
        run_name=cfg.run_name,
        readiness=readiness_df,
        qc_summary=qc_df,
        benchmark=benchmark_df,
        programs=prog_loadings,
        loadings=prog_loadings,
        figures=report_figs,
        malignancy_summary=malignancy_df,
        malignancy_counts=malignancy_counts_df,
        stage7_html=stage7_html,
        report_cfg=cfg.report,
    )

    _write_json(
        logs_dir / "stage_complete.json",
        {
            "outputs": [
                "annotated_output.h5ad",
                "perturbation_vectors.csv",
                "stage_5_expression_bundle.npz",
                "transported_expression.csv",
                "perturbation_matrix.csv",
                "stage_5_spot_summary.csv",
                "gene_programs.csv",
                "pathway_enrichment.csv",
                "best_term_per_program.csv",
                "benchmark_metrics.csv",
                "stage_7_spatial_coherence.csv",
                "stage_7_biological_plausibility.csv",
                "stage_7_stage4b_program_gain_attribution.csv",
                "stage_7_stage4b_pathway_gain_attribution.csv",
                str(stage7_manifest.get("artifact_manifest_path", "logs/stage_7_artifacts.json")),
                "stage_8_heldout_metrics.csv",
                str(stage8_manifest.get("artifact_manifest_path", "logs/stage_8_artifacts.json")),
                "stage_9_cross_modal_metrics.csv",
                str(stage9_manifest.get("artifact_manifest_path", "logs/stage_9_artifacts.json")),
                "qc_summary.csv",
                "malignancy_scoring_summary.csv",
                "malignancy_counts_by_section.csv",
                "figures/",
                "report.html",
                "report.pdf",
                "run_config.yaml",
            ]
            + [
                str(stage8_manifest.get("metrics_artifact_paths", {}).get("heldout_metrics_csv", "")),
                str(stage9_manifest.get("metrics_artifact_paths", {}).get("cross_modal_metrics_csv", "")),
            ]
        },
    )
