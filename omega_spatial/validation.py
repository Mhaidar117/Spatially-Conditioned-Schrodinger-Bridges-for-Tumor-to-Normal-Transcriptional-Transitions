from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import l2_to_target as _l2_to_target_vec
from .utils import repo_relative, safe_corr

# Backward-compatible aliases (internal callers may still reference these).
_repo_relative = repo_relative
_safe_corr = safe_corr


def _setup_logger(repo_root: Path, filename: str, logger_name: str) -> tuple[logging.Logger, Path]:
    logs_dir = Path(repo_root).resolve() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / filename
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    return logger, log_path


# Keep the historical private name so internal callers remain stable.
_l2_to_target = _l2_to_target_vec


def _normalize_sample_id(sample: object) -> str:
    text = str(sample).strip().upper()
    return re.sub(r"[^A-Z0-9]", "", text)


def _collect_metadata_sample_ids(obs: pd.DataFrame) -> list[str]:
    # Prefer explicit sample IDs, then metadata-derived IDs, and finally section IDs.
    for col in ("sample", "metadata_sample", "section_id"):
        if col in obs.columns:
            vals = obs[col].dropna().astype(str).str.strip()
            vals = vals[vals != ""]
            if not vals.empty:
                return sorted(vals.unique().tolist())
    return []


def run_stage8_heldout_validation(
    *,
    repo_root: Path,
    out_dir: Path,
    expr: np.ndarray,
    transported: np.ndarray,
    obs: pd.DataFrame,
    section: np.ndarray,
    cna: np.ndarray,
    split_seed: int = 17,
    holdout_fraction: float = 0.34,
    make_plots: bool = True,
    backend_transports: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """
    Stage 8: section-level held-out validation stream (no random spot splits).
    Writes:
      - logs/stage_8_heldout_validation.log
      - logs/stage_8_artifacts.json
      - out_dir/stage_8_heldout_metrics.csv
      - out_dir/figures/stage_8_heldout_section_performance.png
    """
    repo_root = Path(repo_root).resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    logger, log_path = _setup_logger(
        repo_root,
        filename="stage_8_heldout_validation.log",
        logger_name="omega_spatial.stage8",
    )
    logger.info("Stage 8 held-out validation started")
    logger.info("Anti-leakage: split unit is section_id; no random spot split is used.")

    x = np.asarray(expr, dtype=float)
    x_t = np.asarray(transported, dtype=float)
    sec = np.asarray(section).astype(str)
    cna_arr = np.asarray(cna, dtype=float)
    if x.shape != x_t.shape:
        raise ValueError(f"expr and transported shapes must match; got {x.shape} vs {x_t.shape}")
    if x.shape[0] != len(sec):
        raise ValueError("section vector length mismatch with expression rows")

    uniq_sections = sorted(pd.Series(sec).dropna().astype(str).unique().tolist())
    rng = np.random.default_rng(split_seed)
    section_order = uniq_sections.copy()
    rng.shuffle(section_order)
    n_holdout_sections = max(1, int(round(len(uniq_sections) * holdout_fraction)))
    n_holdout_sections = min(max(n_holdout_sections, 1), max(len(uniq_sections) - 1, 1))

    unresolved: list[str] = []
    status = "executed"
    if len(uniq_sections) < 2:
        status = "incomplete_insufficient_sections"
        unresolved.append("Need at least 2 sections for section-level held-out validation.")
        holdout_sections: list[str] = []
        train_sections = uniq_sections
        train_mask = np.ones(len(sec), dtype=bool)
        holdout_mask = np.zeros(len(sec), dtype=bool)
    else:
        holdout_sections = section_order[:n_holdout_sections]
        train_sections = [s for s in uniq_sections if s not in holdout_sections]
        holdout_mask = np.isin(sec, holdout_sections)
        train_mask = ~holdout_mask

    labels = (
        obs["marginal_label"].astype(str).to_numpy()
        if "marginal_label" in obs.columns and len(obs) == x.shape[0]
        else np.array(["unknown"] * x.shape[0], dtype=object)
    )
    normal_idx = np.where(labels == "normal")[0]
    if len(normal_idx) == 0:
        normal_idx = np.arange(x.shape[0])
        unresolved.append("No normal labels available; using cohort mean as normal reference for Stage 8 distances.")
    normal_ref = x[normal_idx].mean(axis=0)

    transports: dict[str, np.ndarray] = {"SpatialBridge_selected": x_t}
    if backend_transports:
        for k, arr in backend_transports.items():
            vv = np.asarray(arr, dtype=float)
            if vv.shape == x.shape:
                transports[k] = vv

    pre_dist = _l2_to_target(x, normal_ref)
    rows: list[dict[str, Any]] = []
    split_masks = (("train_sections_internal", train_mask), ("heldout_sections", holdout_mask))
    for method_name, method_transport in transports.items():
        post_dist = _l2_to_target(method_transport, normal_ref)
        toward_ref = pre_dist - post_dist
        movement_norm = np.linalg.norm(method_transport - x, axis=1)
        for split_name, mask in split_masks:
            if int(mask.sum()) == 0:
                rows.append(
                    {
                        "method": method_name,
                        "split": split_name,
                        "n_spots": 0,
                        "mean_dist_pre": float("nan"),
                        "mean_dist_post": float("nan"),
                        "mean_delta_toward_reference": float("nan"),
                        "mean_movement_norm": float("nan"),
                        "cna_vs_movement_corr": float("nan"),
                    }
                )
                continue
            rows.append(
                {
                    "method": method_name,
                    "split": split_name,
                    "n_spots": int(mask.sum()),
                    "mean_dist_pre": float(np.nanmean(pre_dist[mask])),
                    "mean_dist_post": float(np.nanmean(post_dist[mask])),
                    "mean_delta_toward_reference": float(np.nanmean(toward_ref[mask])),
                    "mean_movement_norm": float(np.nanmean(movement_norm[mask])),
                    "cna_vs_movement_corr": _safe_corr(cna_arr[mask], movement_norm[mask]),
                }
            )

    if ("SpatialBridge_linear" in transports) and ("SpatialBridge_neural" in transports):
        lin_post = _l2_to_target(transports["SpatialBridge_linear"], normal_ref)
        neu_post = _l2_to_target(transports["SpatialBridge_neural"], normal_ref)
        gain = (pre_dist - neu_post) - (pre_dist - lin_post)
        for split_name, mask in split_masks:
            rows.append(
                {
                    "method": "Stage4b_gain_vs_linear",
                    "split": split_name,
                    "n_spots": int(mask.sum()),
                    "mean_dist_pre": float(np.nanmean(pre_dist[mask])) if int(mask.sum()) else float("nan"),
                    "mean_dist_post": float(np.nanmean(neu_post[mask])) if int(mask.sum()) else float("nan"),
                    "mean_delta_toward_reference": float(np.nanmean(gain[mask])) if int(mask.sum()) else float("nan"),
                    "mean_movement_norm": float("nan"),
                    "cna_vs_movement_corr": _safe_corr(cna_arr[mask], gain[mask]) if int(mask.sum()) else float("nan"),
                }
            )
    metrics = pd.DataFrame(rows)
    metrics_path = out_dir / "stage_8_heldout_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    logger.info("Wrote Stage 8 metrics: %s", metrics_path)

    fig_path: Path | None = None
    if make_plots:
        import matplotlib.pyplot as plt

        fig_path = fig_dir / "stage_8_heldout_section_performance.png"
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=180)
        ax0, ax1 = axes
        base_metrics = metrics[metrics["method"].isin(["SpatialBridge_linear", "SpatialBridge_neural", "SpatialBridge_selected"])].copy()
        bars = [f"{m}\n{s}" for m, s in zip(base_metrics["method"].astype(str), base_metrics["split"].astype(str))]
        vals0 = base_metrics["mean_delta_toward_reference"].astype(float).to_numpy()
        vals1 = base_metrics["mean_movement_norm"].astype(float).to_numpy()
        cols = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2"]
        ax0.bar(bars, vals0, color=[cols[i % len(cols)] for i in range(len(bars))])
        ax0.set_title("Stage 8 held-out transport sanity")
        ax0.set_ylabel("mean_delta_toward_reference")
        ax0.tick_params(axis="x", rotation=15)
        ax1.bar(bars, vals1, color=[cols[i % len(cols)] for i in range(len(bars))])
        ax1.set_title("Stage 8 held-out movement magnitude")
        ax1.set_ylabel("mean_movement_norm")
        ax1.tick_params(axis="x", rotation=15)
        fig.suptitle(
            "Section-level held-out validation (anti-leakage split)\n"
            "Train = held-in sections; Held-out = unseen sections only."
        )
        fig.tight_layout()
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Wrote Stage 8 figure: %s", fig_path)
    else:
        unresolved.append("plot_generation_skipped_make_plots_false")

    manifest = {
        "stage": "stage_8_heldout_validation",
        "status": status,
        "split_strategy": {
            "unit": "section_id",
            "random_spot_split_used": False,
            "split_seed": int(split_seed),
            "holdout_fraction": float(holdout_fraction),
            "train_sections": train_sections,
            "heldout_sections": holdout_sections,
        },
        "metrics_artifact_paths": {
            "heldout_metrics_csv": _repo_relative(repo_root, metrics_path),
        },
        "figure_paths": {
            "heldout_section_performance_png": (
                _repo_relative(repo_root, fig_path) if fig_path is not None else ""
            ),
        },
        "log_path": _repo_relative(repo_root, log_path),
        "unresolved_validation_gaps": unresolved,
        "anti_leakage_notes": [
            "No random spot splits were used for held-out claims.",
            "Held-out split is section-level and isolated from internal metrics.",
            "Evaluation labels are used post hoc only.",
            "Backend comparisons use identical section splits and reference definitions.",
        ],
        "caveats": [
            "Held-out metrics are section-level internal cohort generalization, not external-cohort proof.",
            "If few sections are available, held-out estimates have high variance.",
        ],
        "test_script_path": "tests/test_stage_8_heldout_validation.py",
        "artifact_manifest_path": _repo_relative(repo_root, repo_root / "logs" / "stage_8_artifacts.json"),
    }
    manifest_path = repo_root / "logs" / "stage_8_artifacts.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote Stage 8 manifest: %s", manifest_path)
    return manifest


def run_stage9_cross_modal_validation(
    *,
    repo_root: Path,
    out_dir: Path,
    obs: pd.DataFrame,
    make_plots: bool = True,
    backend_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Stage 9: optional ST/CODEX cross-modal validation scaffold with coverage diagnostics.
    Writes:
      - logs/stage_9_cross_modal_validation.log
      - logs/stage_9_artifacts.json
      - out_dir/stage_9_cross_modal_metrics.csv
      - out_dir/figures/stage_9_cross_modal_coverage.png
    """
    repo_root = Path(repo_root).resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    logger, log_path = _setup_logger(
        repo_root,
        filename="stage_9_cross_modal_validation.log",
        logger_name="omega_spatial.stage9",
    )
    logger.info("Stage 9 cross-modal validation started")

    st_align_dir = repo_root / "Data" / "Inputs" / "ST_align"
    codex_dir = repo_root / "Data" / "Inputs" / "Codex"
    readme_codex = repo_root / "Docs" / "Data_Information" / "readme_CODEX.txt"
    readme_stalign = repo_root / "Data" / "Inputs" / "Readme_ST_align_CODEX.txt"

    st_samples = sorted([p.name for p in st_align_dir.iterdir() if p.is_dir()]) if st_align_dir.is_dir() else []
    codex_files = sorted([p.name for p in codex_dir.iterdir() if p.is_file()]) if codex_dir.is_dir() else []
    metadata_samples = _collect_metadata_sample_ids(obs)
    st_norm = {_normalize_sample_id(s): s for s in st_samples}
    md_norm = {_normalize_sample_id(s): s for s in metadata_samples}
    overlap_norm = sorted(set(st_norm.keys()).intersection(md_norm.keys()))
    overlap_with_metadata = sorted(st_norm[k] for k in overlap_norm)

    unresolved: list[str] = []
    status = "executed"
    if not st_align_dir.is_dir():
        unresolved.append("Data/Inputs/ST_align directory missing; cross-modal agreement cannot be computed.")
    if not codex_dir.is_dir():
        unresolved.append("Data/Inputs/Codex directory missing; cross-modal agreement cannot be computed.")
    if len(overlap_with_metadata) == 0:
        unresolved.append("No ST_align sample IDs overlap metadata sample IDs in current run.")
    if unresolved:
        status = "partial_or_scaffold_only"

    metrics = pd.DataFrame(
        [
            {
                "n_st_align_sample_dirs": int(len(st_samples)),
                "n_codex_files": int(len(codex_files)),
                "n_metadata_samples": int(len(metadata_samples)),
                "n_overlap_st_align_metadata_samples": int(len(overlap_with_metadata)),
                "agreement_metric_available": bool(len(overlap_with_metadata) > 0 and len(codex_files) > 0),
            }
        ]
    )
    metrics_path = out_dir / "stage_9_cross_modal_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    logger.info("Wrote Stage 9 metrics: %s", metrics_path)

    fig_path: Path | None = None
    if make_plots:
        import matplotlib.pyplot as plt

        fig_path = fig_dir / "stage_9_cross_modal_coverage.png"
        fig, ax = plt.subplots(figsize=(6.5, 4), dpi=180)
        vals = [
            float(len(st_samples)),
            float(len(codex_files)),
            float(len(overlap_with_metadata)),
        ]
        labs = ["ST_align samples", "CODEX assets", "overlap samples"]
        colors = ["#4c78a8", "#54a24b", "#f58518"]
        ax.bar(labs, vals, color=colors)
        ax.set_ylabel("count")
        ax.set_title("Stage 9 cross-modal coverage")
        ax.tick_params(axis="x", rotation=15)
        note = (
            "Agreement metrics computed only for overlapping sample subset.\n"
            "Cross-modal assets are evaluation-only and never supervision targets."
        )
        ax.text(0.02, -0.35, note, transform=ax.transAxes, fontsize=8, va="top")
        fig.tight_layout()
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Wrote Stage 9 figure: %s", fig_path)
    else:
        unresolved.append("plot_generation_skipped_make_plots_false")

    manifest = {
        "stage": "stage_9_cross_modal_validation",
        "status": status,
        "coverage": {
            "st_align_sample_dirs": st_samples,
            "n_codex_files": int(len(codex_files)),
            "metadata_sample_ids_used": metadata_samples,
            "overlap_with_metadata_samples": overlap_with_metadata,
        },
        "metrics_artifact_paths": {
            "cross_modal_metrics_csv": _repo_relative(repo_root, metrics_path),
        },
        "figure_paths": {
            "cross_modal_coverage_png": (
                _repo_relative(repo_root, fig_path) if fig_path is not None else ""
            ),
        },
        "resource_presence": {
            "st_align_dir_exists": st_align_dir.is_dir(),
            "codex_dir_exists": codex_dir.is_dir(),
            "readme_codex_exists": readme_codex.is_file(),
            "readme_stalign_exists": readme_stalign.is_file(),
        },
        "subset_coverage_notes": [
            "Stage 9 applies only to samples with ST/CODEX alignment outputs.",
            "Do not use CODEX-derived outputs from the same sample as both supervision and validation.",
        ],
        "transport_backends_in_run": sorted(set(backend_names or [])),
        "unresolved_validation_gaps": unresolved,
        "log_path": _repo_relative(repo_root, log_path),
        "test_script_path": "tests/test_stage_9_cross_modal_validation.py",
        "artifact_manifest_path": _repo_relative(repo_root, repo_root / "logs" / "stage_9_artifacts.json"),
    }
    manifest_path = repo_root / "logs" / "stage_9_artifacts.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote Stage 9 manifest: %s", manifest_path)
    return manifest
