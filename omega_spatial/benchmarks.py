from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def _l2_to_target(x: np.ndarray, target_mu: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(x - target_mu, axis=1)))


def _normal_reference_expr(expr: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (normal_indices, target_mu row) for distance-to-normal metrics."""
    lab = np.asarray(labels).astype(str)
    normal_idx = np.where(lab == "normal")[0]
    if len(normal_idx) == 0:
        normal_idx = np.arange(expr.shape[0])
    target_mu = expr[normal_idx].mean(axis=0, keepdims=True)
    return normal_idx, target_mu


def de_shift_counterfactual(expr: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Differential-expression-style shift: add (normal_mean - tumor_mean) to tumor spots only.
    """
    x = np.asarray(expr, dtype=float)
    lab = np.asarray(labels).astype(str)
    tumor_m = lab == "tumor"
    normal_m = lab == "normal"
    out = x.copy()
    if not tumor_m.any() or not normal_m.any():
        return out
    tumor_mean = x[tumor_m].mean(axis=0, keepdims=True)
    normal_mean = x[normal_m].mean(axis=0, keepdims=True)
    delta = normal_mean - tumor_mean
    out[tumor_m] = x[tumor_m] + delta
    return out


def latent_nn_normal_blend_counterfactual(
    expr: np.ndarray,
    labels: np.ndarray,
    *,
    k_neighbors: int = 8,
    blend: float = 0.5,
) -> np.ndarray:
    """
    Simple latent heuristic: each spot blended toward the mean of k nearest *normal* spots in expression space.
    Non-tumor spots unchanged; uses sklearn NearestNeighbors on normal subset only.
    """
    x = np.asarray(expr, dtype=float)
    lab = np.asarray(labels).astype(str)
    normal_idx = np.where(lab == "normal")[0]
    out = x.copy()
    if len(normal_idx) < 2:
        return out
    k = max(1, min(k_neighbors, len(normal_idx)))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(x[normal_idx])
    dist, ind = nn.kneighbors(x)
    neigh_global = normal_idx[ind]
    neigh_mean = x[neigh_global].mean(axis=1)
    tumor_m = lab == "tumor"
    if tumor_m.any():
        out[tumor_m] = (1.0 - blend) * x[tumor_m] + blend * neigh_mean[tumor_m]
    return out


def compute_baseline_counterfactuals(
    expr: np.ndarray,
    transported: np.ndarray,
    labels: np.ndarray,
    *,
    spatial_methods: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """
    Gene-space counterfactuals for benchmarking and Stage-7 UMAP comparisons.
    Keys align with benchmark table method names where applicable.
    """
    x = np.asarray(expr, dtype=float)
    lab = np.asarray(labels).astype(str)
    _, target_mu = _normal_reference_expr(x, lab)

    baseline_shift = x + (target_mu - x.mean(axis=0, keepdims=True))
    non_spatial = x + 0.5 * (target_mu - x)

    out = {
        "observed": x,
        "SpatialBridge": np.asarray(transported, dtype=float),
        "StaticOT_centroid": baseline_shift,
        "UnconditionalBridge": non_spatial,
        "DE_shift": de_shift_counterfactual(x, lab),
        "LatentNN_normal_blend": latent_nn_normal_blend_counterfactual(x, lab),
    }
    if spatial_methods:
        for k, v in spatial_methods.items():
            vv = np.asarray(v, dtype=float)
            if vv.shape == x.shape:
                out[k] = vv
    return out


def run_benchmarks_and_baselines(
    expr: np.ndarray,
    counterfactual: np.ndarray,
    labels: np.ndarray,
    section: np.ndarray,
    cna: np.ndarray,
    *,
    spatial_methods: dict[str, np.ndarray] | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """
    Extended benchmarks: spatial bridge, documented baseline families, DE proxy, CNA summary.
    Returns (metrics table, counterfactual dict for visualization).
    """
    baselines = compute_baseline_counterfactuals(
        expr,
        counterfactual,
        labels,
        spatial_methods=spatial_methods,
    )
    _, target_mu = _normal_reference_expr(expr, labels)

    rows: list[dict[str, object]] = []
    methods_for_distance = [
        ("SpatialBridge", baselines["SpatialBridge"]),
        ("StaticOT_centroid", baselines["StaticOT_centroid"]),
        ("UnconditionalBridge", baselines["UnconditionalBridge"]),
        ("DE_shift", baselines["DE_shift"]),
        ("LatentNN_normal_blend", baselines["LatentNN_normal_blend"]),
    ]
    for method_name in ("SpatialBridge_linear", "SpatialBridge_neural"):
        if method_name in baselines:
            methods_for_distance.insert(0, (method_name, baselines[method_name]))

    for method_name, sample in methods_for_distance:
        movement = np.linalg.norm(sample - expr, axis=1)
        dist_pre = np.linalg.norm(expr - target_mu, axis=1)
        dist_post = np.linalg.norm(sample - target_mu, axis=1)
        toward_ref = dist_pre - dist_post
        rows.append(
            {
                "method": method_name,
                "metric": "distance_to_normal_mean_l2",
                "value": _l2_to_target(sample, target_mu),
                "split_scope": "spot_level_internal",
            }
        )
        rows.append(
            {
                "method": method_name,
                "metric": "movement_norm_mean_l2",
                "value": float(np.mean(movement)),
                "split_scope": "spot_level_internal",
            }
        )
        rows.append(
            {
                "method": method_name,
                "metric": "mean_delta_toward_normal_reference",
                "value": float(np.mean(toward_ref)),
                "split_scope": "spot_level_internal",
            }
        )
        rows.append(
            {
                "method": method_name,
                "metric": "collapse_fraction_near_normal_reference",
                "value": float(np.mean(dist_post <= max(float(np.median(dist_post)) * 0.05, 1e-6))),
                "split_scope": "spot_level_internal",
            }
        )
        inter_idx = np.where(np.asarray(labels).astype(str) == "intermediate")[0]
        if len(inter_idx) > 0:
            rows.append(
                {
                    "method": method_name,
                    "metric": "intermediate_state_preservation_mean_movement_l2",
                    "value": float(np.mean(movement[inter_idx])),
                    "split_scope": "spot_level_internal",
                }
            )

    if ("SpatialBridge_linear" in baselines) and ("SpatialBridge_neural" in baselines):
        lin = np.asarray(baselines["SpatialBridge_linear"], dtype=float)
        neu = np.asarray(baselines["SpatialBridge_neural"], dtype=float)
        lin_post = np.linalg.norm(lin - target_mu, axis=1)
        neu_post = np.linalg.norm(neu - target_mu, axis=1)
        lin_gain = np.linalg.norm(expr - target_mu, axis=1) - lin_post
        neu_gain = np.linalg.norm(expr - target_mu, axis=1) - neu_post
        gain_delta = neu_gain - lin_gain
        rows.append(
            {
                "method": "Stage4b_gain_vs_linear",
                "metric": "mean_delta_toward_normal_reference",
                "value": float(np.mean(gain_delta)),
                "split_scope": "spot_level_internal",
            }
        )
        lin_mean = float(np.mean(lin_gain))
        neu_mean = float(np.mean(neu_gain))
        rows.append(
            {
                "method": "Stage4b_gain_vs_linear",
                "metric": "percent_gain_vs_linear",
                "value": float(((neu_mean - lin_mean) / max(abs(lin_mean), 1e-8)) * 100.0),
                "split_scope": "spot_level_internal",
            }
        )

    for method_name, sample in methods_for_distance:
        per_section: list[float] = []
        for sec in np.unique(section):
            idx = np.where(section == sec)[0]
            if len(idx) == 0:
                continue
            per_section.append(_l2_to_target(sample[idx], target_mu))
        rows.append(
            {
                "method": f"{method_name}_macro_by_section",
                "metric": "distance_to_normal_mean_l2",
                "value": float(np.mean(per_section)) if per_section else float("nan"),
                "split_scope": "section_macro_internal",
            }
        )

    tumor_idx = np.where(labels == "tumor")[0]
    normal_idx = np.where(labels == "normal")[0]
    if len(tumor_idx) > 0 and len(normal_idx) > 0:
        de_effect = float(
            np.mean(np.abs(expr[tumor_idx].mean(axis=0) - expr[normal_idx].mean(axis=0)))
        )
    else:
        de_effect = float("nan")
    rows.append(
        {
            "method": "DE_proxy_mean_abs_diff",
            "metric": "mean_abs_tumor_normal_expression_diff",
            "value": de_effect,
            "split_scope": "spot_level_descriptive",
        }
    )
    rows.append(
        {
            "method": "CNA_mean",
            "metric": "mean_cna_or_malignancy_score",
            "value": float(np.nanmean(cna)),
            "split_scope": "spot_level_descriptive",
        }
    )

    return pd.DataFrame(rows), baselines


def run_benchmarks(
    expr: np.ndarray,
    counterfactual: np.ndarray,
    labels: np.ndarray,
    section: np.ndarray,
    cna: np.ndarray,
) -> pd.DataFrame:
    """Backward-compatible: returns only the metrics table (two-column layout via method + value)."""
    df, _ = run_benchmarks_and_baselines(expr, counterfactual, labels, section, cna)
    return df


def spatial_coherence_summary(
    perturb_norm: np.ndarray,
    knn_indices: np.ndarray | None,
) -> pd.DataFrame:
    """
    Spatial coherence of perturbation magnitude vs local neighborhood mean (evaluation).
    knn_indices: shape (n_spots, k) global indices; if None, metrics are skipped.
    """
    if knn_indices is None or knn_indices.size == 0:
        return pd.DataFrame(
            [
                {
                    "metric": "neighbor_mean_perturb_norm_correlation",
                    "value": float("nan"),
                    "note": "missing knn_indices; skipped",
                }
            ]
        )
    p = np.asarray(perturb_norm, dtype=float).ravel()
    knn = np.asarray(knn_indices, dtype=np.int64)
    neigh_mean = np.array([p[knn[i]].mean() for i in range(len(p))], dtype=float)
    mask = np.isfinite(p) & np.isfinite(neigh_mean)
    if mask.sum() < 3:
        r = float("nan")
    else:
        r = float(np.corrcoef(p[mask], neigh_mean[mask])[0, 1])
    return pd.DataFrame(
        [
            {
                "metric": "neighbor_mean_perturb_norm_correlation",
                "value": r,
                "note": "Moran-like: corr(||u||, mean_neighbor ||u||); section-aware kNN expected upstream",
            }
        ]
    )


def biological_plausibility_summary(
    obs: pd.DataFrame,
    perturb_norm: np.ndarray,
    cna_values: np.ndarray,
    *,
    cna_column: str = "cna_score",
) -> pd.DataFrame:
    """Stratified summaries by marginal_label (weak supervision); CNA vs perturbation (evaluation)."""
    p = np.asarray(perturb_norm, dtype=float).ravel()
    cna = np.asarray(cna_values, dtype=float).ravel()
    rows: list[dict[str, object]] = []
    if "marginal_label" in obs.columns and len(obs) == len(p):
        g = obs["marginal_label"].astype(str).to_numpy()
        for lab in np.unique(g):
            m = g == lab
            rows.append(
                {
                    "stratum": "marginal_label",
                    "label": lab,
                    "n_spots": int(m.sum()),
                    "mean_perturbation_norm": float(np.nanmean(p[m])),
                    "mean_cna": float(np.nanmean(cna[m])) if len(cna) == len(p) else float("nan"),
                }
            )
    if len(cna) == len(p) and np.isfinite(cna).sum() > 3:
        mask = np.isfinite(cna) & np.isfinite(p)
        r = float(np.corrcoef(cna[mask], p[mask])[0, 1]) if mask.sum() > 3 else float("nan")
        rows.append(
            {
                "stratum": "global",
                "label": "cna_vs_perturbation_norm_pearson",
                "n_spots": int(mask.sum()),
                "mean_perturbation_norm": float("nan"),
                "mean_cna": r,
            }
        )
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["stratum", "label", "n_spots", "mean_perturbation_norm", "mean_cna"])
