from __future__ import annotations

import numpy as np
import pandas as pd


def _l2_to_target(x: np.ndarray, target_mu: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(x - target_mu, axis=1)))


def run_benchmarks(
    expr: np.ndarray,
    counterfactual: np.ndarray,
    labels: np.ndarray,
    section: np.ndarray,
    cna: np.ndarray,
) -> pd.DataFrame:
    normal_idx = np.where(labels == "normal")[0]
    if len(normal_idx) == 0:
        normal_idx = np.arange(expr.shape[0])
    target_mu = expr[normal_idx].mean(axis=0, keepdims=True)

    baseline_shift = expr + (target_mu - expr.mean(axis=0, keepdims=True))
    non_spatial = expr + 0.5 * (target_mu - expr)

    rows = [
        {"method": "SpatialBridge", "distance_to_normal": _l2_to_target(counterfactual, target_mu)},
        {"method": "StaticOT_centroid", "distance_to_normal": _l2_to_target(baseline_shift, target_mu)},
        {"method": "UnconditionalBridge", "distance_to_normal": _l2_to_target(non_spatial, target_mu)},
    ]

    # Add imbalance-aware macro metric by section.
    for method_name, sample in [("SpatialBridge", counterfactual), ("StaticOT_centroid", baseline_shift), ("UnconditionalBridge", non_spatial)]:
        per_section = []
        for sec in np.unique(section):
            idx = np.where(section == sec)[0]
            if len(idx) == 0:
                continue
            per_section.append(_l2_to_target(sample[idx], target_mu))
        rows.append(
            {
                "method": f"{method_name}_macro_by_section",
                "distance_to_normal": float(np.mean(per_section)) if per_section else np.nan,
            }
        )

    # Differential-expression proxy score.
    tumor_idx = np.where(labels == "tumor")[0]
    if len(tumor_idx) > 0 and len(normal_idx) > 0:
        de_effect = np.mean(np.abs(expr[tumor_idx].mean(axis=0) - expr[normal_idx].mean(axis=0)))
    else:
        de_effect = float("nan")
    rows.append({"method": "DE_proxy_mean_abs_diff", "distance_to_normal": float(de_effect)})
    rows.append({"method": "CNA_mean", "distance_to_normal": float(np.nanmean(cna))})
    return pd.DataFrame(rows)
