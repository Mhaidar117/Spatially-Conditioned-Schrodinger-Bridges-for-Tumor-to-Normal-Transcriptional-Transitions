from __future__ import annotations

import numpy as np
import pandas as pd

from .cna import assign_within_section_marginals
from .config import PipelineConfig
from .types import DatasetBundle


def _ensure_sections(obs: pd.DataFrame, section_column: str) -> pd.Series:
    if section_column not in obs.columns:
        obs[section_column] = "section_0"
    return obs[section_column].astype(str)


def assign_marginals(bundle: DatasetBundle, cfg: PipelineConfig) -> DatasetBundle:
    obs = bundle.obs.copy()
    expr = bundle.expr
    _ensure_sections(obs, cfg.state.section_column)
    score_col = cfg.cna.canonical_column
    if score_col not in obs.columns:
        raise ValueError(
            f"Cannot assign marginals without malignancy score. Expected '{cfg.cna.canonical_column}' in obs."
        )

    obs, _, _ = assign_within_section_marginals(obs, cfg)
    return DatasetBundle(
        expr=expr,
        obs=obs,
        var_names=bundle.var_names,
        source_path=bundle.source_path,
        dataset_kind=bundle.dataset_kind,
    )


def split_by_section(bundle: DatasetBundle, cfg: PipelineConfig) -> dict[str, np.ndarray]:
    sec = bundle.obs[cfg.state.section_column].astype(str) if cfg.state.section_column in bundle.obs.columns else pd.Series(["section_0"] * bundle.expr.shape[0])
    unique = sorted(sec.unique())
    if len(unique) < 3:
        all_idx = np.arange(len(sec))
        return {"train": all_idx, "val": all_idx[: max(1, len(all_idx) // 5)], "test": all_idx[: max(1, len(all_idx) // 5)]}
    n = len(unique)
    train_sec = set(unique[: int(0.6 * n)])
    val_sec = set(unique[int(0.6 * n) : int(0.8 * n)])
    test_sec = set(unique[int(0.8 * n) :])
    return {
        "train": np.where(sec.isin(train_sec))[0],
        "val": np.where(sec.isin(val_sec))[0],
        "test": np.where(sec.isin(test_sec))[0],
    }
