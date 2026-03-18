from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class DatasetBundle:
    expr: np.ndarray
    obs: pd.DataFrame
    var_names: list[str]
    source_path: Path
    dataset_kind: str


@dataclass
class ReadinessReport:
    is_ready: bool
    dataset_kind: str
    n_sections: int
    n_spots: int
    n_genes: int
    spatial_columns: tuple[str, str] | None
    cna_column: str | None
    cna_source: str
    will_run_cna_inference: bool
    reference_normal_available: bool
    sufficient_for_pseudopairing: bool
    issues: list[str]
    recommendations: list[str]
