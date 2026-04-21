from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass
class SpatialNeighborhoodDiagnostics:
    """Summary statistics from section-restricted spatial graph construction."""

    n_spots: int
    n_genes: int
    k_neighbors: int
    x_column: str
    y_column: str
    section_column: str
    per_section_spot_counts: dict[str, int]
    per_section_effective_k: dict[str, int]
    isolated_spot_indices: list[int]
    duplicate_coordinate_spot_indices: list[int]
    n_duplicate_coord_spots: int
    warnings: list[str]


@dataclass
class SpatialNeighborhoodResult:
    """Section-restricted kNN graph and aligned neighbor-mean context (Stage 3 handoff)."""

    knn_indices: np.ndarray
    neighbor_distances: np.ndarray
    context_matrix: np.ndarray
    mean_neighbor_distance: np.ndarray
    local_density: np.ndarray
    section_neighbor_purity: np.ndarray
    diagnostics: SpatialNeighborhoodDiagnostics


@dataclass
class IngestionDiagnostics:
    """Structured Stage 1 join and schema diagnostics (also mirrored in obs.attrs['omega_ingestion'])."""

    join_keys_attempted: list[str] = field(default_factory=list)
    join_key_used: str = ""
    metadata_path: str | None = None
    metadata_rows_matched: int = 0
    metadata_match_rate: float = 0.0
    samples_loaded: list[str] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)
    missing_metadata_columns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    per_sample_match_rates: dict[str, float] = field(default_factory=dict)
