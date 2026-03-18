from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class QCConfig:
    min_counts_per_spot: float = 500.0
    min_genes_per_spot: int = 100
    top_hvg: int = 2000
    log1p: bool = True


@dataclass
class StateConfig:
    cna_column: str = "cna_score"
    low_quantile: float = 0.2
    high_quantile: float = 0.8
    section_column: str = "section_id"


@dataclass
class CNAConfig:
    canonical_column: str = "cna_score"
    aliases: list[str] = field(default_factory=lambda: ["cna_score", "malignancy_score", "cna", "malignancy"])
    infer_if_missing: bool = True
    gene_annotation_path: str = ""
    gene_id_column: str = "gene_id"
    chromosome_column: str = "chromosome"
    position_column: str = "position"
    reference_normal_path: str = ""
    smoothing_window: int = 101
    constant_score_std_threshold: float = 1e-6
    min_mapped_genes: int = 500
    min_spots_per_group: int = 25


@dataclass
class SpatialConfig:
    x_column: str = "x"
    y_column: str = "y"
    k_neighbors: int = 8
    spatial_smoothing_alpha: float = 0.3


@dataclass
class TrainConfig:
    steps: int = 250
    learning_rate: float = 1e-3
    batch_size: int = 1024
    random_seed: int = 7
    gradient_accumulation: int = 1
    mixed_precision: bool = True
    checkpoint_every_steps: int = 50


@dataclass
class ProgramConfig:
    max_components: int = 12
    chosen_components: int = 6
    random_seed: int = 7


@dataclass
class ReportConfig:
    title: str = "Omega Spatial Control Report"
    include_pdf: bool = True


@dataclass
class PipelineConfig:
    input_path: str = ""
    output_path: str = ""
    dataset_name: str = "auto"
    run_name: str = "omega_spatial_run"
    qc: QCConfig = field(default_factory=QCConfig)
    state: StateConfig = field(default_factory=StateConfig)
    cna: CNAConfig = field(default_factory=CNAConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    programs: ProgramConfig = field(default_factory=ProgramConfig)
    report: ReportConfig = field(default_factory=ReportConfig)


def _overlay_dataclass(dc: Any, overrides: dict[str, Any]) -> Any:
    for key, value in overrides.items():
        if not hasattr(dc, key):
            continue
        current = getattr(dc, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _overlay_dataclass(current, value)
        else:
            setattr(dc, key, value)
    return dc


def load_config(config_path: str | None, input_path: str, output_path: str) -> PipelineConfig:
    cfg = PipelineConfig(input_path=input_path, output_path=output_path)
    if config_path:
        with Path(config_path).open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        _overlay_dataclass(cfg, loaded)
        if not cfg.input_path:
            cfg.input_path = input_path
        if not cfg.output_path:
            cfg.output_path = output_path
    return cfg


def dump_config(cfg: PipelineConfig, path: str) -> None:
    with Path(path).open("w", encoding="utf-8") as fh:
        yaml.safe_dump(asdict(cfg), fh, sort_keys=False)
