"""
Validation script for strict true-CNA integration (Stage 1 -> Stage 2).

Run:
  python tests/test_true_cna_pipeline.py
Or:
  pytest tests/test_true_cna_pipeline.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omega_spatial.cna import run_marginal_definition  # noqa: E402
from omega_spatial.config import PipelineConfig  # noqa: E402
from omega_spatial.io import load_dataset, resolve_dataset  # noqa: E402


def _cfg(tmp_path: Path) -> PipelineConfig:
    cfg = PipelineConfig(input_path=str(REPO_ROOT), output_path=str(tmp_path / "out"))
    cfg.cna.require_true_score = True
    cfg.cna.infer_if_missing = True
    cfg.cna.true_score_rds_path = "Data/Inputs/CNA/mal_lev.rds"
    cfg.cna.true_score_regions_path = "Data/Inputs/CNA/samples_regions.txt"
    cfg.state.section_column = "section_id"
    cfg.train.random_seed = 7
    return cfg


def test_true_cna_stage1_to_stage2(tmp_path: Path) -> None:
    gbm = REPO_ROOT / "Data" / "Inputs" / "general" / "GBM_data"
    if not gbm.is_dir():
        raise AssertionError("Expected Data/Inputs/general/GBM_data for true-CNA validation.")
    cfg = _cfg(tmp_path)
    resolved, kind = resolve_dataset(str(gbm))
    bundle = load_dataset(resolved, kind, cfg=cfg)
    assert "cna_score" in bundle.obs.columns, "Stage 1 did not attach canonical cna_score."
    assert np.isfinite(bundle.obs["cna_score"].to_numpy(dtype=float)).sum() > 0, "No numeric true CNA scores after Stage 1 join."
    res = run_marginal_definition(bundle, cfg, tmp_path / "stage2")
    assert res.source.startswith("provided"), f"Expected precomputed source, got: {res.source}"
    assert "fallback_expression_program_hvg_fallback" not in " ".join(res.decision_path)
    assert "expression_program_hvg_fallback" not in res.decision_path
    assert set(res.bundle.obs["marginal_label"].unique()) <= {"tumor", "normal", "intermediate"}


def test_true_cna_hard_fail_when_missing_in_strict_mode(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    n = 24
    expr = np.random.default_rng(0).poisson(4.0, size=(n, 60)).astype(float)
    import pandas as pd
    from omega_spatial.types import DatasetBundle

    obs = pd.DataFrame({"section_id": ["S"] * n, "x": np.arange(n), "y": np.arange(n)})
    bundle = DatasetBundle(
        expr=expr,
        obs=obs,
        var_names=[f"g{i}" for i in range(expr.shape[1])],
        source_path=Path("synthetic"),
        dataset_kind="synthetic",
    )
    try:
        run_marginal_definition(bundle, cfg, tmp_path / "strict_fail")
    except ValueError as exc:
        assert "True CNA score is required" in str(exc)
    else:
        raise AssertionError("Expected strict true-CNA mode to fail when true score is absent.")


def main() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_true_cna_hard_fail_when_missing_in_strict_mode(tmp)
        print("OK: strict mode hard-fails when true CNA missing")
        test_true_cna_stage1_to_stage2(tmp)
        print("OK: Stage 1->2 true CNA path")


if __name__ == "__main__":
    main()
