"""
Stage 8: held-out section validation (dedicated stream, anti-leakage split).

Run:  python tests/test_stage_8_heldout_validation.py
Or:   pytest tests/test_stage_8_heldout_validation.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omega_spatial.validation import run_stage8_heldout_validation  # noqa: E402

LOG_PATH = REPO_ROOT / "logs" / "stage_8_heldout_validation.log"
MANIFEST_PATH = REPO_ROOT / "logs" / "stage_8_artifacts.json"
OUT_DIR = REPO_ROOT / "results" / "stage_8_artifacts"


def _synthetic_inputs(seed: int = 123) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n, g = 120, 64
    expr = rng.normal(0.0, 1.0, size=(n, g))
    transported = expr + rng.normal(0.05, 0.2, size=(n, g))
    section = np.array(["S1"] * 30 + ["S2"] * 30 + ["S3"] * 30 + ["S4"] * 30, dtype=object)
    labels = np.array(["tumor"] * 40 + ["intermediate"] * 40 + ["normal"] * 40, dtype=object)
    obs = pd.DataFrame({"marginal_label": labels, "section_id": section})
    cna = np.concatenate(
        [
            rng.normal(0.85, 0.06, 40),
            rng.normal(0.55, 0.05, 40),
            rng.normal(0.20, 0.04, 40),
        ]
    )
    return expr, transported, obs, section, cna


def test_stage8_outputs_exist_and_schema() -> None:
    expr, transported, obs, section, cna = _synthetic_inputs()
    out = REPO_ROOT / "results" / "tmp_stage8_test"
    out.mkdir(parents=True, exist_ok=True)
    manifest = run_stage8_heldout_validation(
        repo_root=REPO_ROOT,
        out_dir=out,
        expr=expr,
        transported=transported,
        obs=obs,
        section=section,
        cna=cna,
        split_seed=11,
        make_plots=False,
    )
    assert "split_strategy" in manifest
    assert manifest["split_strategy"]["unit"] == "section_id"
    assert manifest["split_strategy"]["random_spot_split_used"] is False
    csv_path = out / "stage_8_heldout_metrics.csv"
    assert csv_path.is_file()
    df = pd.read_csv(csv_path)
    assert {"method", "split", "n_spots", "mean_delta_toward_reference"}.issubset(df.columns)
    assert "plot_generation_skipped_make_plots_false" in (manifest.get("unresolved_validation_gaps") or [])


def _print_debug(manifest: dict) -> None:
    print("--- Stage 8 debug ---")
    print("Status:", manifest.get("status"))
    print("Split strategy:", json.dumps(manifest.get("split_strategy", {}), indent=2))
    print("Metrics:", json.dumps(manifest.get("metrics_artifact_paths", {}), indent=2))
    print("Figures:", json.dumps(manifest.get("figure_paths", {}), indent=2))
    gaps = manifest.get("unresolved_validation_gaps") or []
    print("Validation gaps:", gaps if gaps else "none")
    print("Log:", LOG_PATH)
    print("Manifest:", MANIFEST_PATH)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    expr, transported, obs, section, cna = _synthetic_inputs()
    manifest = run_stage8_heldout_validation(
        repo_root=REPO_ROOT,
        out_dir=OUT_DIR,
        expr=expr,
        transported=transported,
        obs=obs,
        section=section,
        cna=cna,
        split_seed=19,
        make_plots=False,
    )
    assert LOG_PATH.is_file(), f"missing log: {LOG_PATH}"
    assert MANIFEST_PATH.is_file(), f"missing manifest: {MANIFEST_PATH}"
    loaded = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    _print_debug(loaded)
    print("Manual run output dir:", OUT_DIR)
