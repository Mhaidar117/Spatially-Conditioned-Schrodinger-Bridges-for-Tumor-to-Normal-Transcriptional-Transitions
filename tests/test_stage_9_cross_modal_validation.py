"""
Stage 9: optional cross-modal ST/CODEX validation (dedicated stream).

Run:  python tests/test_stage_9_cross_modal_validation.py
Or:   pytest tests/test_stage_9_cross_modal_validation.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omega_spatial.validation import run_stage9_cross_modal_validation  # noqa: E402

LOG_PATH = REPO_ROOT / "logs" / "stage_9_cross_modal_validation.log"
MANIFEST_PATH = REPO_ROOT / "logs" / "stage_9_artifacts.json"
OUT_DIR = REPO_ROOT / "results" / "stage_9_artifacts"


def test_stage9_outputs_exist_even_when_assets_missing(tmp_path: Path) -> None:
    obs = pd.DataFrame({"sample": ["S1", "S2", "S3"], "spot_id": ["a", "b", "c"]})
    out = tmp_path / "stage9_out"
    out.mkdir(parents=True, exist_ok=True)
    manifest = run_stage9_cross_modal_validation(
        repo_root=REPO_ROOT,
        out_dir=out,
        obs=obs,
        make_plots=False,
        backend_names=["SpatialBridge_linear", "SpatialBridge_neural"],
    )
    assert "coverage" in manifest
    assert "resource_presence" in manifest
    csv_path = out / "stage_9_cross_modal_metrics.csv"
    assert csv_path.is_file()
    df = pd.read_csv(csv_path)
    assert "n_st_align_sample_dirs" in df.columns
    assert "transport_backends_in_run" in manifest
    assert "plot_generation_skipped_make_plots_false" in (manifest.get("unresolved_validation_gaps") or [])


def test_stage9_uses_section_id_when_sample_missing(tmp_path: Path) -> None:
    obs = pd.DataFrame(
        {
            "section_id": ["MGH258", "UKF243", "UKF248"],
            "spot_id": ["x", "y", "z"],
        }
    )
    out = tmp_path / "stage9_section_fallback"
    out.mkdir(parents=True, exist_ok=True)
    manifest = run_stage9_cross_modal_validation(
        repo_root=REPO_ROOT,
        out_dir=out,
        obs=obs,
        make_plots=False,
    )
    csv_path = out / "stage_9_cross_modal_metrics.csv"
    df = pd.read_csv(csv_path)
    assert int(df.loc[0, "n_metadata_samples"]) == 3
    assert "metadata_sample_ids_used" in manifest.get("coverage", {})
    assert len(manifest["coverage"]["metadata_sample_ids_used"]) == 3


def _print_debug(manifest: dict) -> None:
    print("--- Stage 9 debug ---")
    print("Status:", manifest.get("status"))
    print("Coverage:", json.dumps(manifest.get("coverage", {}), indent=2))
    print("Resources:", json.dumps(manifest.get("resource_presence", {}), indent=2))
    print("Metrics paths:", json.dumps(manifest.get("metrics_artifact_paths", {}), indent=2))
    print("Figure paths:", json.dumps(manifest.get("figure_paths", {}), indent=2))
    gaps = manifest.get("unresolved_validation_gaps") or []
    print("Validation gaps:", gaps if gaps else "none")
    print("Log:", LOG_PATH)
    print("Manifest:", MANIFEST_PATH)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    obs = pd.DataFrame({"sample": ["MGH258", "UKF243", "UKF248"], "spot_id": ["x", "y", "z"]})
    manifest = run_stage9_cross_modal_validation(
        repo_root=REPO_ROOT,
        out_dir=OUT_DIR,
        obs=obs,
        make_plots=False,
    )
    assert LOG_PATH.is_file(), f"missing log: {LOG_PATH}"
    assert MANIFEST_PATH.is_file(), f"missing manifest: {MANIFEST_PATH}"
    loaded = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    _print_debug(loaded)
    print("Manual run output dir:", OUT_DIR)
