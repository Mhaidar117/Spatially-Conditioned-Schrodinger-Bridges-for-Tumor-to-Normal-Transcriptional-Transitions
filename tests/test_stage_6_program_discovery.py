"""
Stage 6: Program discovery (NMF on perturbations; PCA/ICA baselines), artifacts, UMAP validation, logs.

Run:  python tests/test_stage_6_program_discovery.py
Or:   pytest tests/test_stage_6_program_discovery.py -v
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
(REPO_ROOT / ".mpl_cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mpl_cache"))

import matplotlib

matplotlib.use("Agg")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omega_spatial.config import PipelineConfig  # noqa: E402
from omega_spatial.programs import (  # noqa: E402
    compare_factorizations,
    run_nmf_programs,
    run_program_discovery,
    save_program_artifacts,
    top_genes_per_program,
    write_stage6_artifact_manifest,
    write_stage6_umap_figures,
)

FIG_DIR = REPO_ROOT / "results" / "stage_6_figures"
ARTIFACT_DIR = REPO_ROOT / "results" / "stage_6_artifacts"
LOG_PATH = REPO_ROOT / "logs" / "stage_6_program_discovery.log"
ARTIFACT_JSON_PATH = REPO_ROOT / "logs" / "stage_6_artifacts.json"
TEST_SCRIPT_PATH = REPO_ROOT / "tests" / "test_stage_6_program_discovery.py"


def _setup_file_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stage_6_program_discovery")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)
    return logger


def synthetic_perturbation_with_programs(
    rng: np.random.Generator | None = None,
    *,
    n_spots: int = 240,
    n_genes: int = 60,
    n_programs: int = 4,
    noise_scale: float = 0.15,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Nonnegative underlying structure; perturbation = centered log-like deviations."""
    rng = rng or np.random.default_rng(11)
    h = np.zeros((n_programs, n_genes))
    block = n_genes // n_programs
    for k in range(n_programs):
        start = k * block
        end = (k + 1) * block if k < n_programs - 1 else n_genes
        h[k, start:end] = rng.uniform(0.4, 1.0, size=(end - start,))
    w = rng.uniform(0.0, 1.0, size=(n_spots, n_programs))
    base = w @ h
    noise = rng.normal(0.0, noise_scale, size=base.shape)
    perturb = base + noise
    var_names = [f"gene_{j}" for j in range(n_genes)]
    n_n = n_spots // 2
    labels = np.array(["normal"] * n_n + ["tumor"] * (n_spots - n_n), dtype=object)
    obs = pd.DataFrame(
        {
            "spot_id": [f"s{i}" for i in range(n_spots)],
            "marginal_label": labels.astype(str),
        }
    )
    return perturb.astype(np.float64), var_names, obs


def test_compare_factorization_table_shape() -> None:
    p, _, _ = synthetic_perturbation_with_programs(n_spots=80, n_genes=20, n_programs=3)
    cfg = PipelineConfig()
    cfg.programs.chosen_components = 3
    df = compare_factorizations(p, cfg)
    assert len(df) == 3
    assert set(df["method"].tolist()) == {"NMF", "PCA", "ICA"}


def test_run_nmf_programs_dimensions() -> None:
    p, genes, _ = synthetic_perturbation_with_programs()
    cfg = PipelineConfig()
    cfg.programs.chosen_components = 4
    cfg.programs.random_seed = 7
    res = run_nmf_programs(p, genes, cfg)
    assert res.scores.shape == (p.shape[0], 4)
    assert res.loadings.shape == (p.shape[1], 4)
    assert res.chosen_method == "NMF"
    assert not res.model_selection.empty


def test_discovery_rejects_gene_mismatch() -> None:
    p, genes, _ = synthetic_perturbation_with_programs(n_spots=30, n_genes=10)
    cfg = PipelineConfig()
    try:
        run_program_discovery(p, genes[:-1], cfg)
    except ValueError as e:
        assert "var_names" in str(e) or "genes" in str(e).lower()
    else:
        raise AssertionError("expected ValueError")


def test_umap_figures_written() -> None:
    logger = logging.getLogger("stage_6_umap_test")
    logger.addHandler(logging.NullHandler())
    p, genes, obs = synthetic_perturbation_with_programs()
    cfg = PipelineConfig()
    cfg.programs.chosen_components = 4
    cfg.programs.umap_random_state = 42
    disc = run_program_discovery(p, genes, cfg)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = write_stage6_umap_figures(p, disc, FIG_DIR, logger, cfg, obs=obs)
    assert len(paths) >= 3
    for pth in paths:
        assert Path(pth).is_file(), f"missing figure: {pth}"
    names = {Path(x).name for x in paths}
    assert "stage_6_umap_nmf_dominant_program.png" in names
    assert "stage_6_umap_nmf_program_score_panels.png" in names
    assert "stage_6_umap_pca_dominant_partition.png" in names


def run_stage6_validation(logger: logging.Logger) -> dict:
    rng = np.random.default_rng(21)
    p, genes, obs = synthetic_perturbation_with_programs(rng)

    cfg = PipelineConfig()
    cfg.programs.chosen_components = 4
    cfg.programs.random_seed = 7
    cfg.programs.umap_random_state = 42
    cfg.programs.top_genes_log = 10

    n_comp = min(cfg.programs.chosen_components, p.shape[0], p.shape[1])
    logger.info("Perturbation shape %s; chosen_components (effective) %s", p.shape, n_comp)
    logger.info("Program config: %s", json.dumps(cfg.programs.__dict__, default=str))
    print("matrix shapes: perturbation", p.shape)
    print("chosen component count (effective):", n_comp)

    discovery = run_program_discovery(p, genes, cfg)
    for w in discovery.warnings:
        logger.warning("%s", w)

    print("reconstruction / comparison:\n", discovery.model_selection.to_string())
    logger.info("Method comparison:\n%s", discovery.model_selection.to_string())

    topg = top_genes_per_program(discovery.nmf_loadings, cfg.programs.top_genes_log)
    for prog, glist in list(topg.items())[: min(3, len(topg))]:
        logger.info("Top genes %s: %s", prog, ", ".join(glist))
        print(f"top genes {prog}:", glist[:5], "...")

    stab = discovery.diagnostics.get("nmf_stability", {})
    logger.info("NMF stability: %s", json.dumps(stab, indent=2, default=str))
    print("NMF stability:", json.dumps(stab, default=str))

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact_paths = save_program_artifacts(ARTIFACT_DIR, discovery, obs=obs, spot_id_column="spot_id")
    for k, v in artifact_paths.items():
        logger.info("Wrote artifact [%s]: %s", k, v)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    umap_paths = write_stage6_umap_figures(p, discovery, FIG_DIR, logger, cfg, obs=obs)
    for u in umap_paths:
        logger.info("UMAP artifact path: %s", u)

    limitations = [
        "Programs are fit on perturbation vectors (Stage 5); not raw expression.",
        "NMF uses nonnegative shift of perturbations; PCA/ICA use raw perturbations for baseline comparison.",
        "ICA can fail on small or rank-deficient matrices; see factorization_comparison note column.",
        "Dominant PCA/ICA labels are argmax |score| heuristics, not mixture model assignments.",
    ]
    write_stage6_artifact_manifest(
        ARTIFACT_JSON_PATH,
        artifact_paths=artifact_paths,
        umap_paths=umap_paths,
        test_script_path=str(TEST_SCRIPT_PATH.resolve()),
        known_limitations=limitations,
        extra={
            "output_directory": str(ARTIFACT_DIR.resolve()),
            "figure_directory": str(FIG_DIR.resolve()),
            "umap_random_state": cfg.programs.umap_random_state,
            "n_components": discovery.n_components,
            "diagnostics": discovery.diagnostics,
            "warnings": discovery.warnings,
        },
    )
    logger.info("Wrote manifest: %s", ARTIFACT_JSON_PATH)

    # Dimensional sanity
    if discovery.nmf_scores.shape[0] != p.shape[0]:
        raise AssertionError("scores rows != n_spots")
    if discovery.nmf_scores.shape[1] != discovery.n_components:
        raise AssertionError("scores cols != n_components")
    if discovery.nmf_loadings.shape[0] != p.shape[1]:
        raise AssertionError("loadings rows != n_genes")
    if discovery.nmf_loadings.empty or discovery.nmf_scores.empty:
        raise AssertionError("empty factorization outputs")

    return {
        "artifact_paths": artifact_paths,
        "umap_paths": umap_paths,
        "manifest_path": str(ARTIFACT_JSON_PATH.resolve()),
        "log_path": str(LOG_PATH.resolve()),
    }


def main() -> None:
    logger = _setup_file_logging()
    logger.info("Stage 6 program discovery validation (repo root: %s)", REPO_ROOT)
    out = run_stage6_validation(logger)
    produced = [out["log_path"], out["manifest_path"]] + list(out["artifact_paths"].values()) + out["umap_paths"]
    logger.info("Produced artifacts: %s", json.dumps(produced, indent=2))
    print("Stage 6 checks complete. Log:", LOG_PATH)
    print("Manifest:", ARTIFACT_JSON_PATH)


if __name__ == "__main__":
    main()
