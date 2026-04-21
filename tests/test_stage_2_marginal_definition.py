"""
Stage 2: marginal definition, malignancy provenance, within-section quantiles, UMAP validation, logs and manifest.

Run:  python tests/test_stage_2_marginal_definition.py
Or:   pytest tests/test_stage_2_marginal_definition.py -v
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omega_spatial.cna import (  # noqa: E402
    PROV_ALIAS_PRECOMPUTED,
    PROV_FALLBACK_EXPRESSION_PROGRAM,
    assign_marginals_from_cna_score,
    resolve_malignancy_scores,
    run_marginal_definition,
    write_stage2_umap_figures,
)
from omega_spatial.config import PipelineConfig, load_config  # noqa: E402
from omega_spatial.types import DatasetBundle  # noqa: E402

LOG_PATH = REPO_ROOT / "logs" / "stage_2_marginal_definition.log"
ARTIFACT_PATH = REPO_ROOT / "logs" / "stage_2_artifacts.json"
TEST_SCRIPT_PATH = REPO_ROOT / "tests" / "test_stage_2_marginal_definition.py"
FIG_DIR = REPO_ROOT / "results" / "stage_2_figures"
GBM_DEFAULT = REPO_ROOT / "Data" / "Inputs" / "general" / "GBM_data"
DEFAULT_YAML = REPO_ROOT / "omega_spatial.default.yaml"


def _setup_file_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stage_2_marginal_definition")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8", mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)
    return logger


def _minimal_cfg(tmp_path: Path) -> PipelineConfig:
    cfg = PipelineConfig(
        input_path=str(tmp_path),
        output_path=str(tmp_path / "out"),
    )
    cfg.cna.infer_if_missing = True
    cfg.cna.require_true_score = False
    cfg.cna.gene_annotation_path = ""
    cfg.cna.program_fallback_top_genes = 50
    cfg.state.low_quantile = 0.2
    cfg.state.high_quantile = 0.8
    cfg.state.section_column = "section_id"
    cfg.train.random_seed = 42
    return cfg


def _synthetic_bundle(
    n_per_section: int,
    rng: np.random.Generator,
    score_a: np.ndarray,
    score_b: np.ndarray,
    extra_obs: dict | None = None,
) -> DatasetBundle:
    n = len(score_a) + len(score_b)
    g = 120
    expr = rng.poisson(5.0, size=(n, g)).astype(float)
    sec = np.array(["A"] * len(score_a) + ["B"] * len(score_b))
    obs = pd.DataFrame(
        {
            "section_id": sec,
            "x": rng.uniform(0, 100, n),
            "y": rng.uniform(0, 100, n),
        }
    )
    if extra_obs:
        for k, v in extra_obs.items():
            obs[k] = v
    return DatasetBundle(
        expr=expr,
        obs=obs,
        var_names=[f"g{i}" for i in range(g)],
        source_path=Path("synthetic"),
        dataset_kind="synthetic",
    )


def test_provided_alias_becomes_canonical_and_provenance(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    n = 80
    scores = np.concatenate([rng.uniform(0, 0.3, n // 2), rng.uniform(0.7, 1.0, n // 2)])
    rng.shuffle(scores)
    obs = pd.DataFrame(
        {
            "section_id": ["S0"] * n,
            "x": np.arange(n),
            "y": np.arange(n),
            "malignancy_score": scores,
        }
    )
    bundle = DatasetBundle(
        expr=rng.poisson(3.0, size=(n, 80)).astype(float),
        obs=obs,
        var_names=[f"g{i}" for i in range(80)],
        source_path=Path("x"),
        dataset_kind="synthetic",
    )
    cfg = _minimal_cfg(tmp_path)
    out = tmp_path / "st2"
    res = run_marginal_definition(bundle, cfg, out)
    obs_out = res.bundle.obs
    assert cfg.cna.canonical_column in obs_out.columns
    assert obs_out["malignancy_provenance"].iloc[0] == PROV_ALIAS_PRECOMPUTED
    assert set(obs_out["marginal_label"].unique()) <= {"tumor", "normal", "intermediate"}
    assert (obs_out["marginal_label"] == "tumor").sum() > 0
    assert (obs_out["marginal_label"] == "normal").sum() > 0
    assert res.section_counts.shape[0] == 1
    umap_names = {Path(p).name for p in res.umap_paths}
    assert "stage_2_umap_cna_score.png" in umap_names
    assert "stage_2_umap_marginal_labels.png" in umap_names
    assert "stage_2_umap_malignancy_provenance.png" in umap_names


def test_expression_program_fallback_when_no_score(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    n = 100
    # Bimodal expression structure so HVG proxy varies
    expr = np.vstack(
        [
            rng.poisson(20.0, size=(n // 2, 200)),
            rng.poisson(2.0, size=(n // 2, 200)),
        ]
    ).astype(float)
    obs = pd.DataFrame(
        {
            "section_id": ["S"] * n,
            "x": np.arange(n),
            "y": np.arange(n),
        }
    )
    bundle = DatasetBundle(expr=expr, obs=obs, var_names=[f"g{i}" for i in range(200)], source_path=Path("x"), dataset_kind="synthetic")
    cfg = _minimal_cfg(tmp_path)
    cfg.cna.program_fallback_top_genes = 80
    res = run_marginal_definition(bundle, cfg, tmp_path / "fb")
    assert res.malignancy_provenance == PROV_FALLBACK_EXPRESSION_PROGRAM
    assert res.bundle.obs[cfg.cna.canonical_column].notna().all()


def test_within_section_thresholds_not_pooled_across_sections(tmp_path: Path) -> None:
    """If we used global quantiles, almost all spots in low section would be 'normal' and high section 'tumor' only by accident; per-section ensures both sections get tumor and normal tails from their own distributions."""
    rng = np.random.default_rng(3)
    na, nb = 150, 150
    score_a = rng.uniform(0.0, 1.0, na)
    score_b = rng.uniform(10.0, 11.0, nb)
    bundle = _synthetic_bundle(na, rng, score_a, score_b, extra_obs={"cna_score": np.concatenate([score_a, score_b])})
    cfg = _minimal_cfg(tmp_path)
    res = run_marginal_definition(bundle, cfg, tmp_path / "sec")
    by_sec = res.section_counts.set_index("section_id")
    assert by_sec.loc["A", "high_threshold"] < 2.0
    assert by_sec.loc["B", "low_threshold"] > 9.0
    # Pooled 80th percentile on combined scores would be ~8+ ; section A tumor count would be wrong
    pooled_hi = float(np.quantile(np.concatenate([score_a, score_b]), 0.8))
    assert pooled_hi > by_sec.loc["A", "high_threshold"] + 1.0
    a_labels = res.bundle.obs[res.bundle.obs["section_id"] == "A"]["marginal_label"]
    b_labels = res.bundle.obs[res.bundle.obs["section_id"] == "B"]["marginal_label"]
    assert (a_labels == "tumor").sum() > 0 and (a_labels == "normal").sum() > 0
    assert (b_labels == "tumor").sum() > 0 and (b_labels == "normal").sum() > 0


def test_required_handoff_columns_present(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    n = 60
    s = np.linspace(0, 1, n)
    obs = pd.DataFrame({"section_id": ["s"] * n, "x": range(n), "y": range(n), "cna_score": s})
    bundle = DatasetBundle(
        expr=rng.poisson(4.0, size=(n, 50)).astype(float),
        obs=obs,
        var_names=[f"g{i}" for i in range(50)],
        source_path=Path("x"),
        dataset_kind="synthetic",
    )
    cfg = _minimal_cfg(tmp_path)
    res = run_marginal_definition(bundle, cfg, tmp_path / "handoff")
    o = res.bundle.obs
    for col in ("cna_score", "marginal_label", "malignancy_provenance", "cna_low_threshold", "cna_high_threshold", "is_pseudo_paired_within_section"):
        assert col in o.columns, f"missing {col}"


def test_cna_bin_metadata_not_used_as_continuous_score(tmp_path: Path) -> None:
    """cna_bin is evaluation-oriented categorical metadata; heuristic must not coerce it to NaNs."""
    rng = np.random.default_rng(6)
    n = 50
    obs = pd.DataFrame(
        {
            "section_id": ["S"] * n,
            "x": range(n),
            "y": range(n),
            "cna_bin": np.random.choice(["low", "high"], n),
        }
    )
    bundle = DatasetBundle(
        expr=rng.poisson(4.0, size=(n, 80)).astype(float),
        obs=obs,
        var_names=[f"g{i}" for i in range(80)],
        source_path=Path("x"),
        dataset_kind="synthetic",
    )
    cfg = _minimal_cfg(tmp_path)
    res = run_marginal_definition(bundle, cfg, tmp_path / "nobin")
    assert res.malignancy_provenance == PROV_FALLBACK_EXPRESSION_PROGRAM
    assert res.bundle.obs[cfg.cna.canonical_column].notna().all()


def test_require_true_cna_hard_fails_when_missing(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    n = 30
    obs = pd.DataFrame({"section_id": ["S"] * n, "x": np.arange(n), "y": np.arange(n)})
    bundle = DatasetBundle(
        expr=rng.poisson(3.0, size=(n, 50)).astype(float),
        obs=obs,
        var_names=[f"g{i}" for i in range(50)],
        source_path=Path("x"),
        dataset_kind="synthetic",
    )
    cfg = _minimal_cfg(tmp_path)
    cfg.cna.require_true_score = True
    try:
        run_marginal_definition(bundle, cfg, tmp_path / "strict_missing")
    except ValueError as exc:
        assert "True CNA score is required" in str(exc)
    else:
        raise AssertionError("Expected strict true-CNA mode to raise when no score is present.")


def test_assign_marginals_respects_section_column_name(tmp_path: Path) -> None:
    cfg = _minimal_cfg(tmp_path)
    cfg.state.section_column = "custom_sec"
    rng = np.random.default_rng(5)
    n = 40
    obs = pd.DataFrame(
        {
            "custom_sec": ["p", "q"] * (n // 2),
            "cna_score": rng.uniform(0, 1, n),
        }
    )
    obs2, counts, _ = assign_marginals_from_cna_score(obs, cfg)
    assert counts["section_id"].tolist() == ["p", "q"]


def run_optional_real_sections(logger: logging.Logger, cfg_base: PipelineConfig) -> tuple[dict, object | None]:
    limitations: list[str] = []
    blockers: list[str] = []
    umap_paths: list[str] = []
    extra: dict = {}

    try:
        from omega_spatial.io import discover_visium_metadata_path, load_dataset, resolve_dataset  # noqa: WPS433
        from omega_spatial.qc import run_qc  # noqa: WPS433
    except ImportError as e:
        limitations.append(f"Import for real-data path failed: {e}")
        return _build_manifest(limitations, blockers, umap_paths, extra, None), None

    if not GBM_DEFAULT.is_dir():
        limitations.append("Data/Inputs/general/GBM_data not found — skipped live Visium Stage 2 integration.")
        logger.warning(limitations[-1])
        return _build_manifest(limitations, blockers, umap_paths, extra, None), None

    import os

    os.environ.setdefault("OMEGA_STAGE1_MAX_SAMPLES", "2")
    meta_path = discover_visium_metadata_path(GBM_DEFAULT)
    p, kind = resolve_dataset(str(GBM_DEFAULT))
    bundle = load_dataset(p, kind)
    bundle = run_qc(bundle, cfg_base)
    out_dir = REPO_ROOT / "results" / "stage_2_live_run"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = run_marginal_definition(bundle, cfg_base, out_dir)
    umap_paths = list(res.umap_paths)
    for u in umap_paths:
        logger.info("UMAP artifact: %s", u)
    logger.info("Malignancy source summary: %s", res.source)
    logger.info("Provenance: %s", res.malignancy_provenance)
    logger.info("Decision path: %s", res.decision_path)
    for _, row in res.section_counts.iterrows():
        logger.info(
            "Section %s tumor=%s normal=%s intermediate=%s thresholds=[%.4f, %.4f]",
            row["section_id"],
            row["tumor_spots"],
            row["normal_spots"],
            row["intermediate_spots"],
            row["low_threshold"],
            row["high_threshold"],
        )
    sc = res.bundle.obs[cfg_base.cna.canonical_column].astype(float)
    logger.info("Score min=%.4f max=%.4f q20=%.4f q50=%.4f q80=%.4f", float(sc.min()), float(sc.max()), float(sc.quantile(0.2)), float(sc.quantile(0.5)), float(sc.quantile(0.8)))
    extra = {
        "live_run_output_dir": str(out_dir),
        "n_sections": int(res.section_counts.shape[0]),
        "cna_source": res.source,
        "malignancy_provenance": res.malignancy_provenance,
        "decision_path": res.decision_path,
        "warnings": res.warnings,
    }
    logger.info("Fallback / decision path: %s", " -> ".join(res.decision_path))
    if res.warnings:
        for w in res.warnings:
            logger.warning("Stage2: %s", w)
    return _build_manifest(limitations, blockers, umap_paths, extra, res), res


def _build_manifest(
    limitations: list[str],
    blockers: list[str],
    umap_paths: list[str],
    extra: dict,
    res,
) -> dict:
    handoff = {
        "canonical_malignancy_column": "cna_score",
        "allowed_marginal_labels": ["tumor", "normal", "intermediate"],
        "intermediate_state_used": True,
        "downstream_warning_semantics": (
            "If malignancy_provenance is inferred_chromosomal_cna_from_expression or "
            "fallback_expression_program_hvg_proxy, downstream should note that training marginals are not true CNA "
            "and may be weakly supervised relative to gold-standard copy number."
        ),
    }
    manifest = {
        "score_source": extra.get("cna_source") if extra else None,
        "malignancy_provenance": extra.get("malignancy_provenance") if extra else None,
        "decision_path": extra.get("decision_path") if extra else None,
        "stage_2_warnings": extra.get("warnings") if extra else None,
        "assigned_output_columns": [
            "cna_score",
            "marginal_label",
            "malignancy_provenance",
            "cna_low_threshold",
            "cna_high_threshold",
            "is_pseudo_paired_within_section",
            "spot_representation",
        ],
        "per_section_summary_artifact_paths": [],
        "umap_figure_paths": umap_paths,
        "test_script_path": str(TEST_SCRIPT_PATH.resolve()),
        "known_edge_cases": [
            "Sections with no finite scores: all intermediate.",
            "Chromosomal CNA inference requires gene_annotation_path and sufficient gene mapping; otherwise HVG fallback runs.",
            "Low tumor/normal counts per section raise warnings per min_spots_per_group.",
            "cna_bin and other evaluation-style columns are blocklisted from heuristic malignancy discovery (not continuous CNA).",
        ],
        "known_limitations": limitations,
        "blockers_for_downstream_stages": blockers,
        "stages_3_4_handoff": handoff,
        "stage_2_extra": extra,
    }
    if res is not None:
        manifest["per_section_summary_artifact_paths"] = [
            str((Path(extra.get("live_run_output_dir", "")) / "malignancy_counts_by_section.csv").resolve())
        ]
    return manifest


def main() -> None:
    logger = _setup_file_logging()
    logger.info("Stage 2 marginal definition validation (repo root: %s)", REPO_ROOT)

    cfg_full = load_config(str(DEFAULT_YAML), str(REPO_ROOT), str(REPO_ROOT / "results" / "stage_2_default_out"))
    logger.info("Aliases configured: %s", cfg_full.cna.aliases)
    logger.info("Canonical column: %s", cfg_full.cna.canonical_column)

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_provided_alias_becomes_canonical_and_provenance(tmp)
        print("OK: provided alias -> canonical cna_score + provenance")
        test_expression_program_fallback_when_no_score(tmp)
        print("OK: expression-program fallback")
        test_within_section_thresholds_not_pooled_across_sections(tmp)
        print("OK: within-section thresholds (no cross-section pooling)")
        test_required_handoff_columns_present(tmp)
        print("OK: handoff columns")
        test_assign_marginals_respects_section_column_name(tmp)
        print("OK: custom section column")
        test_cna_bin_metadata_not_used_as_continuous_score(tmp)
        print("OK: cna_bin not used as continuous malignancy")
        test_require_true_cna_hard_fails_when_missing(tmp)
        print("OK: strict true-CNA hard-fail")

    rng = np.random.default_rng(99)
    mal = rng.uniform(0, 1, 40)
    bundle = _synthetic_bundle(
        40,
        rng,
        mal,
        np.array([], dtype=float),
        extra_obs={"malignancy_score": mal},
    )
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg_plot = _minimal_cfg(REPO_ROOT / "results" / "tmp_cfg")
    obs_r, _, _, _, _ = resolve_malignancy_scores(bundle, cfg_plot)
    obs_r, _, w = assign_marginals_from_cna_score(obs_r, cfg_plot)
    plot_bundle = DatasetBundle(
        expr=bundle.expr[:40],
        obs=obs_r,
        var_names=bundle.var_names,
        source_path=bundle.source_path,
        dataset_kind=bundle.dataset_kind,
    )
    umap_repo = write_stage2_umap_figures(plot_bundle, cfg_plot, FIG_DIR, random_state=42)
    for p in umap_repo:
        logger.info("Repo UMAP (synthetic): %s", p)
    if w:
        for x in w:
            logger.warning("%s", x)

    manifest, live_res = run_optional_real_sections(logger, cfg_full)
    manifest.setdefault("umap_figure_paths", [])
    manifest["umap_figure_paths"] = list(dict.fromkeys(manifest["umap_figure_paths"] + umap_repo))
    manifest["aliases_configured"] = cfg_full.cna.aliases
    manifest["score_column_blocklist"] = [
        "cna_bin",
        "layer",
        "ivygap",
        "org1",
        "org2",
        "mp",
    ]

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote artifact manifest: %s", ARTIFACT_PATH)

    logger.info("Malignancy source (live if run): %s", manifest.get("score_source"))
    logger.info("Aliases configured: %s", manifest.get("aliases_configured"))
    if live_res is not None:
        logger.info("Live run malignancy_provenance column value: %s", live_res.malignancy_provenance)
    logger.info("UMAP paths: %s", json.dumps(manifest.get("umap_figure_paths"), indent=2))
    logger.info(
        "Handoff: column=%s labels=%s",
        manifest["stages_3_4_handoff"]["canonical_malignancy_column"],
        manifest["stages_3_4_handoff"]["allowed_marginal_labels"],
    )

    print("Stage 2 checks complete. Log:", LOG_PATH)
    print("Manifest:", ARTIFACT_PATH)


if __name__ == "__main__":
    main()
