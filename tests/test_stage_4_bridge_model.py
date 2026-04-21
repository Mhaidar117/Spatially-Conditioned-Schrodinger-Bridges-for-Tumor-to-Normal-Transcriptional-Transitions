"""
Stage 4: Spatially conditioned bridge / control model, synthetic fixture, UMAP validation, logs.

Run:  python tests/test_stage_4_bridge_model.py
Or:   pytest tests/test_stage_4_bridge_model.py -v
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
(REPO_ROOT / ".mpl_cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mpl_cache"))

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omega_spatial.config import PipelineConfig  # noqa: E402
from omega_spatial.model import (  # noqa: E402
    generate_counterfactuals,
    load_bridge_checkpoint,
    per_spot_l2_distance_to_reference,
    save_bridge_checkpoint,
    stage_5_handoff_contract,
    train_transport_backend,
    train_spatial_bridge,
    transport_sanity_metrics,
    transport_states,
)

FIG_DIR = REPO_ROOT / "results" / "stage_4_figures"
ARTIFACT_DIR = REPO_ROOT / "results" / "stage_4_artifacts"
LOG_PATH = REPO_ROOT / "logs" / "stage_4_bridge_model.log"
ARTIFACT_JSON_PATH = REPO_ROOT / "logs" / "stage_4_artifacts.json"
TEST_SCRIPT_PATH = REPO_ROOT / "tests" / "test_stage_4_bridge_model.py"
UMAP_RANDOM_STATE = 42

# visualization_standards.md: tumor warm, normal cool, intermediate neutral
MARGINAL_COLORS: dict[str, str] = {
    "tumor": "#d62728",
    "normal": "#1f77b4",
    "intermediate": "#7f7f7f",
}


def _setup_file_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stage_4_bridge_model")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)
    return logger


def _prepare_expression_matrix(expr: np.ndarray, max_genes: int | None = None) -> np.ndarray:
    x = np.maximum(np.asarray(expr, dtype=float), 0.0)
    if max_genes is not None:
        g = min(max_genes, x.shape[1])
        x = x[:, :g]
    lib = x.sum(axis=1, keepdims=True) + 1e-8
    return np.log1p(x / lib * 1e4)


def _embedding_for_umap(X: np.ndarray, random_state: int = UMAP_RANDOM_STATE) -> tuple[np.ndarray, str]:
    try:
        import umap  # type: ignore

        nn = max(2, min(15, X.shape[0] - 1))
        emb = umap.UMAP(
            n_neighbors=nn,
            min_dist=0.1,
            random_state=random_state,
            metric="euclidean",
        ).fit_transform(X)
        return emb, "UMAP (euclidean; reproducible seed=%s)" % random_state
    except Exception:
        from sklearn.decomposition import PCA

        emb = PCA(n_components=2, random_state=random_state).fit_transform(X)
        return emb, "PCA fallback (2D) — install umap-learn for UMAP"


def _shared_embedding_obs_transport(
    x_obs: np.ndarray,
    x_tr: np.ndarray,
    random_state: int = UMAP_RANDOM_STATE,
) -> tuple[np.ndarray, str]:
    """Single fit on stacked [observed; transported] for shared axes (visualization_standards)."""
    z = np.vstack([x_obs, x_tr])
    emb, note = _embedding_for_umap(z, random_state=random_state)
    n = x_obs.shape[0]
    return emb, note


def _plot_umap_marginal(
    emb: np.ndarray,
    labels: np.ndarray,
    title: str,
    y_note: str,
    out_path: Path,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    labs = pd.Series(labels).astype(str)
    for cat in ("normal", "intermediate", "tumor"):
        m = labs == cat
        if not m.any():
            continue
        color = MARGINAL_COLORS.get(cat, "#bcbd22")
        ax.scatter(
            emb[m, 0],
            emb[m, 1],
            s=12,
            alpha=0.85,
            c=color,
            label=cat,
            edgecolors="none",
        )
    other = ~labs.isin(list(MARGINAL_COLORS.keys()))
    if other.any():
        ax.scatter(emb[other, 0], emb[other, 1], s=10, alpha=0.6, c="#cccccc", label="other")
    ax.set_xlabel("UMAP 1" if "UMAP" in y_note else "Component 1")
    ax.set_ylabel("UMAP 2" if "UMAP" in y_note else "Component 2")
    ax.set_title(
        f"{title}\n{y_note}\nAxes: low-dimensional embedding of observed expression "
        f"(validation; labels = marginal_label)."
    )
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(title="marginal_label", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_umap_continuous(
    emb: np.ndarray,
    values: np.ndarray,
    title: str,
    y_note: str,
    out_path: Path,
    cbar_label: str,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=values, s=12, alpha=0.88, cmap="viridis")
    ax.set_xlabel("UMAP 1" if "UMAP" in y_note else "Component 1")
    ax.set_ylabel("UMAP 2" if "UMAP" in y_note else "Component 2")
    ax.set_title(
        f"{title}\n{y_note}\nColormap: perceptually uniform (viridis). "
        f"Positive values = moved closer to normal reference after transport."
    )
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    fig.colorbar(sc, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def synthetic_bridge_fixture(
    rng: np.random.Generator | None = None,
    *,
    n_spots: int = 320,
    n_genes: int = 48,
    tumor_extra_counts: int = 22,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Toy data: nonnegative counts (Visium-like); tumor spots carry extra counts on gene 0
    (known direction toward normal reference after transport).
    Context mimics spatial smoothing: weighted blend of own expression and cohort mean.
    """
    rng = rng or np.random.default_rng(7)
    n_n = n_spots // 2
    n_t = n_spots - n_n
    expr = rng.poisson(10, size=(n_spots, n_genes)).astype(float)
    expr[n_n : n_n + n_t, 0] += rng.poisson(tumor_extra_counts, size=(n_t,)).astype(float)

    global_mu = expr.mean(axis=0, keepdims=True)
    context = np.maximum(
        0.0,
        0.72 * expr + 0.2 * global_mu + rng.poisson(1, size=expr.shape).astype(float),
    )

    labels = np.array(["normal"] * n_n + ["tumor"] * n_t, dtype=object)
    return expr.astype(float), context.astype(float), labels


def test_umap_validation_figures_written() -> None:
    """Required UMAP panels (visualization_standards): observed, transported, movement metric."""
    logger = logging.getLogger("stage_4_umap_test")
    logger.addHandler(logging.NullHandler())
    rng = np.random.default_rng(11)
    expr, context, labels = synthetic_bridge_fixture(rng)
    cfg = PipelineConfig()
    cfg.bridge.ridge_lambda = 1e-2
    cfg.bridge.transport_n_steps = 10
    cfg.bridge.reverse_step_size = 0.18
    cfg.spatial.spatial_smoothing_alpha = 0.25
    model = train_spatial_bridge(expr, context, labels, cfg)
    transported = generate_counterfactuals(model, expr, context)
    paths = write_stage4_umaps(expr, transported, labels, model.normal_reference, logger)
    assert len(paths) == 3
    for p in paths:
        assert Path(p).is_file(), f"missing figure: {p}"


def test_synthetic_training_shapes_movement_and_finite() -> None:
    rng = np.random.default_rng(7)
    expr, context, labels = synthetic_bridge_fixture(rng)
    cfg = PipelineConfig()
    cfg.train.random_seed = 7
    cfg.bridge.ridge_lambda = 1e-2
    cfg.bridge.transport_n_steps = 12
    cfg.bridge.reverse_step_size = 0.18
    cfg.spatial.spatial_smoothing_alpha = 0.25

    model = train_spatial_bridge(expr, context, labels, cfg)
    transported = transport_states(model, expr, context)

    assert transported.shape == expr.shape
    assert np.all(np.isfinite(transported)), "transported states contain NaN/Inf"

    metrics = transport_sanity_metrics(expr, transported, labels, model.normal_reference)
    tumor = metrics.get("label_tumor")
    assert tumor is not None, "expected tumor stratum in metrics"
    assert tumor["mean_delta_toward_ref"] > 0.05, (
        "tumor spots should move toward normal reference on average (sanity check failed)"
    )


def test_transport_backend_selector_linear() -> None:
    rng = np.random.default_rng(17)
    expr, context, labels = synthetic_bridge_fixture(rng, n_spots=80, n_genes=24)
    cfg = PipelineConfig()
    cfg.bridge.backend = "linear"
    model = train_transport_backend(expr, context, labels, cfg)
    transported = model.transport(expr, context)
    assert model.backend == "linear"
    assert transported.shape == expr.shape
    assert np.isfinite(transported).all()


def write_stage4_umaps(
    expr: np.ndarray,
    transported: np.ndarray,
    labels: np.ndarray,
    normal_reference: np.ndarray,
    logger: logging.Logger,
) -> list[str]:
    x_obs = _prepare_expression_matrix(expr)
    x_tr = _prepare_expression_matrix(transported)
    emb_full, note = _shared_embedding_obs_transport(x_obs, x_tr)
    n = x_obs.shape[0]
    emb_obs = emb_full[:n]
    emb_tr = emb_full[n:]
    xlim = float(emb_full[:, 0].min() - 0.5), float(emb_full[:, 0].max() + 0.5)
    ylim = float(emb_full[:, 1].min() - 0.5), float(emb_full[:, 1].max() + 0.5)

    dist_pre = per_spot_l2_distance_to_reference(expr, normal_reference)
    dist_post = per_spot_l2_distance_to_reference(transported, normal_reference)
    delta_toward_ref = dist_pre - dist_post

    paths: list[str] = []

    p1 = FIG_DIR / "stage_4_umap_observed_marginal_labels.png"
    _plot_umap_marginal(
        emb_obs,
        labels,
        "Stage 4 — observed expression (shared embedding with transported states)",
        note,
        p1,
        xlim=xlim,
        ylim=ylim,
    )
    paths.append(str(p1.resolve()))
    logger.info("UMAP artifact: %s", p1)

    p2 = FIG_DIR / "stage_4_umap_transported_marginal_labels.png"
    _plot_umap_marginal(
        emb_tr,
        labels,
        "Stage 4 — transported / counterfactual expression (same UMAP as observed)",
        note,
        p2,
        xlim=xlim,
        ylim=ylim,
    )
    paths.append(str(p2.resolve()))
    logger.info("UMAP artifact: %s", p2)

    p3 = FIG_DIR / "stage_4_umap_delta_distance_toward_reference.png"
    _plot_umap_continuous(
        emb_obs,
        delta_toward_ref,
        "Stage 4 — movement toward normal reference (observed positions in shared UMAP)",
        note,
        p3,
        cbar_label="L2 dist to reference (before) minus L2 dist (after)",
        xlim=xlim,
        ylim=ylim,
    )
    paths.append(str(p3.resolve()))
    logger.info("UMAP artifact: %s", p3)

    return paths


def run_stage4_validation(logger: logging.Logger) -> dict:
    rng = np.random.default_rng(7)
    expr, context, labels = synthetic_bridge_fixture(rng)

    cfg = PipelineConfig()
    cfg.train.random_seed = 7
    cfg.bridge.ridge_lambda = 1e-2
    cfg.bridge.transport_n_steps = 12
    cfg.bridge.reverse_step_size = 0.18
    cfg.spatial.spatial_smoothing_alpha = 0.25

    logger.info("Training config (bridge): %s", json.dumps(cfg.bridge.__dict__, default=str))
    logger.info("Training config (train.random_seed): %s", cfg.train.random_seed)
    logger.info("spatial_smoothing_alpha (reverse-step mixing): %s", cfg.spatial.spatial_smoothing_alpha)

    print("input expr shape:", expr.shape)
    print("input context shape:", context.shape)
    lab_series = pd.Series(labels)
    print("label counts:\n", lab_series.value_counts().to_string())
    print("hyperparameters:", json.dumps({**cfg.bridge.__dict__, "spatial_smoothing_alpha": cfg.spatial.spatial_smoothing_alpha}))

    model = train_spatial_bridge(expr, context, labels, cfg)
    summ = model.training_summary
    logger.info("Model parameter shapes: w_expr %s, w_ctx %s, bias %s", model.w_expr.shape, model.w_ctx.shape, model.bias.shape)
    logger.info("Training summary: %s", json.dumps(summ.to_dict(), indent=2))
    for w in summ.warnings:
        logger.warning("Training warning: %s", w)

    transported = generate_counterfactuals(model, expr, context)
    alt = model.transport(expr, context, n_steps=cfg.bridge.transport_n_steps)
    assert np.allclose(transported, alt)

    if not np.all(np.isfinite(transported)):
        logger.error("Non-finite values in transported states")
        raise AssertionError("NaN/Inf in transported output")

    metrics = transport_sanity_metrics(expr, transported, labels, model.normal_reference)
    print("transport sanity metrics:", json.dumps(metrics, indent=2))

    logger.info("Counterfactual sanity metrics: %s", json.dumps(metrics, indent=2))

    tumor = metrics.get("label_tumor")
    if tumor is None or tumor["mean_delta_toward_ref"] <= 0:
        logger.error("Tumor stratum did not move toward reference on average")
        raise AssertionError("Bridge failed tumor movement diagnostic")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = ARTIFACT_DIR / "bridge_model.npz"
    save_bridge_checkpoint(model, ckpt)
    logger.info("Wrote checkpoint: %s", ckpt)
    reloaded = load_bridge_checkpoint(ckpt)
    assert np.allclose(reloaded.w_expr, model.w_expr)

    umap_paths = write_stage4_umaps(expr, transported, labels, model.normal_reference, logger)
    for p in umap_paths:
        logger.info("Logged UMAP path: %s", p)

    handoff = stage_5_handoff_contract()
    logger.info("Stage 5 handoff (static contract): %s", json.dumps(handoff, indent=2))

    manifest: dict = {
        "checkpoint_path": str(ckpt.resolve()),
        "checkpoint_sidecar_summary": str((ARTIFACT_DIR / "bridge_model.summary.json").resolve()),
        "model_interface_entry_points": handoff["entry_points"],
        "umap_figure_paths": umap_paths,
        "test_script_path": str(TEST_SCRIPT_PATH.resolve()),
        "known_training_limitations": [
            "Linear ridge drift field; not a full entropic OT solver.",
            "Reference mean is computed from spots labeled 'normal' only when present.",
            "Transport uses explicit Euler integration of the learned field (fixed step count).",
            "Synthetic fixture uses Poisson count-like expression; real Visium may need QC and tuning of ridge_lambda / steps.",
        ],
        "expected_downstream_inputs": handoff["input_ordering"],
        "expected_downstream_outputs": handoff["outputs_guaranteed"],
        "stage_5_handoff": handoff,
        "umap_random_state": UMAP_RANDOM_STATE,
        "counterfactual_sanity_metrics": metrics,
        "training_summary": summ.to_dict(),
    }
    return manifest


def main() -> None:
    logger = _setup_file_logging()
    logger.info("Stage 4 bridge model validation (repo root: %s)", REPO_ROOT)

    manifest = run_stage4_validation(logger)

    ARTIFACT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_JSON_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote artifact manifest: %s", ARTIFACT_JSON_PATH)

    produced = (
        [str(LOG_PATH), str(ARTIFACT_JSON_PATH), manifest["checkpoint_path"], manifest["checkpoint_sidecar_summary"]]
        + manifest.get("umap_figure_paths", [])
    )
    logger.info("Produced artifacts: %s", json.dumps(produced, indent=2))

    print("Stage 4 checks complete. Log:", LOG_PATH)
    print("Manifest:", ARTIFACT_JSON_PATH)


if __name__ == "__main__":
    main()
