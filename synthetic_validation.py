#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from omega_spatial.config import PipelineConfig
from omega_spatial.synthetic_validation import run_synthetic_validation


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def _setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("synthetic_validation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)
    return logger


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run standalone synthetic validation for Omega Spatial")
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: results/synthetic_validation)",
    )
    ap.add_argument("--grid-rows", type=int, default=24, help="Synthetic grid row count")
    ap.add_argument("--grid-cols", type=int, default=24, help="Synthetic grid column count")
    ap.add_argument("--genes", type=int, default=48, help="Number of synthetic genes")
    ap.add_argument("--seed", type=int, default=7, help="Random seed")
    ap.add_argument(
        "--backend",
        type=str,
        default="linear",
        choices=["linear", "bayesian_linear", "neural"],
        help="Transport backend to use for the synthetic run",
    )
    ap.add_argument(
        "--noise-scale",
        type=float,
        default=0.02,
        help=(
            "Std of per-spot, per-gene Gaussian noise added to the synthetic "
            "expression field.  Use ~1.5 to see the spatial-vs-nonspatial "
            "ablation advantage (spatial L2 drops substantially)."
        ),
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    root = repo_root()
    out_dir = (args.output or (root / "results" / "synthetic_validation")).resolve()
    logger = _setup_logger(root / "logs" / "synthetic_validation.log")

    cfg = PipelineConfig(output_path=str(out_dir))
    cfg.bridge.backend = str(args.backend)
    logger.info(
        "Running synthetic validation with grid=%sx%s, genes=%s, seed=%s, backend=%s",
        args.grid_rows,
        args.grid_cols,
        args.genes,
        args.seed,
        args.backend,
    )
    manifest = run_synthetic_validation(
        repo_root=root,
        out_dir=out_dir,
        cfg=cfg,
        grid_shape=(int(args.grid_rows), int(args.grid_cols)),
        n_genes=int(args.genes),
        seed=int(args.seed),
        logger=logger,
        noise_scale=float(args.noise_scale),
    )
    logger.info("Synthetic validation manifest: %s", json.dumps(manifest, indent=2))
    print(f"Synthetic validation complete. Output: {out_dir}")
    print(f"Manifest: {out_dir / 'synthetic_validation_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
