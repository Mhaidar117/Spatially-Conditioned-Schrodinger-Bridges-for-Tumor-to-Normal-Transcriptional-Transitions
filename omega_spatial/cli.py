from __future__ import annotations

import argparse
import sys

from .config import load_config
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="Omega-spatial", description="Omega Spatial Control one-command pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run end-to-end pipeline.")
    run.add_argument("--input", required=True, help="Dataset path (.h5ad, .csv/.tsv, or Visium folder).")
    run.add_argument("--output", required=True, help="Output results directory.")
    run.add_argument("--config", required=False, default=None, help="Optional YAML config override.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        cfg = load_config(args.config, args.input, args.output)
        run_pipeline(cfg)
        print(f"Pipeline completed. Results written to: {cfg.output_path}")
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
