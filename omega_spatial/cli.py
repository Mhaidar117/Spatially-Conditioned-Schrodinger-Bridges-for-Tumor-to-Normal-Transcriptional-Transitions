from __future__ import annotations

import argparse
import sys

from .config import load_config
from .pipeline import run_pipeline
from .preflight import run_preflight


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="Omega-spatial", description="Omega Spatial Control one-command pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run end-to-end pipeline.")
    run.add_argument("--input", required=True, help="Dataset path (.h5ad, .csv/.tsv, or Visium folder).")
    run.add_argument("--output", required=True, help="Output results directory.")
    run.add_argument("--config", required=False, default=None, help="Optional YAML config override.")
    run.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the dependency preflight check at startup.",
    )

    sub.add_parser("preflight", help="Check external dependencies and exit.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "preflight":
        report = run_preflight()
        print(report.format_text())
        return 0 if not report.missing() else 1
    if args.command == "run":
        if not getattr(args, "skip_preflight", False):
            report = run_preflight()
            print(report.format_text())
        cfg = load_config(args.config, args.input, args.output)
        run_pipeline(cfg)
        print(f"Pipeline completed. Results written to: {cfg.output_path}")
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
