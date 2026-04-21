"""Preflight checks for external dependencies.

The pipeline has a handful of optional external requirements (Rscript for the
true CNA loader, PyTorch for the neural bridge backend, anndata for reading
``.h5ad`` files).  Missing any of them manifests as a cryptic runtime error
deep in the stack — e.g. a ``FileNotFoundError`` for ``Rscript`` or an
``ImportError`` for torch — that gives users no hint about how to fix the
problem.

This module performs the checks up front and returns a structured report so
the CLI can print a single, clear status line per dependency.  Nothing is
raised: each check is best-effort, and callers decide whether a missing
dependency should abort (e.g. ``require_true_score=True`` with no Rscript)
or downgrade silently (e.g. linear backend without torch installed).
"""
from __future__ import annotations

import importlib
import shutil
from dataclasses import dataclass, field


@dataclass
class CheckResult:
    """Outcome of a single preflight check."""

    name: str
    available: bool
    detail: str
    required_for: str
    hint: str = ""


@dataclass
class PreflightReport:
    """Structured collection of all preflight checks."""

    results: list[CheckResult] = field(default_factory=list)

    def add(self, result: CheckResult) -> None:
        self.results.append(result)

    def missing(self) -> list[CheckResult]:
        return [r for r in self.results if not r.available]

    def format_text(self) -> str:
        lines = ["Omega-spatial preflight:"]
        for r in self.results:
            mark = "ok  " if r.available else "miss"
            lines.append(f"  [{mark}] {r.name:10s}  — {r.detail}")
            if not r.available and r.hint:
                lines.append(f"           hint: {r.hint}")
        return "\n".join(lines)


def check_rscript() -> CheckResult:
    """Locate ``Rscript`` on PATH — required for loading true CNA scores."""
    path = shutil.which("Rscript")
    if path:
        return CheckResult(
            name="Rscript",
            available=True,
            detail=f"found at {path}",
            required_for="loading precomputed CNA .rds files (io.py)",
        )
    return CheckResult(
        name="Rscript",
        available=False,
        detail="not found on PATH",
        required_for="loading precomputed CNA .rds files (io.py)",
        hint=(
            "Install R (https://cran.r-project.org/) or set "
            "cna.require_true_score=false to skip the true-CNA loader."
        ),
    )


def check_torch() -> CheckResult:
    """Probe ``torch`` — required only for bridge.backend='neural'."""
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="torch",
            available=False,
            detail=f"import failed: {exc.__class__.__name__}",
            required_for="bridge.backend='neural' only",
            hint="pip install torch (optional; linear backend is default).",
        )
    return CheckResult(
        name="torch",
        available=True,
        detail=f"version {getattr(torch, '__version__', '?')}",
        required_for="bridge.backend='neural' only",
    )


def check_anndata() -> CheckResult:
    """Probe ``anndata`` — required for reading ``.h5ad`` inputs."""
    try:
        ad = importlib.import_module("anndata")
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="anndata",
            available=False,
            detail=f"import failed: {exc.__class__.__name__}",
            required_for="reading .h5ad inputs",
            hint="pip install anndata",
        )
    return CheckResult(
        name="anndata",
        available=True,
        detail=f"version {getattr(ad, '__version__', '?')}",
        required_for="reading .h5ad inputs",
    )


def run_preflight() -> PreflightReport:
    """Run all checks and return a report.  Never raises."""
    report = PreflightReport()
    report.add(check_rscript())
    report.add(check_torch())
    report.add(check_anndata())
    return report
