"""Tests for the preflight dependency checks."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omega_spatial.preflight import (  # noqa: E402
    check_anndata,
    check_rscript,
    check_torch,
    run_preflight,
)


def test_preflight_report_enumerates_all_three_checks() -> None:
    report = run_preflight()
    names = [r.name for r in report.results]
    assert names == ["Rscript", "torch", "anndata"]


def test_preflight_text_formats_both_states() -> None:
    """format_text should render both present and missing dependencies."""
    report = run_preflight()
    text = report.format_text()
    assert "Omega-spatial preflight:" in text
    for r in report.results:
        assert r.name in text
    # Missing deps should surface a 'hint:' line, present ones should not.
    for r in report.missing():
        if r.hint:
            assert r.hint in text


def test_each_check_returns_required_for_metadata() -> None:
    for result in (check_rscript(), check_torch(), check_anndata()):
        assert isinstance(result.required_for, str)
        assert len(result.required_for) > 0
        # available implies no hint is needed (but optional);
        # missing implies a user-actionable hint is surfaced.
        if not result.available:
            assert result.hint, f"missing check {result.name} should carry a hint"
