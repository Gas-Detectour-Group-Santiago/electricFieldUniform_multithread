from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_hlevels_list_mode() -> None:
    result = subprocess.run(
        [sys.executable, "tools/run_hlevels.py", "--gases", "Ar", "--mixtures", "ArIsoCF4", "--list"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "hLevels: Ar" in result.stdout
    assert "ArIsoCF4" in result.stdout


def test_cross_sections_list_mode() -> None:
    result = subprocess.run(
        [sys.executable, "tools/run_cross_sections.py", "--gases", "Ar,CO2,CH4", "--list"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "Garfield label 'ar'" in result.stdout
    assert "Garfield label 'co2'" in result.stdout


def test_single_runner_exposes_reference_stages() -> None:
    source = (ROOT / "run.py").read_text(encoding="utf-8")
    assert '"hlevels"' in source
    assert '"cross_sections"' in source
    assert 'run_hlevels(["--all"])' in source
    assert 'run_cross_sections(["--all"])' in source
