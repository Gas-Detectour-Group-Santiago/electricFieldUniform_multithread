from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from scintillation.gui.catalog import discover_primary_datasets, discover_secondary_datasets
from scintillation.gui.pipeline import PipelineSelection, build_plan
from scintillation.gui.style_config import (
    active_style_name,
    load_style,
    matplotlib_rc,
    temporary_style,
)


ROOT = Path(__file__).resolve().parents[1]


def test_gui_catalogues_are_discoverable() -> None:
    primary = discover_primary_datasets(ROOT)
    secondary = discover_secondary_datasets(ROOT)
    assert primary
    assert secondary
    assert any(item.family == "simulation_catalog" for item in secondary)


def test_pipeline_plan_uses_only_existing_runners() -> None:
    selection = PipelineSelection(fits=True, primary=True, toys=12, style_preset="tfm")
    plan = build_plan(selection, ROOT)
    assert plan.command[2] == "all"
    assert ("--toys", "12") == (
        plan.command[plan.command.index("--toys")],
        plan.command[plan.command.index("--toys") + 1],
    )
    assert "--min-toys" in plan.command
    assert "--toy-batch" in plan.command
    assert "--toy-tolerance" in plan.command
    assert "--toy-correlation-tolerance" in plan.command
    assert "--toy-stable-checks" in plan.command
    assert plan.command[0].endswith(".venv/bin/python3")
    assert plan.command[1].endswith("run.py")
    assert plan.env["SCINTILLATION_STYLE_FILE"].endswith("config/styles/tfm.json")


def test_style_controls_include_visible_major_and_minor_ticks() -> None:
    style = load_style(ROOT, "tfm")
    style["tick_major_width"] = 2.4
    style["tick_minor_width"] = 1.3
    style["tick_major_length"] = 8.0
    style["tick_minor_length"] = 4.0
    rc = matplotlib_rc(style)
    assert rc["xtick.major.width"] == 2.4
    assert rc["ytick.minor.width"] == 1.3
    assert rc["xtick.major.size"] == 8.0
    assert rc["ytick.minor.size"] == 4.0
    with temporary_style(style):
        fig, ax = plt.subplots()
        ax.minorticks_on()
        assert plt.rcParams["xtick.major.width"] == 2.4
        plt.close(fig)


def test_gui_uses_the_single_project_runner() -> None:
    source = (ROOT / "app/desktop/run_page.py").read_text(encoding="utf-8")
    assert 'project_root / "run.py"' in source
    assert "workflows/" not in source
    assert "run_all.sh" not in source


def test_active_style_is_valid() -> None:
    name = active_style_name(ROOT)
    style = load_style(ROOT, name)
    assert style["figure_wide"][0] > 0
    assert style["line_width_main"] > 0
