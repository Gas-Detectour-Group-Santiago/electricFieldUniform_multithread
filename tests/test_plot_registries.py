from __future__ import annotations

from pathlib import Path

import pandas as pd

from scintillation.plotting.recipe_config import (
    as_bool,
    load_primary_band_plots,
    read_plot_rows,
    split_values,
)
from scintillation._compat.project.primary_predictions.auxiliares.prediction_types import BandPlotConfig


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("fits", "primary", "secondary", "spectra")


def _raw(family: str) -> pd.DataFrame:
    path = ROOT / "config" / "plots" / f"{family}.csv"
    assert path.is_file()
    return pd.read_csv(path, keep_default_na=False)


def test_four_plot_registries_are_the_only_canonical_figure_files() -> None:
    plot_dir = ROOT / "config" / "plots"
    assert {path.name for path in plot_dir.glob("*.csv")} == {f"{name}.csv" for name in FAMILIES}
    assert not (ROOT / "config" / "plot_recipes.csv").exists()
    assert not (ROOT / "config" / "secondary_scans.csv").exists()


def test_enabled_plot_rows_have_ids_outputs_and_consistent_figure_metadata() -> None:
    for family in FAMILIES:
        frame = _raw(family)
        active = frame.loc[frame["enabled"].map(as_bool)]
        assert not active.empty, family
        assert active["plot_id"].astype(str).str.strip().ne("").all()
        assert active["output"].astype(str).str.strip().ne("").all()
        for _, rows in active.groupby("plot_id", sort=False):
            for column in ("output", "title"):
                if column in rows:
                    assert rows[column].astype(str).nunique() == 1, (family, rows.iloc[0]["plot_id"], column)


def test_primary_and_secondary_dataset_references_exist() -> None:
    registry = pd.read_csv(ROOT / "config" / "experimental_datasets.csv", keep_default_na=False)
    known = set(registry.loc[registry["enabled"].map(as_bool), "dataset_id"].astype(str))

    primary = read_plot_rows("primary", ROOT)
    for value in primary.get("datasets", ()):  # pipe-separated overlay ids
        assert set(split_values(value, cast=str)).issubset(known)

    secondary = read_plot_rows("secondary", ROOT)
    experimental = secondary.loc[secondary["kind"].astype(str) == "experimental"]
    assert set(experimental["dataset_id"].astype(str)).issubset(known)


def test_secondary_model_rows_reference_known_selections() -> None:
    selections = pd.read_csv(ROOT / "config" / "secondary_selections.csv", keep_default_na=False)
    known = set(selections.loc[selections["enabled"].map(as_bool), "selection_id"].astype(str))
    secondary = read_plot_rows("secondary", ROOT)
    model_rows = secondary.loc[secondary["plot_type"].isin(["multiband", "metadata"])]
    model_rows = model_rows.loc[model_rows["kind"].isin(["model", "combined"])]
    assert set(model_rows["selection_id"].astype(str)).issubset(known)


def test_primary_loader_reads_legacy_equivalent_band_set_and_xray_switch() -> None:
    legacy_root = ROOT / "src" / "scintillation" / "_compat" / "project"
    plots = load_primary_band_plots(
        legacy_root,
        "standard",
        band_plot_cls=BandPlotConfig,
        normalization_lookup=lambda name: name,
    )
    assert len(plots) == 5
    assert all(plot.scale_xray_with_normalization for plot in plots)
    assert {plot.id for plot in plots} == {
        "ArCF4_primary_uv",
        "ArCF4_primary_vis",
        "ArN2_primary_uv",
        "ArCF4_IR_primary_total",
        "ArN2_IR_primary_total",
    }


def test_joint_primary_and_spectra_are_individually_configured() -> None:
    primary = read_plot_rows("primary", ROOT)
    joint = primary.loc[primary["group"] == "joint_ir"]
    assert joint["plot_id"].nunique() == 3
    assert len(joint.loc[joint["plot_type"] == "joint_multiband"]) == 12
    assert len(joint.loc[joint["plot_type"] == "joint_pure_ar"]) == 3

    spectra = read_plot_rows("spectra", ROOT)
    assert set(spectra["plot_type"]) == {"raw", "generated", "generated_extended", "comparison"}
    insets = spectra.loc[spectra["plot_type"] == "generated_extended"]
    assert len(insets) == 2
    assert insets["inset"].map(as_bool).all()
    assert (pd.to_numeric(insets["wavelength_min_nm"]) == 200.0).all()
    assert (pd.to_numeric(insets["wavelength_max_nm"]) == 800.0).all()


def test_fit_point_errorbars_are_statistical_by_contract() -> None:
    source = (
        ROOT
        / "src/scintillation/_compat/project/primary_fits/auxiliares/ploting.py"
    ).read_text(encoding="utf-8")
    default_block = source[source.index("if err_patterns is None:") : source.index("if line_label_fmt is None:")]
    assert "ErrStat {col}" in default_block
    assert '"Err {col}"' not in default_block


def test_all_cf4_concentration_scans_keep_the_pure_cf4_endpoint() -> None:
    selections = pd.read_csv(ROOT / "config" / "secondary_selections.csv", keep_default_na=False)
    scans = selections.loc[
        selections["enabled"].map(as_bool)
        & selections["selection_id"].astype(str).str.contains("concentration_scan")
        & selections["gas_mixture_in"].astype(str).str.split("|").map(lambda values: "CF4" in values)
    ]
    assert not scans.empty
    # A non-empty exact gas filter would discard rows whose canonical
    # gas_mixture is "CF4", i.e. the real 100% endpoint.
    assert scans["gas"].astype(str).str.strip().eq("").all()
