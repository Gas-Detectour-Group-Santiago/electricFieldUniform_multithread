from pathlib import Path
import pandas as pd


def test_secondary_scan_configuration_alpha_only():
    path = Path(__file__).resolve().parents[1] / 'config' / 'plots' / 'secondary.csv'
    df = pd.read_csv(path, keep_default_na=False)
    scans = df.loc[df['plot_type'] == 'scan']
    assert list(scans['plot_id']) == ['alpha_vs_E', 'alpha_vs_E_over_p']
    assert scans['output'].astype(str).str.contains('alpha_studies').all()
    assert set(scans['series']) == {'pressure'}
    assert set(scans['y']) == {'alpha_eff_cm_inv'}


def test_newsimulation_transport_has_quiet_guides_and_emission_has_per_npe() -> None:
    source = (Path(__file__).resolve().parents[1] / "src/scintillation/predictions/alpha_campaign.py").read_text(encoding="utf-8")
    transport = source[source.index("def _plot_transport"):source.index("def _plot_emission")]
    assert "fill_between" not in transport
    assert "_sample_parameter_curves" not in transport
    assert "_draw_campaign_series" in transport
    assert "connect_points=connect_emission_points" in source
    assert "markerfacecolor=_light_marker_face(color)" in source
    assert "_ordered_composition_labels" in source
    assert "_COMPOSITION_MARKERS" in source
    assert 'out["vuv_per_npe"]' in source
    assert 'out["uv_per_npe"]' in source
    assert "np.isclose(float(gap_mm), 0.150" in source
    assert 'xerr=xe if np.isfinite(xe).any() else None' in source


def test_per_npe_normalisation_is_per_e_times_gain(monkeypatch):
    import numpy as np
    import pandas as pd
    from scintillation.predictions import alpha_campaign as campaign

    monkeypatch.setattr(campaign, "_load_arcf4_parameters", lambda _paths: np.ones(11))
    monkeypatch.setattr(campaign, "_load_ar2nd_parameters_with_hydrocarbon_proxy", lambda _paths: {})
    monkeypatch.setattr(campaign, "theory_yield_ar2nd_continium", lambda *args, **kwargs: 20.0)
    monkeypatch.setattr(
        campaign,
        "theory_yield_uv_components",
        lambda *args, **kwargs: {
            "total": 10.0,
            "cf4_plus": 4.0,
            "ar_third_continuum": 3.0,
            "cf3": 3.0,
        },
    )

    row = {
        "npe": 20.0,
        "gain": 5.0,
        "gain_error": 0.5,
        "pressure_bar": 1.0,
        "campaign_mixture": "ArCF4",
        "fraction_ar": 0.95,
        "fraction_cf4": 0.05,
        "fraction_ic4h10": 0.0,
        "fraction_n2": 0.0,
        "fraction_co2": 0.0,
        "fraction_ch4": 0.0,
    }
    for column in campaign._POPULATION_COLUMNS:
        row[column] = 1.0
    result = campaign._evaluate_secondary_yields(pd.DataFrame([row]), object())
    assert np.isclose(result.loc[0, "vuv_per_npe"], result.loc[0, "vuv_per_e"] * 5.0)
    assert np.isclose(result.loc[0, "uv_per_npe"], result.loc[0, "uv_per_e"] * 5.0)

    relative_population_error = np.sqrt(3.0) / 3.0
    relative_gain_error = 0.5 / 5.0
    assert np.isclose(result.loc[0, "gain_relative_error"], relative_gain_error)
    assert np.isclose(
        result.loc[0, "vuv_per_e_err"],
        abs(result.loc[0, "vuv_per_e"])
        * np.hypot(relative_population_error, relative_gain_error),
    )
    assert np.isclose(
        result.loc[0, "uv_per_e_err"],
        abs(result.loc[0, "uv_per_e"])
        * np.hypot(relative_population_error, relative_gain_error),
    )
    # The same gain is divided out in ph/e- and multiplied back in ph/NPE,
    # therefore its uncertainty cancels from the ph/NPE ordinate.
    assert np.isclose(
        result.loc[0, "vuv_per_npe_err"],
        abs(result.loc[0, "vuv_per_npe"]) * relative_population_error,
    )
    assert np.isclose(
        result.loc[0, "uv_per_npe_err"],
        abs(result.loc[0, "uv_per_npe"]) * relative_population_error,
    )


def test_channel_fractions_are_normalised(monkeypatch):
    import numpy as np
    import pandas as pd
    from scintillation.predictions import alpha_campaign as campaign

    monkeypatch.setattr(campaign, "_load_arcf4_parameters", lambda _paths: np.ones(11))
    monkeypatch.setattr(campaign, "_load_ar2nd_parameters_with_hydrocarbon_proxy", lambda _paths: {"triplet_weight": 1.0})
    monkeypatch.setattr(
        campaign,
        "theory_yield_ar2nd_continium",
        lambda *args, **kwargs: (20.0, 8.0, 12.0),
    )
    monkeypatch.setattr(
        campaign,
        "theory_yield_uv_components",
        lambda *args, **kwargs: {
            "total": 10.0,
            "cf4_plus": 4.0,
            "ar_third_continuum": 3.0,
            "cf3": 3.0,
        },
    )

    row = {
        "npe": 20.0,
        "gain": 5.0,
        "gain_error": 0.5,
        "pressure_bar": 1.0,
        "campaign_mixture": "ArCF4",
        "fraction_ar": 0.95,
        "fraction_cf4": 0.05,
        "fraction_ic4h10": 0.0,
        "fraction_n2": 0.0,
        "fraction_co2": 0.0,
        "fraction_ch4": 0.0,
    }
    for column in campaign._POPULATION_COLUMNS:
        row[column] = 1.0
    result = campaign._evaluate_secondary_yields(pd.DataFrame([row]), object())
    assert np.isclose(
        result.loc[0, "uv_cf4_plus_fraction"]
        + result.loc[0, "uv_ar_third_fraction"]
        + result.loc[0, "uv_cf3_fraction"],
        1.0,
    )
    assert np.isclose(
        result.loc[0, "vuv_singlet_fraction"]
        + result.loc[0, "vuv_triplet_fraction"],
        1.0,
    )


def test_campaign_legend_order_is_physical() -> None:
    from scintillation.predictions import alpha_campaign as campaign

    frame = pd.DataFrame(
        [
            {"composition_label": "ternary high", "fraction_cf4": 0.02, "fraction_ic4h10": 0.20},
            {"composition_label": "binary low", "fraction_cf4": 0.10, "fraction_ic4h10": 0.00},
            {"composition_label": "ternary low", "fraction_cf4": 0.02, "fraction_ic4h10": 0.03},
            {"composition_label": "binary high", "fraction_cf4": 0.20, "fraction_ic4h10": 0.00},
        ]
    )
    assert campaign._ordered_composition_labels(frame) == [
        "binary high",
        "binary low",
        "ternary low",
        "ternary high",
    ]


def test_campaign_marker_fill_is_lighter_but_not_white() -> None:
    import numpy as np
    from matplotlib.colors import to_rgba
    from scintillation.predictions import alpha_campaign as campaign

    original = np.asarray(to_rgba("tab:blue"))
    face = np.asarray(campaign._light_marker_face("tab:blue"))
    assert np.all(face[:3] > original[:3])
    assert np.all(face[:3] < 1.0)
    assert np.isclose(face[3], original[3])


def test_retired_component_plots_are_removed_without_touching_main_outputs(tmp_path) -> None:
    from scintillation.predictions import alpha_campaign as campaign

    uv_root = tmp_path / "uv"
    vuv_root = tmp_path / "vuv"
    uv_root.mkdir()
    vuv_root.mkdir()

    retired_uv = uv_root / "uv_channel_fractions_vs_gain_mix.pdf"
    retired_vuv = vuv_root / "vuv_channel_fractions_vs_efield_mix.pdf"
    kept_uv = uv_root / "uv_vs_gain_all_mixtures.pdf"
    kept_vuv = vuv_root / "vuv_vs_gain_all_mixtures.pdf"
    for path in (retired_uv, retired_vuv, kept_uv, kept_vuv):
        path.write_bytes(b"pdf")

    campaign._remove_deprecated_component_plots(uv_root, vuv_root)

    assert not retired_uv.exists()
    assert not retired_vuv.exists()
    assert kept_uv.exists()
    assert kept_vuv.exists()
