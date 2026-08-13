from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DATA = ROOT / "data"
for folder in (ROOT, SRC, DATA):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from scintillation.models.ar_second_continuum import (  # noqa: E402
    apply_ch4_isobutane_proxy,
    read_ar2nd_parameters,
    theory_yield_ar2nd_continium,
)


def _load_analysis_module():
    path = ROOT / "src/scintillation/_compat/data_scripts/Analysis_secondary_garfield.py"
    spec = importlib.util.spec_from_file_location("analysis_secondary_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _population_row() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "concentration": [0.05],
            "Ar_dbleStar": [100.0],
            "Ar_1s2_1s3": [20.0],
            "Ar_1s4_1s5": [30.0],
        }
    )


def _parameters_with_isobutane_same_as_cf4() -> dict[str, object]:
    params = read_ar2nd_parameters(ROOT / "data" / "reference" / "parameters" / "Ar2nd_continium.csv")
    params = dict(params)
    params["k_Ar_dbleStar_Q_m3_s__IC4H10"] = params["k_Ar_dbleStar_Q_m3_s__CF4"]
    params["k_Ar_4s_Q_m3_s__IC4H10"] = params["k_Ar_4s_Q_m3_s__CF4"]
    params["k_Ar2star_Q_m3_s__IC4H10"] = params["k_Ar2star_Q_m3_s__CF4"]
    return params


def test_binary_api_remains_numerically_identical_to_mapping_api():
    params = read_ar2nd_parameters(ROOT / "data" / "reference" / "parameters" / "Ar2nd_continium.csv")
    data = _population_row()
    legacy = theory_yield_ar2nd_continium(
        params, data, 0.05, 1.0, gas_mixture="ArCF4", energy_xray_ev=15.0
    )
    mapped = theory_yield_ar2nd_continium(
        params,
        data,
        {"CF4": 0.05},
        1.0,
        gas_mixture="ArCF4",
        additive_fractions={"CF4": 0.05},
        energy_xray_ev=15.0,
    )
    assert np.isclose(legacy, mapped, rtol=0.0, atol=1.0e-14)


def test_ternary_losses_add_linearly_when_rates_are_equal():
    params = _parameters_with_isobutane_same_as_cf4()
    data = _population_row()
    binary = theory_yield_ar2nd_continium(
        params, data, 0.05, 1.0, gas_mixture="ArCF4", energy_xray_ev=15.0
    )
    ternary = theory_yield_ar2nd_continium(
        params,
        data,
        {"CF4": 0.02, "IC4H10": 0.03},
        1.0,
        gas_mixture="ArIsoCF4",
        additive_fractions={"CF4": 0.02, "IC4H10": 0.03},
        energy_xray_ev=15.0,
    )
    assert np.isclose(binary, ternary, rtol=1.0e-12, atol=1.0e-14)


def test_isobutane_uses_common_effective_quenching():
    params = apply_ch4_isobutane_proxy(
        read_ar2nd_parameters(ROOT / "data" / "reference" / "parameters" / "Ar2nd_continium.csv")
    )
    for base in ("k_Ar_dbleStar_Q_m3_s", "k_Ar_4s_Q_m3_s", "k_Ar2star_Q_m3_s"):
        assert np.isclose(params[f"{base}__IC4H10"], 7.10e-16)
    value = theory_yield_ar2nd_continium(
        params,
        _population_row(),
        {"CF4": 0.02, "IC4H10": 0.03},
        1.0,
        gas_mixture="ArIsoCF4",
        additive_fractions={"CF4": 0.02, "IC4H10": 0.03},
        energy_xray_ev=15.0,
    )
    assert np.isfinite(value)


def test_ternary_root_metadata_is_preserved():
    analysis = _load_analysis_module()
    path = Path("ar_95.0_iso_3.0_cf4_2.0_49.4kVcm_1.000bar_0.1500mm_20npe.root")
    meta = analysis.parse_root_metadata(path, gas_concentration="cf4")
    assert meta["gas_mixture"] == "ArIsoCF4"
    assert meta["n_components"] == 3
    assert meta["gas3"] == "cf4"
    assert meta["fraction_ar"] == 95.0
    assert meta["fraction_ic4h10"] == 3.0
    assert meta["fraction_cf4"] == 2.0
    assert meta["concentration"] == 2.0


def test_invalid_composition_sum_is_rejected():
    analysis = _load_analysis_module()
    path = Path("ar_95.0_iso_10.0_cf4_2.0_49.4kVcm_1.000bar_0.1500mm_20npe.root")
    with np.testing.assert_raises_regex(ValueError, "suman"):
        analysis.parse_root_metadata(path, gas_concentration="cf4")
