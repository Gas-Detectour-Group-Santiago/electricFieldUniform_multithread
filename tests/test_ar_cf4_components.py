from __future__ import annotations

import numpy as np
import pandas as pd

from scintillation.models.ar_cf4 import theory_yield_uv, theory_yield_uv_components


def _degrad() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "concentration": [0.0, 0.05, 0.10],
            "CF3": [2.0, 3.0, 4.0],
            "Ar_dbleStar": [5.0, 4.0, 3.0],
            "CF4": [1.0, 2.0, 3.0],
            "Ar_3rd": [7.0, 6.0, 5.0],
        }
    )


def _parameters() -> np.ndarray:
    return np.array([1.0, 0.2, 0.3, 0.4, 0.5, 0.02, 0.03, 0.6, 50.05, 0.7, 0.1])


def test_uv_components_sum_exactly_to_total() -> None:
    components = theory_yield_uv_components(_parameters(), _degrad(), 0.05, 1.0)
    assert np.isclose(
        components["total"],
        components["cf4_plus"] + components["ar_third_continuum"] + components["cf3"],
    )
    direct = theory_yield_uv(_parameters(), _degrad(), 0.05, 1.0, activate_components=True)
    assert np.allclose(
        direct,
        [
            components["total"],
            components["cf4_plus"],
            components["ar_third_continuum"],
            components["cf3"],
        ],
    )


def test_cf4_direct_parameter_does_not_overwrite_cf3_branch() -> None:
    first = _parameters()
    second = first.copy()
    second[7] *= 1.8
    before = theory_yield_uv_components(first, _degrad(), 0.05, 1.0)
    after = theory_yield_uv_components(second, _degrad(), 0.05, 1.0)
    assert not np.isclose(before["cf4_plus"], after["cf4_plus"])
    assert np.isclose(before["cf3"], after["cf3"])
