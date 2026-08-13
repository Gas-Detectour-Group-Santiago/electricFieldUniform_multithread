from __future__ import annotations

import numpy as np
import pandas as pd

from scintillation.plotting.spectra_recipe_engine import _generated_groups


def test_generated_grid_keeps_point_one_percent_from_float32_cache() -> None:
    """0.1% must survive the compact float32 spectra-cache round trip."""
    frame = pd.DataFrame(
        {
            "gas_mixture": ["ArCF4", "ArCF4"],
            "concentration_percent": np.asarray([0.1, 0.1], dtype=np.float32),
            "pressure_bar": np.asarray([1.0, 1.0], dtype=np.float32),
            "component": ["total", "total"],
            "wavelength_nm": np.asarray([250.0, 300.0], dtype=np.float32),
            "intensity_ph_MeV_nm": np.asarray([2.0, 3.0], dtype=np.float32),
        }
    )

    groups = _generated_groups(frame, gas="ArCF4", components=("total",))

    assert (0.1, 1.0) in groups
    assert groups[(0.1, 1.0)]["intensity_ph_MeV_nm"].tolist() == [2.0, 3.0]
