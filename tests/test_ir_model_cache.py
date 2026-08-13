from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scintillation.models import ar_cf4_ir, ar_n2_ir


@pytest.mark.parametrize(
    ("module", "prefix"),
    (
        (ar_cf4_ir, "ArCF4"),
        (ar_n2_ir, "ArN2"),
    ),
)
def test_ir_populations_are_interpolated_once_per_grid(monkeypatch, module, prefix) -> None:
    concentration = np.asarray([0.001, 0.01, 0.05, 0.10], dtype=float)
    frame = pd.DataFrame(
        {
            "concentration": concentration,
            "Ar_696": [1.0, 1.1, 1.2, 1.3],
            "Ar_727": [2.0, 2.1, 2.2, 2.3],
            "Ar_750": [3.0, 3.1, 3.2, 3.3],
            "Ar_763": [4.0, 4.1, 4.2, 4.3],
            "Ar_772": [5.0, 5.1, 5.2, 5.3],
            "Ar_794": [6.0, 6.1, 6.2, 6.3],
        }
    )
    parameters = np.linspace(0.05, 1.20, 24)
    target = np.asarray([0.002, 0.02, 0.08], dtype=float)

    module._INTERPOLATED_POPULATION_CACHE.clear()
    original = module.interpolation
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "interpolation", counted)

    getattr(module, f"theory_yield_{prefix}_Ir_696")(parameters, frame, target, 1.0)
    getattr(module, f"theory_yield_{prefix}_Ir_727")(parameters, frame, target, 2.0)
    assert calls == 1

    getattr(module, f"theory_yield_{prefix}_Ir_750")(
        parameters,
        frame,
        np.asarray([0.003, 0.03], dtype=float),
        1.0,
    )
    assert calls == 2


def test_primary_arcf4_yields_are_interpolated_once_per_grid(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "concentration": [0.001, 0.01, 0.05, 0.10],
            "CF3": [1.0, 1.1, 1.2, 1.3],
            "Ar_dbleStar": [2.0, 2.1, 2.2, 2.3],
            "CF4": [3.0, 3.1, 3.2, 3.3],
            "Ar_3rd": [4.0, 4.1, 4.2, 4.3],
        }
    )
    target = np.asarray([0.002, 0.02, 0.08], dtype=float)
    parameters = np.linspace(0.05, 0.60, 11)
    ar_cf4 = __import__("scintillation.models.ar_cf4", fromlist=["dummy"])
    ar_cf4._INTERPOLATED_YIELD_CACHE.clear()

    original = ar_cf4.PchipInterpolator
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(ar_cf4, "PchipInterpolator", counted)
    ar_cf4.theory_yield_vis(parameters, frame, target, 1.0)
    ar_cf4.theory_yield_uv(parameters, frame, target, 2.0)
    assert calls == 1


def test_primary_arn2_yields_are_interpolated_once_per_grid(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "concentration": [0.001, 0.01, 0.05, 0.10],
            "N2_star": [1.0, 1.1, 1.2, 1.3],
            "Ar_meta": [2.0, 2.1, 2.2, 2.3],
            "Ar_res": [3.0, 3.1, 3.2, 3.3],
            "Ar_dbleStar": [4.0, 4.1, 4.2, 4.3],
        }
    )
    target = np.asarray([0.002, 0.02, 0.08], dtype=float)
    parameters = np.linspace(0.05, 0.65, 13)
    ar_n2 = __import__("scintillation.models.ar_n2", fromlist=["dummy"])
    ar_n2._INTERPOLATED_YIELD_CACHE.clear()

    original = ar_n2.PchipInterpolator
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(ar_n2, "PchipInterpolator", counted)
    ar_n2.theory_yield_N2_uv(parameters, frame, target, 1.0)
    ar_n2.theory_yield_N2_uv(parameters, frame, target, 2.0)
    assert calls == 1
