from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src/scintillation/_compat/project/primary_fits/auxiliares/fit_exports.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("fit_exports_test", MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_toy_correlation_diagonal_is_writable():
    module = _load_module()
    toys = np.asarray(
        [
            [1.0, 5.0, 7.0],
            [2.0, 5.0, 9.0],
            [3.0, 5.0, 8.0],
        ],
        dtype=float,
    )
    covariance, correlation, n_valid = module.toy_covariance_correlation(
        toys, ["free_a", "fixed", "free_b"]
    )
    assert n_valid == 3
    assert np.isclose(correlation.loc["free_a", "free_a"], 1.0)
    assert np.isnan(correlation.loc["fixed", "fixed"])
    assert covariance.shape == (3, 3)
