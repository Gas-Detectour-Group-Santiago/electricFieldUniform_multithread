import unittest

import numpy as np

from alpha_model import AlphaFit
from fit_diagnostics import _alpha_curve_from_gains, _field_curve_for_gains


class FitDiagnosticsTests(unittest.TestCase):
    def make_saturating_fit(self):
        return AlphaFit(
            A=720.0,
            B=18.0,
            m=0.0,
            n=1.0,
            covariance=[[0.0] * 4 for _ in range(4)],
            valid_reduced_field=[22.0, 50.0],
            n_points=8,
            relative_rmse=0.0,
            model_name="townsend_2p",
            model_label="Townsend 2p",
        )

    def test_high_gain_curve_stops_at_local_extrapolation_limit(self):
        fit = self.make_saturating_fit()
        requested = np.concatenate(([1.0 + 1.0e-9], np.logspace(0.001, 7.0, 240)))
        fields, gains, _ = _field_curve_for_gains(
            fit, 1.0, 0.15, requested,
            max_upper_extrapolation_fraction=0.75,
        )
        self.assertGreater(len(gains), 2)
        self.assertLess(gains[-1], 1.0e7)
        maximum_local_field = 50.0 + 0.75 * (50.0 - 22.0)
        self.assertLessEqual(fields[-1], maximum_local_field + 1.0e-9)

    def test_gain_one_maps_to_zero_alpha(self):
        alpha = _alpha_curve_from_gains(np.asarray([1.0, 10.0, 100.0]), 0.15)
        self.assertEqual(alpha[0], 0.0)
        self.assertGreater(alpha[1], 0.0)
        self.assertGreater(alpha[2], alpha[1])


if __name__ == "__main__":
    unittest.main()
