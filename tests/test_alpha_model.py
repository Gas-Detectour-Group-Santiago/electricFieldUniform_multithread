import math
import unittest

from alpha_model import (
    AlphaPoint,
    alpha_to_gain,
    combine_duplicate_points,
    field_for_gain,
    fit_alpha,
    gain_to_alpha,
    model_reduced_alpha,
    predict_gain,
)


def point(field_kv_cm, gain, *, npe=100, gain_error=None, pressure=1.0):
    if gain_error is None:
        gain_error = 0.05 * gain
    gap = 0.15
    return AlphaPoint(
        mixture="Test",
        fraction=10.0,
        pressure_bar=pressure,
        gap_mm=gap,
        field_v_cm=1000.0 * field_kv_cm,
        gain=gain,
        gain_error=gain_error,
        alpha_effective=gain_to_alpha(gain, gap),
        alpha_error=gain_error / (gain * 0.1 * gap),
        npe=npe,
        root=f"{field_kv_cm}.root",
        composition="ar_90__cf4_10",
    )


class AlphaModelTests(unittest.TestCase):
    def test_duplicate_runs_are_combined_once(self):
        combined = combine_duplicate_points([
            point(40.0, 100.0, npe=20),
            point(40.0005, 120.0, npe=20),
            point(45.0, 1000.0, npe=20),
        ])
        self.assertEqual(len(combined), 2)
        self.assertEqual(combined[0].n_runs, 2)
        self.assertAlmostEqual(combined[0].gain, 110.0, places=6)
        self.assertEqual(combined[0].npe, 40)

    def test_progressive_model_selects_supported_order(self):
        points = []
        gap = 0.15
        for index, field in enumerate(range(20, 65, 4)):
            alpha = model_reduced_alpha(field, 0.8, 18.0, 1.3, 1.0)
            gain = alpha_to_gain(float(alpha), gap)
            points.append(point(field, gain, gain_error=0.03 * gain))
        fit = fit_alpha(points)
        self.assertIsNotNone(fit)
        self.assertEqual(fit.model_name, "townsend_3p")
        self.assertTrue(fit.usable_for_prediction)
        self.assertLess(fit.cv_median_factor, 1.05)

    def test_inversion_is_monotonic_and_self_consistent(self):
        points = []
        gap = 0.15
        for field in range(20, 65, 4):
            alpha = model_reduced_alpha(field, 0.8, 18.0, 1.3, 1.0)
            gain = alpha_to_gain(float(alpha), gap)
            points.append(point(field, gain, gain_error=0.03 * gain))
        fit = fit_alpha(points)
        fields = [field_for_gain(fit, 1.0, gap, gain) for gain in (10, 100, 1000, 10000)]
        self.assertEqual(fields, sorted(fields))
        for target_gain, field in zip((10, 100, 1000, 10000), fields):
            predicted = predict_gain(fit, 1.0, gap, field)
            self.assertAlmostEqual(math.log(predicted / target_gain), 0.0, places=8)


if __name__ == "__main__":
    unittest.main()
