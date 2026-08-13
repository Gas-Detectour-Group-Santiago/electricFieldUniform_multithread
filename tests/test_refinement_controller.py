import math
import unittest

try:
    from run_campaign import (
        CampaignOptions,
        Family,
        RefinementOptions,
        Target,
        _enforce_last_attempt_progress,
        interpolate_field,
        propose_field,
    )
    from alpha_model import AlphaPoint, gain_to_alpha
    IMPORT_ERROR = None
except Exception as error:  # pragma: no cover - dependency guard for minimal CI
    IMPORT_ERROR = error


def make_point(field, gain):
    gap = 0.15
    return AlphaPoint(
        mixture="ArCF4Iso",
        fraction=10.0,
        pressure_bar=1.0,
        gap_mm=gap,
        field_v_cm=field * 1000.0,
        gain=gain,
        gain_error=0.05 * gain,
        alpha_effective=gain_to_alpha(gain, gap),
        alpha_error=0.05 / (0.1 * gap),
        npe=20,
        root=f"{field}.root",
        composition="ar_87__cf4_10__ic4h10_3",
    )


@unittest.skipIf(IMPORT_ERROR is not None, f"run_campaign dependencies unavailable: {IMPORT_ERROR}")
class RefinementControllerTests(unittest.TestCase):
    def test_last_overshoot_can_never_raise_field(self):
        last = make_point(51.3, 54922.7)
        field, _ = _enforce_last_attempt_progress(
            60.01e3,
            [last],
            last,
            Target(1.0, 1000.0),
            RefinementOptions(),
        )
        self.assertLess(field, last.field_v_cm)

    def test_last_undershoot_can_never_lower_field(self):
        last = make_point(40.0, 100.0)
        field, _ = _enforce_last_attempt_progress(
            35.0e3,
            [last],
            last,
            Target(1.0, 1000.0),
            RefinementOptions(),
        )
        self.assertGreater(field, last.field_v_cm)

    def test_noisy_non_monotonic_points_still_bracket_safely(self):
        points = [
            make_point(40.0, 100.0),
            make_point(42.0, 800.0),
            make_point(44.0, 700.0),  # noisy inversion
            make_point(46.0, 10000.0),
        ]
        field = interpolate_field(points, Target(1.0, 1000.0))
        self.assertIsNotNone(field)
        self.assertGreater(field, 40.0e3)
        self.assertLess(field, 46.0e3)


if __name__ == "__main__":
    unittest.main()
