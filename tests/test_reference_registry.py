from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.reference_registry import load_gases, load_mixtures, parse_components


def test_default_registry_is_valid() -> None:
    gases = load_gases(ROOT / "config" / "reference" / "gases.csv")
    mixtures = load_mixtures(ROOT / "config" / "reference" / "mixtures.csv", gases)
    assert {gas.garfield_label for gas in gases} >= {"ar", "cf4", "ic4h10", "co2", "ch4"}
    ternary = next(mix for mix in mixtures if mix.mixture_id == "ArIsoCF4")
    assert ternary.components == (("ar", 95.0), ("ic4h10", 3.0), ("cf4", 2.0))


def test_components_preserve_order() -> None:
    assert parse_components("ar=77;ic4h10=20;cf4=3") == (
        ("ar", 77.0),
        ("ic4h10", 20.0),
        ("cf4", 3.0),
    )
