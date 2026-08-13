from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_hlevels_exporter_contains_all_required_catalogues():
    source = (ROOT / "data/reference/levels/obtain_levels.cxx").read_text(encoding="utf-8")
    expected = {
        "Ar_level_data.csv", "He_level_data.csv", "CF4_level_data.csv",
        "N2_level_data.csv", "iC4H10_level_data.csv", "CO2_level_data.csv",
        "CH4_level_data.csv", "ArCF4_level_data.csv", "ArN2_level_data.csv",
        "HeCF4_level_data.csv", "ArIso_level_data.csv",
        "ArIsoCF4_level_data.csv", "ArCO2_level_data.csv",
        "ArCH4_level_data.csv",
    }
    assert all(name in source for name in expected)


def test_cross_section_exporter_contains_all_pure_gases():
    source = (
        ROOT / "src/scintillation/_compat/project/cross_sections/cross_section.cxx"
    ).read_text(encoding="utf-8")
    assert all(token in source for token in ('"ar"', '"he"', '"cf4"', '"n2"', '"ic4h10"', '"co2"', '"ch4"'))


def test_reference_runners_are_exposed_in_the_app():
    page = (ROOT / "app/desktop/reference_pages.py").read_text(encoding="utf-8")
    assert 'tools/run_hlevels.py' in page
    assert 'tools/run_cross_sections.py' in page
