from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_runner_supports_xenon_and_penning_switch():
    source = (ROOT / "run_campaign.py").read_text()
    assert '"xe": "xe"' in source
    assert '"xenon": "xe"' in source
    assert "penning: bool = False" in source
    assert 'config.get("penning", False)' in source
    assert '"--penning"' in source
    assert 'suffix += "_penning"' in source
    assert "str(int(job.options.penning))" in source


def test_diego_campaign_is_arxe_99_1_at_requested_fields():
    config = yaml.safe_load((ROOT / "campaign_DiegoR.yaml").read_text())
    assert config["penning"] is False
    assert len(config["field_sets"]) == 1
    field_set = config["field_sets"][0]
    assert field_set["components"] == {"ar": 99, "xe": 1}
    assert field_set["pressures_bar"] == [4]
    assert field_set["gap_mm"] == 5.0
    assert field_set["fields_kv_cm"] == [5, 6, 7, 8, 9, 10, 11]


def test_cpp_wires_penning_to_medium_magboltz_and_root_metadata():
    source = (ROOT / "uniformE.cxx").read_text()
    assert "gas.EnablePenningTransfer();" in source
    assert "gas.DisablePenningTransfer();" in source
    assert 'gas_data.Branch("penning"' in source


def test_gui_exposes_penning_checkbox():
    source = (ROOT / "gui.py").read_text()
    assert 'QCheckBox("Penning transfer")' in source
    assert 'read_yaml_bool(text, "penning", False)' in source
    assert 'write_yaml_bool(text, "penning", self.penning.isChecked())' in source
