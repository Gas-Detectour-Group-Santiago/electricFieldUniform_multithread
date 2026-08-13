from __future__ import annotations

import unittest
from pathlib import Path
import numpy as np
from scintillation.core.paths import ProjectPaths
from scintillation.core.registry import ProjectRegistry
from scintillation.physics.parameters import build_project_parameter_registry
from scintillation.physics.second_continuum import read_ar2nd_parameters, available_additives
from scintillation.predictions.results import DomainStatus, PredictionResult, UncertaintyBand


ROOT = Path(__file__).resolve().parents[1]


class ArchitectureTests(unittest.TestCase):
    def test_required_directories_exist(self) -> None:
        paths = ProjectPaths.from_root(ROOT)
        for path in (paths.config, paths.raw, paths.processed, paths.reference, paths.src):
            self.assertTrue(path.exists(), path)


    def test_human_facing_structure_is_unique(self) -> None:
        assert (ROOT / "run.py").is_file()
        assert (ROOT / "src/scintillation/models/ar_cf4.py").is_file()
        assert (ROOT / "src/scintillation/models/ar_n2.py").is_file()
        assert (ROOT / "src/scintillation/pipelines/secondary.py").is_file()
        assert not (ROOT / "workflows").exists()
        assert not (ROOT / "models").exists()
        assert not (ROOT / "secondary_predictions").exists()

    def test_registry_is_consistent(self) -> None:
        registry = ProjectRegistry.load(ROOT)
        self.assertIn("ArCF4", registry.mixtures)
        self.assertIn("ArN2", registry.mixtures)
        self.assertIn("ArJoint_IR_primary", registry.fits)
        self.assertGreaterEqual(len(registry.secondary_parameter_sets), 4)

    def test_fit_parameters_are_not_cross_resolved(self) -> None:
        registry = build_project_parameter_registry(ROOT)
        rows = registry.frame
        selected = rows.loc[rows["comparison_group"] == "argon.ir.696.lifetime"]
        ids = set(selected["physical_parameter_id"].astype(str))
        self.assertIn("fit.ArCF4_IR_primary.tau_CF4_696", ids)
        self.assertIn("fit.ArN2_IR_primary.tau_N2_696", ids)
        self.assertIn("fit.ArJoint_IR_primary.tau_joint_696", ids)
        self.assertEqual(len(ids), 3)
        with self.assertRaises(RuntimeError):
            registry.resolved_physical_parameters()

    def test_second_continuum_additives_are_dynamic(self) -> None:
        params = read_ar2nd_parameters(ROOT / "data/reference/parameters/Ar2nd_continium.csv")
        additives = set(available_additives(params))
        self.assertIn("CF4", additives)
        self.assertIn("N2", additives)

    def test_prediction_result_validates_shapes(self) -> None:
        x = np.array([1.0, 2.0])
        central = np.array([10.0, 11.0])
        band = UncertaintyBand(np.array([9.0, 10.0]), np.array([11.0, 12.0]), "stat")
        result = PredictionResult(
            x=x,
            central=central,
            unit="ph/MeV",
            bands={"stat": band},
            domain_status=np.array([DomainStatus.SIMULATED, DomainStatus.INTERPOLATED]),
        )
        self.assertEqual(result.central.shape, (2,))


if __name__ == "__main__":
    unittest.main()
