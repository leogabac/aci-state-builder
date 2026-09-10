from pathlib import Path
import tempfile
import unittest

import numpy as np

from aci_state_builder.formats import StateFormatError, read_csv, to_frame, write_csv
from aci_state_builder.energy import (
    EnergyParameters, calculate_energy, dipole_prefactor_pn_nm_um3,
    geometric_dipole_sum_pbc,
)
from aci_state_builder.geometry import periodic_square, periodic_vertex_charges
from aci_state_builder.model import IceDocument
from aci_state_builder.presets import af2, af4
from aci_state_builder.validation import validate


WORKSPACE = Path(__file__).resolve().parents[2]


class GeometryTests(unittest.TestCase):
    def test_historical_ordering(self) -> None:
        document = periodic_square(2, 2, 8.0, 3.0)
        self.assertEqual(len(document.traps), 8)
        np.testing.assert_allclose(document.traps[0].center, [4, 0, 0])
        np.testing.assert_allclose(document.traps[3].center, [12, 8, 0])
        np.testing.assert_allclose(document.traps[4].center, [0, 4, 0])
        np.testing.assert_allclose(document.traps[7].center, [8, 12, 0])

    def test_historical_preset_charge_patterns(self) -> None:
        document = periodic_square(10, 10, 8.374011537, 3.0)
        document.set_occupancies(af2(document))
        values, counts = np.unique(periodic_vertex_charges(document), return_counts=True)
        self.assertEqual(dict(zip(values.tolist(), counts.tolist(), strict=True)), {-2: 50, 2: 50})
        document.set_occupancies(af4(document))
        values, counts = np.unique(periodic_vertex_charges(document), return_counts=True)
        self.assertEqual(dict(zip(values.tolist(), counts.tolist(), strict=True)), {-4: 50, 4: 50})

    def test_snapshot_preserves_imported_displacement(self) -> None:
        document = periodic_square(2, 2, 8.0, 3.0)
        document.traps[0].displacement = np.array([1.2, 0.1, 0.0])
        before = document.state_snapshot()
        document.flip_indices([0])
        np.testing.assert_allclose(document.traps[0].displacement, [-1.2, -0.1, 0.0])
        document.restore_snapshot(before)
        self.assertEqual(document.traps[0].occupancy, 1)
        np.testing.assert_allclose(document.traps[0].displacement, [1.2, 0.1, 0.0])

    def test_charge_is_in_minus_out_across_periodic_seam(self) -> None:
        document = periodic_square(2, 2, 8.0, 3.0)
        # Flipping the horizontal trap from x=1 to x=0 reverses a seam-crossing edge.
        document.flip_indices([1])
        charge = periodic_vertex_charges(document)
        self.assertEqual(int(charge[0, 1]), 2)
        self.assertEqual(int(charge[0, 0]), -2)
        self.assertEqual(int(charge.sum()), 0)


class FormatTests(unittest.TestCase):
    def test_canonical_csv_round_trip(self) -> None:
        source = periodic_square(3, 3, 8.0, 3.0)
        source.set_occupancies(af2(source))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.csv"
            write_csv(source, path)
            loaded = read_csv(path)
        self.assertEqual((loaded.geometry, loaded.nx, loaded.ny), ("square", 3, 3))
        np.testing.assert_array_equal(loaded.occupancies(), source.occupancies())
        self.assertEqual(to_frame(loaded).columns, list(("id", "x", "y", "z", "dx", "dy", "dz", "cx", "cy", "cz")))
        self.assertEqual(validate(loaded), [])

    def test_rejects_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.csv"
            path.write_text(
                "frame,id,x,y,z,dx,dy,dz,cx,cy,cz\n"
                "0,0,1,0,0,1,0,0,1,0,0\n"
                "1,0,1,0,0,-1,0,0,-1,0,0\n", encoding="utf-8")
            with self.assertRaisesRegex(StateFormatError, "multi-frame trajectory"):
                read_csv(path)

    def test_project_json_round_trip(self) -> None:
        source = periodic_square(2, 3, 7.5, 2.5)
        source.set_occupancies(af4(source))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.aci.json"
            source.save(path)
            loaded = IceDocument.load(path)
        self.assertEqual(loaded.to_dict(), source.to_dict())

    def test_surviving_af2_fixture(self) -> None:
        fixture = WORKSPACE / "stuckgs/data/configurations/af2/10.csv"
        if not fixture.exists(): self.skipTest("stuckgs fixture is not mounted")
        loaded = read_csv(fixture)
        expected = periodic_square(10, 10, loaded.lattice_constant, loaded.trap_separation)
        np.testing.assert_array_equal(loaded.occupancies(), af2(expected))

    def test_surviving_af4_fixture(self) -> None:
        fixture = WORKSPACE / "stuckgs/data/configurations/af4/10.csv"
        if not fixture.exists(): self.skipTest("stuckgs fixture is not mounted")
        loaded = read_csv(fixture)
        expected = periodic_square(10, 10, loaded.lattice_constant, loaded.trap_separation)
        np.testing.assert_array_equal(loaded.occupancies(), af4(expected))


class EnergyTests(unittest.TestCase):
    def test_two_particle_energy_and_minimum_image(self) -> None:
        positions = np.array([[0.5, 0, 0], [9.5, 0, 0]])
        # They are one micrometre apart through the periodic x seam and parallel to B.
        self.assertAlmostEqual(geometric_dipole_sum_pbc(positions, (10, 10)), -2.0)

    def test_field_scaling_and_project_defaults(self) -> None:
        document = periodic_square(3, 3, 8.0, 3.0)
        low = calculate_energy(document, EnergyParameters(field_mT=5))
        high = calculate_energy(document, EnergyParameters(field_mT=10))
        self.assertAlmostEqual(high.total_pn_nm / low.total_pn_nm, 4.0)
        self.assertGreater(dipole_prefactor_pn_nm_um3(EnergyParameters()), 0)

    def test_energy_parameters_survive_project_round_trip(self) -> None:
        document = periodic_square(2, 2, 8.0, 3.0)
        document.energy_parameters = EnergyParameters(
            particle_radius_um=5, susceptibility=0.0576, field_mT=15,
            field_angle_deg=90, cutoff_um=40,
        ).to_dict()
        loaded = IceDocument.from_dict(document.to_dict())
        self.assertEqual(loaded.energy_parameters, document.energy_parameters)


if __name__ == "__main__":
    unittest.main()
