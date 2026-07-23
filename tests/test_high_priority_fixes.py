import contextlib
import io
import unittest

import numpy as np
import qutip

from mbl_eigen import level_repulsion
from mbl_eigen.cli import build_mbldtc_parser
from mbl_eigen.cli import build_mbl_parser
from mbl_eigen.cli import build_qmbs_parser
from mbl_eigen.eigensolver import solve_general_eigenproblem
from mbl_eigen.eigensolver import solve_hermitian_eigenproblem
from mbl_eigen.eigensolver import _resolve_jax_device
from mbl_eigen.mbl_app import _compute_overlap_matrix
from mbl_eigen.mbl_app import _time_grid
from mbl_eigen.mbl_model import build_mbl_hamiltonian
from mbl_eigen.mbl_model import sample_mbl_disorder
from mbl_eigen.mbl_model import spin_operators
from mbl_eigen.qiskit_propagators import sample_mbldtc_angles
from mbl_eigen.reflection import reflection_about_center


class HighPriorityFixTests(unittest.TestCase):
    def test_circular_spacing_includes_wraparound_gap(self):
        phases = np.array([0.1, 1.0, 2.0])
        spacings = np.diff(np.concatenate((phases, [phases[0] + 2.0 * np.pi])))
        expected = np.mean(
            np.minimum(spacings[:-1], spacings[1:])
            / np.maximum(spacings[:-1], spacings[1:])
        )

        actual = level_repulsion.calc_mean_adjacent_level_spacing_ratio(
            phases,
            fraction_cutoff=0.0,
            circular_period=2.0 * np.pi,
        )

        self.assertAlmostEqual(actual, expected)

    def test_qmbs_parser_rejects_missing_and_small_systemsize(self):
        parser = build_qmbs_parser()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args([])
            with self.assertRaises(SystemExit):
                parser.parse_args([
                    "--systemsize", "1",
                    "--tduration", "1",
                    "--Delta", "0.1",
                ])

    def test_mbl_parser_rejects_zero_systemsize(self):
        parser = build_mbl_parser()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args([
                    "--systemsize", "0",
                    "--tduration", "1",
                    "--jIntMean", "1",
                    "--bFieldMean", "1",
                    "--jIntStd", "1",
                    "--bFieldStd", "1",
                    "--anglePolarPiMin", "0",
                    "--anglePolarPiMax", "1",
                ])

    def test_seed_is_available_on_random_workflow_parsers(self):
        mbldtc_args = build_mbldtc_parser().parse_args([
            "--systemsize", "2",
            "--thetaXPi", "0.76",
            "--seed", "123",
        ])
        mbl_args = build_mbl_parser().parse_args([
            "--systemsize", "2",
            "--tduration", "1",
            "--jIntMean", "1",
            "--bFieldMean", "1",
            "--jIntStd", "1",
            "--bFieldStd", "1",
            "--anglePolarPiMin", "0",
            "--anglePolarPiMax", "1",
            "--seed", "123",
        ])
        self.assertEqual(mbldtc_args.seed, 123)
        self.assertEqual(mbl_args.seed, 123)

    def test_seeded_mbl_disorder_is_reproducible(self):
        kwargs = dict(
            systemsize=3,
            jIntMean=1.0,
            jIntStd=0.2,
            bFieldMean=1.0,
            bFieldStd=0.2,
            anglePolarPiMin=0.0,
            anglePolarPiMax=1.0,
        )
        first = sample_mbl_disorder(rng=np.random.default_rng(123), **kwargs)
        second = sample_mbl_disorder(rng=np.random.default_rng(123), **kwargs)
        for first_values, second_values in zip(first, second):
            np.testing.assert_array_equal(first_values, second_values)

    def test_seeded_mbldtc_angles_are_reproducible(self):
        first = sample_mbldtc_angles(3, rng=np.random.default_rng(123))
        second = sample_mbldtc_angles(3, rng=np.random.default_rng(123))
        for first_values, second_values in zip(first, second):
            np.testing.assert_array_equal(first_values, second_values)

    def test_time_grid_uses_requested_duration(self):
        times = _time_grid(1.0)
        self.assertEqual(len(times), 17)
        self.assertAlmostEqual(times[-1], 1.0)

    def test_single_site_reflection_is_identity(self):
        swap = qutip.tensor(qutip.qeye(2), qutip.qeye(2))
        reflection = reflection_about_center(1, swap)
        self.assertTrue(reflection == qutip.qeye(2))

    def test_eigenvalue_only_paths_return_no_vectors(self):
        operator = qutip.Qobj([[1.0, 0.2], [0.2, 2.0]])
        with_vectors = solve_hermitian_eigenproblem(
            operator, backend="qobj", return_eigenvectors=True)
        without_vectors = solve_hermitian_eigenproblem(
            operator, backend="qobj", return_eigenvectors=False)
        np.testing.assert_allclose(
            without_vectors.eigenvalues, with_vectors.eigenvalues)
        self.assertIsNone(without_vectors.eigenvectors_array)

        general_without_vectors = solve_general_eigenproblem(
            operator, backend="qobj", return_eigenvectors=False)
        self.assertIsNone(general_without_vectors.eigenvectors_array)

    def test_torch_mps_is_rejected(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch extra is not installed")

        with self.assertRaisesRegex(ValueError, "does not support MPS"):
            solve_hermitian_eigenproblem(
                qutip.qeye(2), backend="torch", device="mps",
                return_eigenvectors=False)

    def test_jax_mps_requires_metal_platform(self):
        class Device:
            def __init__(self, platform):
                self.platform = platform

        class FakeJax:
            def __init__(self, platforms):
                self._devices = [Device(platform) for platform in platforms]

            def devices(self):
                return self._devices

        metal = _resolve_jax_device(FakeJax(["cpu", "metal"]), "mps")
        self.assertEqual(metal.platform, "metal")
        with self.assertRaises(ValueError):
            _resolve_jax_device(FakeJax(["cpu", "gpu"]), "mps")

    def test_overlap_matrix_has_all_channel_blocks_initialized(self):
        _, sigmax, sigmay, sigmaz = spin_operators()
        initial_arrays = []
        evolved_arrays = []

        for operator in (sigmax, sigmay, sigmaz):
            initial = np.empty((1,), dtype=object)
            initial[0] = operator
            evolved = np.empty((1, 1), dtype=object)
            evolved[0, 0] = operator
            initial_arrays.append(initial)
            evolved_arrays.append(evolved)

        overlap_matrix = _compute_overlap_matrix(
            initial_operator_arrays=tuple(initial_arrays),
            time_evolved_operator_arrays=tuple(evolved_arrays),
            hamiltonian_dims=[[2], [2]],
        )

        self.assertEqual(overlap_matrix.shape, (3, 3, 1))
        self.assertTrue(np.isfinite(overlap_matrix).all())


try:
    from qiskit.quantum_info import Operator
    from qutip.qip.operations import expand_operator
    from mbl_eigen.qiskit_propagators import build_mbldtc_floquet_circuit
    from mbl_eigen.qiskit_propagators import build_mbl_trotter_circuit
except ImportError:
    Operator = None
    build_mbldtc_floquet_circuit = None
    build_mbl_trotter_circuit = None


@unittest.skipIf(Operator is None, "Qiskit extra is not installed")
class QiskitOrderingTests(unittest.TestCase):
    def test_mbl_circuit_matches_quutip_tensor_order(self):
        systemsize = 3
        j_int = np.array([0.2, 0.7])
        b_field = np.array([0.4, 0.9, 1.3])
        theta = np.array([0.1, 0.5, 1.0])
        time = 0.03

        sigma0, sigmax, _, sigmaz = spin_operators()
        hamiltonian = build_mbl_hamiltonian(
            systemsize,
            j_int,
            b_field,
            theta,
            sigma0,
            sigmax,
            sigmaz,
        )
        circuit = build_mbl_trotter_circuit(
            systemsize,
            j_int,
            b_field,
            theta,
            time,
            trotter_steps=100,
        )

        qiskit_operator = Operator(circuit).data
        quutip_operator = (-1j * hamiltonian * time).expm().full()

        self.assertLess(
            np.linalg.norm(qiskit_operator - quutip_operator),
            1.0e-6,
        )

    def test_qiskit_rejects_invalid_real_inputs(self):
        common = dict(
            systemsize=2,
            jInt_samples=np.array([0.2]),
            bField_samples=np.array([0.4, 0.9]),
            theta_samples=np.array([0.1, 0.5]),
            time=0.1,
        )
        with self.assertRaises(ValueError):
            build_mbl_trotter_circuit(
                **{**common, "bField_samples": np.array([np.nan, 0.9])})
        with self.assertRaises(ValueError):
            build_mbl_trotter_circuit(
                **{**common, "theta_samples": np.array([1.0 + 0.0j, 0.5])})
        with self.assertRaises(ValueError):
            build_mbl_trotter_circuit(**{**common, "time": np.inf})
        with self.assertRaises(ValueError):
            build_mbldtc_floquet_circuit(
                2, np.nan, np.array([0.1, 0.2]), np.array([0.3]))
        with self.assertRaises(ValueError):
            build_mbldtc_floquet_circuit(
                2, 0.1, np.array([0.1, 0.2]), np.array([0.3]), cycles=1.0)

    def test_mbldtc_circuit_matches_quutip_tensor_order(self):
        systemsize = 3
        theta_x = 0.76 * np.pi
        phi_z = np.array([0.2, 0.7, 1.1])
        phi_zz = np.array([0.4, 0.9])
        cycles = 2

        sigmax = qutip.sigmax()
        sigmaz = qutip.sigmaz()
        sigmaz_sigmaz = qutip.tensor(sigmaz, sigmaz)
        rotation_x = (-1j * theta_x * 0.5 * sigmax).expm()
        rotation_z = [
            (-1j * angle * 0.5 * sigmaz).expm()
            for angle in phi_z
        ]
        interaction_zz = [
            (-1j * angle * 0.5 * sigmaz_sigmaz).expm()
            for angle in phi_zz
        ]

        quutip_operator = expand_operator(
            qutip.qeye(2), N=systemsize, targets=(0,))
        for _ in range(cycles):
            for ix_site in range(systemsize):
                quutip_operator = expand_operator(
                    rotation_x,
                    N=systemsize,
                    targets=(ix_site,),
                ) * quutip_operator
            for ix_site in range(systemsize):
                quutip_operator = expand_operator(
                    rotation_z[ix_site],
                    N=systemsize,
                    targets=(ix_site,),
                ) * quutip_operator
            for ix_site in range(systemsize - 1):
                quutip_operator = expand_operator(
                    interaction_zz[ix_site],
                    N=systemsize,
                    targets=(ix_site, ix_site + 1),
                ) * quutip_operator

        qiskit_operator = Operator(build_mbldtc_floquet_circuit(
            systemsize,
            theta_x,
            phi_z,
            phi_zz,
            cycles=cycles,
        )).data

        self.assertLess(
            np.linalg.norm(qiskit_operator - quutip_operator.full()),
            1.0e-12,
        )


if __name__ == "__main__":
    unittest.main()
