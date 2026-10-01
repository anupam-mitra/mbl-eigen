"""MBL spectrum, dynamics, and propagator analysis workflows."""

import logging

import numpy as np
import qutip

from . import eigensolver
from . import level_repulsion
from . import output_names
from .eigenphase import (
    build_time_propagator_phases,
    eigenvalues_to_unitary,
    extract_sorted_eigenphases,
)
from .mbl_model import build_mbl_model
from .operators import build_site_operator_array, rotate_to_eigenbasis
from .plotting import (
    plot_eigenvector_entropy,
    plot_return_rate,
)


logger = logging.getLogger(__name__)


def _rng_from_args(args, rng):
    if rng is not None:
        return rng
    seed = getattr(args, "seed", None)
    return None if seed is None else np.random.default_rng(seed)


def _build_model_from_args(args, rng=None):
    return build_mbl_model(
        systemsize=args.systemsize,
        jIntMean=args.jIntMean,
        jIntStd=args.jIntStd,
        bFieldMean=args.bFieldMean,
        bFieldStd=args.bFieldStd,
        anglePolarPiMin=args.anglePolarPiMin,
        anglePolarPiMax=args.anglePolarPiMax,
        rng=rng,
    )


def _mbl_model_title(args):
    return (
        r"$N=%d$, $\theta=[%g, %g]\pi$, $J=%g\pm%g$, $B=%g\pm%g$"
        % (
            args.systemsize,
            args.anglePolarPiMin,
            args.anglePolarPiMax,
            args.jIntMean,
            args.jIntStd,
            args.bFieldMean,
            args.bFieldStd,
        )
    )


def _build_and_diagonalize(args, rng=None):
    model = _build_model_from_args(args, _rng_from_args(args, rng))
    logger.info("bField_samples = %s", model.bField_samples)
    logger.info("theta_samples = %s", model.theta_samples)
    logger.info("jInt_samples = %s", model.jInt_samples)
    diag = eigensolver.solve_hermitian_eigenproblem(
        model.hamiltonian,
        backend=args.eigenBackend,
        device=args.eigenDevice,
    )
    return model, diag


def run_mbl(args, rng=None):
    """Run MBL spectrum and half-chain entanglement entropy analysis."""
    systemsize = args.systemsize
    tduration = _validate_duration(args.tduration)
    model, diag = _build_and_diagonalize(args, rng)
    eigenvalues = diag.eigenvalues
    eigenvectors = diag.as_qobj_kets()

    half_chain = list(range(systemsize // 2))
    eigenvector_entropies = np.array([
        qutip.entropy_vn(qutip.ptrace(v, half_chain))
        for v in eigenvectors
    ])
    logger.info("eigenvector_entropies.shape = %s", eigenvector_entropies.shape)

    energy_ratio = level_repulsion.calc_mean_adjacent_level_spacing_ratio(
        eigenvalues, fraction_cutoff=0.0, use_spacing=True
    )
    eigenvalues_unitary = eigenvalues_to_unitary(eigenvalues, tduration)
    eigenphases = extract_sorted_eigenphases(eigenvalues_unitary)
    phase_ratio = level_repulsion.calc_mean_adjacent_level_spacing_ratio(
        eigenphases,
        fraction_cutoff=0.0,
        use_spacing=True,
        circular_period=2.0 * np.pi,
    )
    logger.info("ratio(energy) = %g", energy_ratio)
    logger.info("ratio(eigenphase) = %g", phase_ratio)

    plot_eigenvector_entropy(
        eigenvalues,
        eigenvector_entropies,
        np.log(2) * (systemsize // 2),
        output_names.mbl_entropy_plot_name(
            systemsize=systemsize,
            anglePolarPiMin=args.anglePolarPiMin,
            anglePolarPiMax=args.anglePolarPiMax,
            jIntMean=args.jIntMean,
            jIntStd=args.jIntStd,
            bFieldMean=args.bFieldMean,
            bFieldStd=args.bFieldStd,
        ),
        title="MBL model: " + _mbl_model_title(args),
    )


def run_mbl_dynamics(args, rng=None):
    """Run MBL return-rate dynamics and save a plot."""
    systemsize = args.systemsize
    model, diag = _build_and_diagonalize(args, rng)
    eigenvectors = diag.as_qobj_kets()
    ket_initial = qutip.ket("1" * systemsize)
    amplitudes = np.asarray([v.overlap(ket_initial) for v in eigenvectors])
    times_array = _time_grid(args.tduration)
    phases = build_time_propagator_phases(diag.eigenvalues, times_array)
    amplitude_return_array = phases @ (np.abs(amplitudes) ** 2)

    logger.info("amplitude_return_array = %s", amplitude_return_array)
    plot_return_rate(
        times_array,
        amplitude_return_array,
        systemsize,
        output_names.mbl_dynamics_plot_name(
            systemsize=systemsize,
            anglePolarPiMin=args.anglePolarPiMin,
            anglePolarPiMax=args.anglePolarPiMax,
            jIntMean=args.jIntMean,
            jIntStd=args.jIntStd,
            bFieldMean=args.bFieldMean,
            bFieldStd=args.bFieldStd,
        ),
        title="MBL model: " + _mbl_model_title(args),
    )


def run_mbl_propagator(args, rng=None):
    """Run MBL propagator and operator-overlap analysis."""
    systemsize = args.systemsize
    model, diag = _build_and_diagonalize(args, rng)
    energies = diag.eigenvalues
    basis_changer = diag.as_basis_qobj()

    operator_arrays = tuple(
        build_site_operator_array(operator, systemsize, normalize=True)
        for operator in (model.sigmax, model.sigmay, model.sigmaz)
    )
    eigenbasis_arrays = tuple(
        rotate_to_eigenbasis(
            [_as_qobj_with_dims(operator, model.hamiltonian.dims)
             for operator in operator_array],
            basis_changer,
        )
        for operator_array in operator_arrays
    )

    times_array = _time_grid(args.tduration)
    phases = build_time_propagator_phases(energies, times_array)
    evolved_arrays = tuple(
        np.empty((systemsize, len(times_array)), dtype=object)
        for _ in range(3)
    )
    for ix_time in range(len(times_array)):
        propagator = basis_changer * qutip.Qobj(
            np.diag(phases[ix_time]), dims=basis_changer.dims
        )
        for ix_site in range(systemsize):
            for eigenbasis_array, evolved_array in zip(
                    eigenbasis_arrays, evolved_arrays):
                evolved_array[ix_site, ix_time] = (
                    propagator
                    * eigenbasis_array[ix_site]
                    * propagator.dag()
                )

    overlap_matrix = _compute_overlap_matrix(
        initial_operator_arrays=operator_arrays,
        time_evolved_operator_arrays=evolved_arrays,
        hamiltonian_dims=model.hamiltonian.dims,
    )
    logger.info(
        "overlap_matrix (t=%s):\n%s",
        times_array[-1],
        np.round(overlap_matrix[:, :, -1], 4),
    )


def _as_qobj_with_dims(operator, dims):
    if isinstance(operator, qutip.Qobj):
        return operator
    return qutip.Qobj(operator, dims=dims)


def _compute_overlap_matrix(
        initial_operator_arrays,
        time_evolved_operator_arrays,
        hamiltonian_dims):
    systemsize = initial_operator_arrays[0].shape[0]
    times_count = time_evolved_operator_arrays[0].shape[1]
    overlap_matrix = np.empty(
        (3 * systemsize, 3 * systemsize, times_count),
        dtype=complex,
    )

    for ix_time in range(times_count):
        for ix_channel_initial in range(3):
            for ix_channel_final in range(3):
                for ix_site_initial in range(systemsize):
                    for ix_site_final in range(systemsize):
                        op_initial = _as_qobj_with_dims(
                            initial_operator_arrays[ix_channel_initial][ix_site_initial],
                            hamiltonian_dims,
                        )
                        op_final = _as_qobj_with_dims(
                            time_evolved_operator_arrays[ix_channel_final][
                                ix_site_final, ix_time
                            ],
                            hamiltonian_dims,
                        )
                        overlap_matrix[
                            ix_channel_initial * systemsize + ix_site_initial,
                            ix_channel_final * systemsize + ix_site_final,
                            ix_time,
                        ] = (op_initial * op_final).tr()
    return overlap_matrix


def _validate_duration(duration, *, allow_zero=False):
    duration = float(duration)
    invalid = (
        not np.isfinite(duration)
        or duration < 0.0
        or (duration == 0.0 and not allow_zero)
    )
    if invalid:
        raise ValueError("tduration must be finite and positive")
    return duration


def _time_grid(duration, step=0.0625):
    duration = _validate_duration(duration, allow_zero=True)
    if duration == 0.0:
        return np.array([0.0])
    sample_count = max(1, int(np.ceil(duration / step)))
    if sample_count > 10_000_000:
        raise ValueError(
            "tduration too large: sample_count exceeds 10_000_000"
        )
    return np.linspace(0.0, duration, sample_count + 1)


time_grid = _time_grid
