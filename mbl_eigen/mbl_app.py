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
    plot_eigenphases_unit_circle,
    plot_return_rate,
)


def _build_model_from_args(args):
    return build_mbl_model(
        systemsize=args.systemsize,
        jIntMean=args.jIntMean,
        jIntStd=args.jIntStd,
        bFieldMean=args.bFieldMean,
        bFieldStd=args.bFieldStd,
        anglePolarPiMin=args.anglePolarPiMin,
        anglePolarPiMax=args.anglePolarPiMax,
    )


def run_mbl(args):
    """Run MBL spectrum and half-chain entanglement entropy analysis.

    Parameters
    ----------
    args : argparse.Namespace
        Must provide: ``systemsize``, ``tduration``, ``jIntMean``,
        ``jIntStd``, ``bFieldMean``, ``bFieldStd``, ``anglePolarPiMin``,
        ``anglePolarPiMax``, ``eigenBackend``, ``eigenDevice``.
    """
    systemsize = args.systemsize
    tduration = args.tduration

    model = _build_model_from_args(args)
    logging.info("bField_samples = %s", model.bField_samples)
    logging.info("theta_samples = %s", model.theta_samples)
    logging.info("jInt_samples = %s", model.jInt_samples)

    # --- Diagonalize ---
    diag = eigensolver.solve_hermitian_eigenproblem(
        model.hamiltonian,
        backend=args.eigenBackend,
        device=args.eigenDevice,
    )
    eigenvalues = diag.eigenvalues
    eigenvectors = diag.as_qobj_kets()

    # --- Half-chain entanglement entropy ---
    half_chain = list(range(systemsize >> 1))
    eigenvector_entropies = np.array(
        [qutip.entropy_vn(qutip.ptrace(v, half_chain)) for v in eigenvectors]
    )
    logging.info("eigenvector_entropies.shape = %s", eigenvector_entropies.shape)

    ratio = level_repulsion.calc_mean_adjacent_level_spacing_ratio(
        eigenvalues, fraction_cutoff=0.0, use_spacing=True
    )
    logging.info("ratio(energy) = %g", ratio)

    # --- Eigenphases ---
    eigenvalues_unitary = eigenvalues_to_unitary(eigenvalues, tduration)
    eigenphases = extract_sorted_eigenphases(eigenvalues_unitary)

    ratio = level_repulsion.calc_mean_adjacent_level_spacing_ratio(
        eigenphases, fraction_cutoff=0.0, use_spacing=True
    )
    logging.info("ratio(eigenphase) = %g", ratio)

    # --- Plot ---
    page_entropy = np.log(2) * (systemsize >> 1)
    plot_eigenvector_entropy(
        eigenvalues,
        eigenvector_entropies,
        page_entropy,
        output_names.mbl_entropy_plot_name(
            systemsize=systemsize,
            anglePolarPiMin=args.anglePolarPiMin,
            anglePolarPiMax=args.anglePolarPiMax,
            jIntMean=args.jIntMean,
            jIntStd=args.jIntStd,
            bFieldMean=args.bFieldMean,
            bFieldStd=args.bFieldStd,
        ),
    )


def run_mbl_dynamics(args):
    """Run MBL return-rate dynamics and save a plot.

    Parameters
    ----------
    args : argparse.Namespace
        Same attributes as :func:`run_mbl`; ``tduration`` is accepted but
        not used (the time grid is hard-coded).
    """
    systemsize = args.systemsize

    model = _build_model_from_args(args)
    logging.info("bField_samples = %s", model.bField_samples)
    logging.info("theta_samples = %s", model.theta_samples)
    logging.info("jInt_samples = %s", model.jInt_samples)

    # --- Diagonalize ---
    diag = eigensolver.solve_hermitian_eigenproblem(
        model.hamiltonian,
        backend=args.eigenBackend,
        device=args.eigenDevice,
    )
    eigenvalues = diag.eigenvalues
    eigenvectors = diag.as_qobj_kets()

    # --- Return-rate dynamics ---
    ket_initial = qutip.ket("1" * systemsize)
    logging.info("ket_initial = %s", ket_initial)

    amplitude_eigenvectors = [v.overlap(ket_initial) for v in eigenvectors]
    logging.info("amplitude_eigenvectors = %s", amplitude_eigenvectors)

    times_array = np.arange(0.0, 10.0625, 0.0625)
    phases = build_time_propagator_phases(eigenvalues, times_array)  # (T, D)
    weights = np.abs(amplitude_eigenvectors) ** 2
    amplitude_return_array = phases @ weights  # (T,)

    logging.info("amplitude_return_array = %s", amplitude_return_array)

    # --- Plot ---
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
    )


def run_mbl_propagator(args):
    """Run MBL propagator and operator-overlap analysis (logs only, no plot).

    Parameters
    ----------
    args : argparse.Namespace
        Same attributes as :func:`run_mbl`; ``tduration`` is accepted but
        not used (the time grid is hard-coded).
    """
    systemsize = args.systemsize

    model = _build_model_from_args(args)
    logging.info("bField_samples = %s", model.bField_samples)
    logging.info("theta_samples = %s", model.theta_samples)
    logging.info("jInt_samples = %s", model.jInt_samples)

    hamiltonian = model.hamiltonian

    # --- Diagonalize ---
    diag = eigensolver.solve_hermitian_eigenproblem(
        hamiltonian,
        backend=args.eigenBackend,
        device=args.eigenDevice,
    )
    energies = diag.eigenvalues
    basis_changer = diag.as_basis_qobj()
    logging.info("basis_changer = %s", basis_changer)

    # --- Build normalised site-operator arrays ---
    sigmax_array = build_site_operator_array(model.sigmax, systemsize, normalize=True)
    sigmay_array = build_site_operator_array(model.sigmay, systemsize, normalize=True)
    sigmaz_array = build_site_operator_array(model.sigmaz, systemsize, normalize=True)

    logging.info("sigmax:\n%s", sigmax_array)
    logging.info("sigmay:\n%s", sigmay_array)
    logging.info("sigmaz:\n%s", sigmaz_array)

    # --- Rotate into Hamiltonian eigenbasis ---
    # Wrap each Qobj element so it carries the correct dims for matrix mul.
    def _as_qobj_with_dims(arr):
        return [qutip.Qobj(op, dims=hamiltonian.dims) for op in arr]

    sigmax_eb = rotate_to_eigenbasis(_as_qobj_with_dims(sigmax_array), basis_changer)
    sigmay_eb = rotate_to_eigenbasis(_as_qobj_with_dims(sigmay_array), basis_changer)
    sigmaz_eb = rotate_to_eigenbasis(_as_qobj_with_dims(sigmaz_array), basis_changer)

    logging.info("sigmax (eigenbasis):\n%s", sigmax_eb)
    logging.info("sigmay (eigenbasis):\n%s", sigmay_eb)
    logging.info("sigmaz (eigenbasis):\n%s", sigmaz_eb)

    # --- Time evolution ---
    times_array = np.arange(0.0, 1.0625, 0.0625)
    phases = build_time_propagator_phases(energies, times_array)  # (T, D)

    sigmax_evolved = np.empty((systemsize, len(times_array)), dtype=object)
    sigmay_evolved = np.empty((systemsize, len(times_array)), dtype=object)
    sigmaz_evolved = np.empty((systemsize, len(times_array)), dtype=object)

    for ix_t, t in enumerate(times_array):
        propagator = basis_changer * qutip.Qobj(
            np.diag(phases[ix_t]), dims=basis_changer.dims
        )
        for ix_site in range(systemsize):
            sigmax_evolved[ix_site, ix_t] = propagator * sigmax_eb[ix_site] * propagator.dag()
            sigmay_evolved[ix_site, ix_t] = propagator * sigmay_eb[ix_site] * propagator.dag()
            sigmaz_evolved[ix_site, ix_t] = propagator * sigmaz_eb[ix_site] * propagator.dag()

    logging.info("sigmax evolved (t=-1):\n%s", sigmax_evolved[0, -1])
    logging.info("sigmay evolved (t=-1):\n%s", sigmay_evolved[0, -1])
    logging.info("sigmaz evolved (t=-1):\n%s", sigmaz_evolved[0, -1])

    # --- Operator overlap matrix ---
    overlap_matrix = np.empty(
        (3 * systemsize, 3 * systemsize, len(times_array)), dtype=complex
    )

    channels = [
        (sigmax_array, sigmax_evolved, 0),
        (sigmay_array, sigmay_evolved, systemsize),
        (sigmaz_array, sigmaz_evolved, 2 * systemsize),
    ]

    for ix_t in range(len(times_array)):
        for op_initial_arr, op_evolved_arr, offset in channels:
            for ix_i in range(systemsize):
                for ix_f in range(systemsize):
                    op_i = qutip.Qobj(op_initial_arr[ix_i], dims=hamiltonian.dims)
                    op_f = qutip.Qobj(op_evolved_arr[ix_f, ix_t], dims=hamiltonian.dims)
                    overlap_matrix[offset + ix_i, offset + ix_f, ix_t] = (op_i * op_f).tr()

    logging.info(
        "overlap_matrix (t=-1):\n%s", np.round(overlap_matrix[:, :, -1], 4)
    )
