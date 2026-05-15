"""QMBS / PXP eigenphase analysis workflow."""

import logging

from . import eigensolver
from . import level_repulsion
from . import output_names
from .eigenphase import eigenvalues_to_unitary, extract_sorted_eigenphases
from .plotting import plot_eigenphases_unit_circle
from .qmbs_model import build_pxp_hamiltonian, build_qmbs_ising_hamiltonian


def run_qmbs(args):
    """Run QMBS / PXP eigenphase analysis and save complex-plane plots.

    Parameters
    ----------
    args : argparse.Namespace
        Must provide: ``systemsize``, ``tduration``, ``Delta``,
        ``eigenBackend``, ``eigenDevice``.
    """
    systemsize = args.systemsize
    tduration = args.tduration

    Omega = 1.0
    Delta = args.Delta * Omega
    Vrr = 100.0 * Omega

    # --- Build Hamiltonians ---
    hamiltonian = build_qmbs_ising_hamiltonian(
        systemsize=systemsize, Omega=Omega, Delta=Delta, Vrr=Vrr
    )
    hamiltonian_pxp = build_pxp_hamiltonian(
        systemsize=systemsize, Omega=Omega, Delta=Delta
    )

    # --- Diagonalize ---
    diag = eigensolver.solve_hermitian_eigenproblem(
        hamiltonian,
        backend=args.eigenBackend,
        device=args.eigenDevice,
        return_eigenvectors=False,
    )
    diag_pxp = eigensolver.solve_hermitian_eigenproblem(
        hamiltonian_pxp,
        backend=args.eigenBackend,
        device=args.eigenDevice,
        return_eigenvectors=False,
    )

    eigenvalues = diag.eigenvalues
    eigenvalues_pxp = diag_pxp.eigenvalues

    # --- Eigenphases ---
    eigenvalues_unitary = eigenvalues_to_unitary(eigenvalues, tduration)
    eigenvalues_pxp_unitary = eigenvalues_to_unitary(eigenvalues_pxp, tduration)

    eigenphases = extract_sorted_eigenphases(eigenvalues_unitary)
    eigenphases_pxp = extract_sorted_eigenphases(eigenvalues_pxp_unitary)

    logging.info("----- Eigenvalues -----")
    logging.info(eigenvalues)
    logging.info("----- Eigenphases -----")
    logging.info(
        "Eigenphases(U) = %s",
        [v / 3.141592653589793 for v in eigenphases],
    )
    logging.info(
        "Eigenphases(U_PXP) = %s",
        [v / 3.141592653589793 for v in eigenphases_pxp],
    )

    # --- Level spacing ratios ---
    ratio = level_repulsion.calc_mean_adjacent_level_spacing_ratio(
        eigenphases, fraction_cutoff=0.0, use_spacing=True
    )
    ratio_pxp = level_repulsion.calc_mean_adjacent_level_spacing_ratio(
        eigenphases_pxp, fraction_cutoff=0.0, use_spacing=True
    )
    logging.info("ratio = %g, ratio_pxp = %g", ratio, ratio_pxp)

    ratio = level_repulsion.calc_mean_adjacent_level_spacing_ratio(
        eigenvalues, fraction_cutoff=0.0, use_spacing=True
    )
    ratio_pxp = level_repulsion.calc_mean_adjacent_level_spacing_ratio(
        eigenvalues_pxp, fraction_cutoff=0.0, use_spacing=True
    )
    logging.info("ratio = %g, ratio_pxp = %g", ratio, ratio_pxp)

    # --- Plots ---
    plot_eigenphases_unit_circle(
        eigenvalues_unitary,
        output_names.qmbs_sfim_plot_name(
            systemsize=systemsize,
            tduration=tduration,
            Vrr=Vrr,
            Omega=Omega,
            Delta=Delta,
        ),
    )
    plot_eigenphases_unit_circle(
        eigenvalues_pxp_unitary,
        output_names.qmbs_pxp_plot_name(
            systemsize=systemsize,
            tduration=tduration,
            Omega=Omega,
            Delta=Delta,
        ),
    )
