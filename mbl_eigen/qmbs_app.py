"""QMBS / PXP eigenphase analysis workflow."""

import logging

import numpy as np

from . import eigensolver
from . import level_repulsion
from . import output_names
from .eigenphase import eigenvalues_to_unitary, extract_sorted_eigenphases
from .plotting import plot_eigenphases_unit_circle
from .qmbs_model import build_pxp_hamiltonian, build_qmbs_ising_hamiltonian


logger = logging.getLogger(__name__)

_OMEGA = 1.0
_VRR = 100.0 * _OMEGA


def run_qmbs(args, rng=None):
    """Run QMBS / PXP eigenphase analysis and save both plots.

    ``rng`` is accepted for API uniformity with sibling workflows; unused.
    """
    systemsize = args.systemsize
    tduration = float(args.tduration)
    if not np.isfinite(tduration) or tduration <= 0.0:
        raise ValueError("tduration must be finite and positive")
    delta = args.Delta * _OMEGA

    hamiltonian = build_qmbs_ising_hamiltonian(
        systemsize=systemsize,
        Omega=_OMEGA,
        Delta=delta,
        Vrr=_VRR,
    )
    hamiltonian_pxp = build_pxp_hamiltonian(
        systemsize=systemsize,
        Omega=_OMEGA,
        Delta=delta,
    )

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
    eigenvalues_unitary = eigenvalues_to_unitary(eigenvalues, tduration)
    eigenvalues_pxp_unitary = eigenvalues_to_unitary(
        eigenvalues_pxp, tduration
    )
    eigenphases = extract_sorted_eigenphases(eigenvalues_unitary)
    eigenphases_pxp = extract_sorted_eigenphases(eigenvalues_pxp_unitary)

    energy_ratio = level_repulsion.calc_mean_adjacent_level_spacing_ratio(
        eigenvalues, fraction_cutoff=0.0, use_spacing=True
    )
    energy_ratio_pxp = level_repulsion.calc_mean_adjacent_level_spacing_ratio(
        eigenvalues_pxp, fraction_cutoff=0.0, use_spacing=True
    )
    phase_ratio = level_repulsion.calc_mean_adjacent_level_spacing_ratio(
        eigenphases,
        fraction_cutoff=0.0,
        use_spacing=True,
        circular_period=2.0 * np.pi,
    )
    phase_ratio_pxp = level_repulsion.calc_mean_adjacent_level_spacing_ratio(
        eigenphases_pxp,
        fraction_cutoff=0.0,
        use_spacing=True,
        circular_period=2.0 * np.pi,
    )
    logger.info("ratio(energy) = %g, ratio_pxp(energy) = %g",
                energy_ratio, energy_ratio_pxp)
    logger.info("ratio(eigenphase) = %g, ratio_pxp(eigenphase) = %g",
                phase_ratio, phase_ratio_pxp)

    plot_eigenphases_unit_circle(
        eigenvalues_unitary,
        output_names.qmbs_sfim_plot_name(
            systemsize=systemsize,
            tduration=tduration,
            Vrr=_VRR,
            Omega=_OMEGA,
            Delta=delta,
        ),
    )
    plot_eigenphases_unit_circle(
        eigenvalues_pxp_unitary,
        output_names.qmbs_pxp_plot_name(
            systemsize=systemsize,
            tduration=tduration,
            Omega=_OMEGA,
            Delta=delta,
        ),
    )


__all__ = ["run_qmbs"]
