"""MBL-DTC Floquet eigenphase analysis workflow."""

import logging

import numpy as np
import qutip
from qutip.qip.operations import expand_operator

from . import eigensolver
from . import level_repulsion
from . import output_names
from .operators import spin_operators, zero_operator
from .plotting import plot_eigenphases_unit_circle


def run_mbldtc(args):
    """Run the MBL-DTC Floquet eigenphase analysis and save a plot.

    Parameters
    ----------
    args : argparse.Namespace
        Must provide: ``systemsize``, ``thetaXPi``, ``eigenBackend``,
        ``eigenDevice``.
    """
    systemsize = args.systemsize
    thetaXPi = args.thetaXPi

    _, sigmax, _, sigmaz = spin_operators()
    sigmaz_sigmaz = qutip.tensor(sigmaz, sigmaz)

    theta_x = np.pi * thetaXPi
    phi_z = [np.random.rand() * np.pi for _ in range(systemsize)]
    phi_zz = [np.random.rand() * np.pi for _ in range(systemsize - 1)]

    # --- Build Floquet operator ---
    rotation_x = (-1j * theta_x * 0.5 * sigmax).expm()
    rotation_z = [(-1j * phi_z[i] * 0.5 * sigmaz).expm() for i in range(systemsize)]
    interaction_zz = [
        (-1j * phi_zz[i] * 0.5 * sigmaz_sigmaz).expm() for i in range(systemsize - 1)
    ]

    u_floquet = zero_operator(systemsize)
    # Identity as starting point: add 1 back
    u_floquet = u_floquet + expand_operator(qutip.qeye(2), N=systemsize, targets=(0,))

    for g in (expand_operator(rotation_x, N=systemsize, targets=(i,)) for i in range(systemsize)):
        u_floquet = g * u_floquet

    for i, rz in enumerate(rotation_z):
        g = expand_operator(rz, N=systemsize, targets=(i,))
        u_floquet = g * u_floquet

    for i, zz in enumerate(interaction_zz):
        g = expand_operator(zz, N=systemsize, targets=(i, i + 1))
        u_floquet = g * u_floquet

    # --- Diagonalize ---
    diag = eigensolver.solve_general_eigenproblem(
        u_floquet,
        backend=args.eigenBackend,
        device=args.eigenDevice,
        return_eigenvectors=False,
    )
    eigenvalues = diag.eigenvalues

    # --- Eigenphases ---
    eigenphases = np.sort([np.angle(v) % (2 * np.pi) for v in eigenvalues])

    logging.info("----- Eigenvalues -----")
    logging.info("Eigenvalues = %s", eigenvalues)
    logging.info("----- Eigenphases -----")
    logging.info("Eigenphases(U) = %s", eigenphases)
    logging.info(
        "Eigenphases(U^2) = %s", np.sort((eigenphases * 2) % (2 * np.pi))
    )

    ratio = level_repulsion.calc_mean_adjacent_level_spacing_ratio(
        eigenphases % (2 * np.pi), fraction_cutoff=0.0, use_spacing=True
    )
    logging.info("ratio = %g", ratio)

    # --- Plot ---
    plot_eigenphases_unit_circle(
        eigenvalues,
        output_names.mbldtc_plot_name(systemsize=systemsize, theta_x=theta_x),
    )
