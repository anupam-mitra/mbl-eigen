"""MBL-DTC Floquet eigenphase analysis workflow."""

import logging

import numpy as np
import qutip

from . import eigensolver
from . import level_repulsion
from . import output_names
from .eigenphase import extract_sorted_eigenphases
from .mbl_app import _rng_from_args
from .operators import spin_operators, zero_operator
from .plotting import plot_eigenphases_unit_circle
from .qiskit_propagators import sample_mbldtc_angles

logger = logging.getLogger(__name__)


def run_mbldtc(args, rng=None):
    """Run MBL-DTC Floquet eigenphase analysis and save a plot."""
    systemsize = args.systemsize
    theta_x = np.pi * args.thetaXPi
    from qutip.qip.operations import expand_operator

    _, sigmax, _, sigmaz = spin_operators()
    sigmaz_sigmaz = qutip.tensor(sigmaz, sigmaz)
    phi_z, phi_zz = sample_mbldtc_angles(
        systemsize, rng=_rng_from_args(args, rng)
    )

    rotation_x = (-1j * theta_x * 0.5 * sigmax).expm()
    rotation_z = [
        (-1j * phi_z[ix_site] * 0.5 * sigmaz).expm()
        for ix_site in range(systemsize)
    ]
    interaction_zz = [
        (-1j * phi_zz[ix_site] * 0.5 * sigmaz_sigmaz).expm()
        for ix_site in range(systemsize - 1)
    ]

    u_floquet = zero_operator(systemsize)
    u_floquet = u_floquet + expand_operator(
        qutip.qeye(2), N=systemsize, targets=(0,)
    )
    for ix_site in range(systemsize):
        u_floquet = expand_operator(
            rotation_x, N=systemsize, targets=(ix_site,)
        ) * u_floquet
    for ix_site, rotation in enumerate(rotation_z):
        u_floquet = expand_operator(
            rotation, N=systemsize, targets=(ix_site,)
        ) * u_floquet
    for ix_site, interaction in enumerate(interaction_zz):
        u_floquet = expand_operator(
            interaction, N=systemsize, targets=(ix_site, ix_site + 1)
        ) * u_floquet

    diag = eigensolver.solve_general_eigenproblem(
        u_floquet,
        backend=args.eigenBackend,
        device=args.eigenDevice,
        return_eigenvectors=False,
    )
    eigenvalues = diag.eigenvalues
    eigenphases = extract_sorted_eigenphases(eigenvalues)
    ratio = level_repulsion.calc_mean_adjacent_level_spacing_ratio(
        eigenphases,
        fraction_cutoff=0.0,
        use_spacing=True,
        circular_period=2.0 * np.pi,
    )

    logger.info("Eigenvalues = %s", eigenvalues)
    logger.info("Eigenphases(U) = %s", eigenphases)
    logger.info("Eigenphases(U^2) = %s", (eigenphases * 2) % (2.0 * np.pi))
    logger.info("ratio = %g", ratio)

    plot_eigenphases_unit_circle(
        eigenvalues,
        output_names.mbldtc_plot_name(systemsize=systemsize, theta_x=theta_x),
    )


__all__ = ["run_mbldtc"]
