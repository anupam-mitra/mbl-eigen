import logging

import matplotlib.pyplot as plt
import numpy as np
import qutip
from qutip.qip.operations import expand_operator

from . import eigensolver
from . import level_repulsion
from . import output_names


"""
Generating the Floquet operator for MBL-DTC,

"""


def run_mbldtc(args):
    systemsize = args.systemsize
    thetaXPi = args.thetaXPi

    sigmax = qutip.sigmax()
    sigmay = qutip.sigmay()
    sigmaz = qutip.sigmaz()

    sigmaz_sigmaz = qutip.tensor(sigmaz, sigmaz)

    theta_x = np.pi * thetaXPi
    phi_z = [np.random.rand() * np.pi for ix_site in range(systemsize)]
    phi_zz = [np.random.rand() * np.pi for ix_site in range(systemsize - 1)]

    ## Generating Floquet operator
    rotation_x = (-1j * theta_x * 0.5 * sigmax).expm()

    rotation_z = [
        (-1j * phi_z[ix_site] * 0.5 * sigmaz).expm()
            for ix_site in range(systemsize)]

    interaction_zz = [
        (-1j * phi_zz[ix_site] * 0.5 * sigmaz_sigmaz).expm()
            for ix_site in range(systemsize - 1)]

    gates_expanded_rotation_x = [
        expand_operator(
            rotation_x, N=systemsize, targets=(ix_site))
        for ix_site in range(systemsize)
    ]

    gates_expanded_rotation_z = [
        expand_operator(
            rotation_z[ix_site], N=systemsize, targets=(ix_site))
        for ix_site in range(systemsize)
    ]

    gates_expanded_interaction_zz = [
        expand_operator(
            interaction_zz[ix_site], N=systemsize, targets=(ix_site, ix_site + 1))
        for ix_site in range(systemsize - 1)
    ]

    u_floquet = expand_operator(
           qutip.qeye(2), N=systemsize, targets=(0,))

    for g in gates_expanded_rotation_x:
       u_floquet = g * u_floquet

    for g in gates_expanded_rotation_z:
        u_floquet = g * u_floquet

    for g in gates_expanded_interaction_zz:
        u_floquet = g * u_floquet

    ## Diagonalizing Floquet operator
    diagonalization = eigensolver.solve_general_eigenproblem(
        u_floquet,
        backend=args.eigenBackend,
        return_eigenvectors=False,
    )
    eigenvalues = diagonalization.eigenvalues

    logging.info("----- Eigenvalues -----")
    logging.info("Eigenvalues = %s" % (eigenvalues,))

    logging.info("----- Eigenphases -----")
    eigenphases = np.sort(
            [np.angle(v) % (2 * np.pi) for v in eigenvalues])

    logging.info("Eigenphases(U) = %s" % (eigenphases,))
    logging.info("Eigenphases(U^2) = %s" %
            (np.sort((eigenphases * 2) % (2 * np.pi)),))

    ratio = level_repulsion.calc_mean_adjacent_level_spacing_ratio(
            (eigenphases) % (2 * np.pi),
            fraction_cutoff=0.0, use_spacing=True)

    logging.info("ratio = %g" % (ratio,))

    ## Plotting
    fig, ax = plt.subplots(1, 1, figsize=(12 / 2.54, 12 / 2.54))

    ax.set_xlabel(r'$\mathrm{Re}(\eta)$')
    ax.set_ylabel(r'$\mathrm{Im}(\eta)$')

    ax.plot(np.real(eigenvalues), np.imag(eigenvalues),
            marker='.', ls='',
            label=r'$\eta$')

    ax.plot(np.real(eigenvalues), -np.imag(eigenvalues),
            marker='o', fillstyle='none', ls='',
            label=r'$\eta^*$')

    ax.legend(loc='center')

    plt.tight_layout()
    plt.savefig(output_names.mbldtc_plot_name(systemsize=systemsize, theta_x=theta_x))
