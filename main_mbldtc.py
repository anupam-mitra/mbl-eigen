import numpy as np
import qutip
import uuid
import argparse
import matplotlib.pyplot as plt

"""
Generating the Floquet operator for MBL-DTC,

"""
if __name__ == '__main__':

    argument_parser = argparse.ArgumentParser(
        prog="main_qmbs.py",
        description="Plots eigenphases of the time evolution operator under a PXP like Hamiltonian",
        epilog=""
    )

    argument_parser.add_argument("--systemsize", type=int)
    argument_parser.add_argument("--thetaXPi", type=float)

    args = argument_parser.parse_args()

    systemsize = args.systemsize
    thetaXPi = args.thetaXPi

    sigmax = qutip.sigmax()
    sigmay = qutip.sigmay()
    sigmaz = qutip.sigmaz()

    sigmaz_sigmaz = qutip.tensor(sigmaz, sigmaz)

    theta_x = np.pi * thetaXPi
    phi_z = [np.random.rand() * np.pi for ix_site in range(systemsize)] 
    phi_zz = [np.random.rand() * np.pi for ix_site in range(systemsize-1)] 

    ## Generating Floquet operator
    rotation_x = (-1j * theta_x * 0.5 * sigmax).expm()

    rotation_z = [
        (-1j * phi_z[ix_site] * 0.5 * sigmaz).expm() \
            for ix_site in range(systemsize)]

    interaction_zz = [
        (-1j * phi_zz[ix_site] * 0.5 * sigmaz_sigmaz).expm()
            for ix_site in range(systemsize-1)]

    gates_expanded_rotation_x = [
        qutip.qip.operations.expand_operator(
            rotation_x, N=systemsize, targets=(ix_site))
        for ix_site in range(systemsize)
    ]

    gates_expanded_rotation_z = [
        qutip.qip.operations.expand_operator(
            rotation_z[ix_site], N=systemsize, targets=(ix_site))
        for ix_site in range(systemsize)
    ]

    gates_expanded_interaction_zz = [
        qutip.qip.operations.expand_operator(
            interaction_zz[ix_site], N=systemsize, targets=(ix_site, ix_site+1))
        for ix_site in range(systemsize-1)
    ]

    u_floquet = qutip.qip.operations.expand_operator(\
           qutip.qeye(2), N = systemsize, targets=(0,))

    for g in gates_expanded_rotation_x:
       u_floquet = g * u_floquet

    for g in gates_expanded_rotation_z:
        u_floquet = g * u_floquet

    for g in gates_expanded_interaction_zz:
        u_floquet = g * u_floquet

    ## Diagonalizing Floquet operator
    eigenvalues, eigenvectors = u_floquet.eigenstates()

    print("----- Eigenvalues -----")
    print("Eigenvalues = %s" % (eigenvalues,))

    print("----- Eigenphases -----")
    print("Eigenphases(U) = %s" % ([np.angle(v) / np.pi for v in eigenvalues],))
    print("Eigenphases(U^2) = %s" % (np.sort([np.angle(v**2) / np.pi for v in eigenvalues]),)) 

    fig, ax = plt.subplots(1, 1, figsize=(12/2.54, 12/2.54))

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
    plt.savefig("mbldtc_N=%02d_thetax=%gpi_%s.pdf" % (systemsize, theta_x/np.pi, uuid.uuid4()))
