import numpy as np
import qutip
import uuid
import argparse
import matplotlib.pyplot as plt

"""
Generating the PXP Hamiltonian from Rydberg blockade
"""
if __name__ == '__main__':

    argument_parser = argparse.ArgumentParser(
        prog="main_qmbs.py",
        description="Plots eigenphases of the time evolution operator under a PXP like Hamiltonian",
        epilog=""
    )

    argument_parser.add_argument("--systemsize", type=int)
    argument_parser.add_argument("--tduration", type=float)
    argument_parser.add_argument("--Delta", type=float)

    args = argument_parser.parse_args()

    systemsize = args.systemsize
    tduration = args.tduration
    Delta = args.Delta

    sigma0 = qutip.qeye(2)
    sigmax = qutip.sigmax()
    sigmay = qutip.sigmay()
    sigmaz = qutip.sigmaz()
    
    projector_g = (sigma0 + sigmaz) * 0.5
    projector_r = (sigma0 - sigmaz) * 0.5

    projector_rr = qutip.tensor(projector_r, projector_r)

    Omega = 1.0
    Delta = Delta * Omega
    Vrr = 100.0 * Omega

    ## Generating Hamiltonian 
    drive_x_terms = [
        Omega * 0.5 * sigmax \
            for ix_site in range(systemsize)]

    detuning_z_terms = [
        Delta * 0.5 * sigmaz \
            for ix_site in range(systemsize)]

    interaction_zz_terms = [
       Vrr * projector_rr \
            for ix_site in range(systemsize-1)]

    pxp_terms = [ 
        Omega * qutip.tensor(sigmax, projector_g)
    ] + \
    [
        Omega * qutip.tensor(projector_g, sigmax, projector_g)
            for ix_site in range(systemsize - 2)
    ] + \
    [
        Omega * qutip.tensor(projector_g, sigmax)
    ]

    hamiltonian = 0.0 * qutip.qip.operations.expand_operator(\
           qutip.qeye(2), N = systemsize, targets=(0,))

    for ix_site in range(systemsize):
        h = drive_x_terms[ix_site]
        hamiltonian = hamiltonian + \
                qutip.qip.operations.expand_operator(
                        h, N=systemsize, targets=(ix_site,))

        h = detuning_z_terms[ix_site]
        hamiltonian = hamiltonian + \
                qutip.qip.operations.expand_operator(
                        h, N=systemsize, targets=(ix_site,))

    for ix_site in range(systemsize-1):
        h = interaction_zz_terms[ix_site]
        hamiltonian = hamiltonian + \
                qutip.qip.operations.expand_operator(
                        h, N=systemsize, targets=(ix_site, ix_site+1))

    hamiltonian_pxp = 0.0 * qutip.qip.operations.expand_operator(\
           qutip.qeye(2), N = systemsize, targets=(0,))

    for ix_site in range(systemsize):
        h = pxp_terms[ix_site]

        if ix_site == 0:
            h_expanded = qutip.qip.operations.expand_operator(
                        h, N=systemsize, targets=(ix_site, ix_site+1))
        elif ix_site == systemsize - 1:
            h_expanded = qutip.qip.operations.expand_operator(
                        h, N=systemsize, targets=(ix_site-1, ix_site))
        else:
            h_expanded = qutip.qip.operations.expand_operator(
                        h, N=systemsize, targets=(ix_site-1, ix_site, ix_site+1))

        hamiltonian_pxp = hamiltonian_pxp + h_expanded
         
    for ix_site in range(systemsize):
        h = detuning_z_terms[ix_site]
        hamiltonian_pxp = hamiltonian_pxp + \
                qutip.qip.operations.expand_operator(
                        h, N=systemsize, targets=(ix_site,))

    ## Diagonalizing Hamiltonian

    eigenvalues, eigenvectors = hamiltonian.eigenstates()

    eigenvalues_pxp, eigenvectors_pxp = hamiltonian_pxp.eigenstates()

    eigenvalues_unitary = np.exp(-1j * eigenvalues * tduration)

    eigenvalues_pxp_unitary = np.exp(-1j * eigenvalues_pxp * tduration)

    print("----- Eigenvalues -----")
    print(eigenvalues)

    print("----- Eigenphases -----")
    print([np.angle(v) / np.pi for v in eigenvalues_unitary])

    ## Eigenphases plot for the Ising Hamiltonian
    fig, ax = plt.subplots(1, 1, figsize=(12/2.54, 12/2.54))

    ax.set_xlabel(r'$\mathrm{Re}(\eta)$')
    ax.set_ylabel(r'$\mathrm{Im}(\eta)$')

    ax.plot(np.real(eigenvalues_unitary), np.imag(eigenvalues_unitary),
            marker='.', ls='',
            label=r'$\eta$')

    ax.plot(np.real(eigenvalues_unitary), -np.imag(eigenvalues_unitary),
            marker='o', fillstyle='none', ls='',
            label=r'$\eta^*$')

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)

    ax.legend(loc='center')

    plt.tight_layout()
    plt.savefig("qmbs_sfim_N=%02d_tduration=%g_Vrr=%g_Omega=%g_Delta=%g.pdf" \
            % (systemsize, tduration, Vrr, Omega, Delta))
    plt.close()

    ## Eigenphases plot for the PXP Hamiltonian
    fig, ax = plt.subplots(1, 1, figsize=(12/2.54, 12/2.54))

    ax.set_xlabel(r'$\mathrm{Re}(\eta)$')
    ax.set_ylabel(r'$\mathrm{Im}(\eta)$')

    ax.plot(np.real(eigenvalues_pxp_unitary), np.imag(eigenvalues_pxp_unitary),
            marker='.', ls='',
            label=r'$\eta$')

    ax.plot(np.real(eigenvalues_pxp_unitary), -np.imag(eigenvalues_pxp_unitary),
            marker='o', fillstyle='none', ls='',
            label=r'$\eta^*$')

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)

    ax.legend(loc='center')

    plt.tight_layout()
    plt.savefig("qmbs_pxp_N=%02d_tduration=%g_Omega=%g_Delta=%g.pdf" \
            % (systemsize, tduration, Omega, Delta))
    plt.close()

