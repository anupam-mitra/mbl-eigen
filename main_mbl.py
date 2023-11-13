import numpy as np
import scipy
import scipy.stats
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
    argument_parser.add_argument("--jIntMean", type=float)
    argument_parser.add_argument("--bFieldMean", type=float)
    argument_parser.add_argument("--jIntStd", type=float)
    argument_parser.add_argument("--bFieldStd", type=float)
    argument_parser.add_argument("--anglePolarPiMin", type=float)
    argument_parser.add_argument("--anglePolarPiMax", type=float)

    args = argument_parser.parse_args()

    systemsize = args.systemsize
    tduration = args.tduration
    jIntMean = args.jIntMean
    jIntStd = args.jIntStd
    bFieldMean = args.bFieldMean
    bFieldStd = args.bFieldStd
    anglePolarPiMin = args.anglePolarPiMin
    anglePolarPiMax = args.anglePolarPiMax

    sigma0 = qutip.qeye(2)
    sigmax = qutip.sigmax()
    sigmay = qutip.sigmay()
    sigmaz = qutip.sigmaz()

    sigmaz_sigmaz = qutip.tensor(sigmaz, sigmaz)
    
    projector_g = (sigma0 + sigmaz) * 0.5
    projector_r = (sigma0 - sigmaz) * 0.5

    projector_rr = qutip.tensor(projector_r, projector_r)

    jInt_samples = scipy.stats.norm.rvs(
        size=(systemsize-1), loc=jIntMean, scale=jIntStd)
    
    bField_samples = scipy.stats.norm.rvs(
        size=systemsize, loc=bFieldMean, scale=bFieldStd)

    theta_samples = scipy.stats.uniform.rvs(
        size=systemsize,
        loc=anglePolarPiMin * np.pi,
        scale=anglePolarPiMax * np.pi,)

    ## Generating Hamiltonian 
    bperp_terms = [
         bField_samples[ix_site] * np.cos(theta_samples[ix_site]) * sigmax \
            for ix_site in range(systemsize)]

    bparallel_terms = [
        bField_samples[ix_site] * np.sin(theta_samples[ix_site]) * sigmaz \
            for ix_site in range(systemsize)]

    interaction_zz_terms = [
        jInt_samples[ix_site] * sigmaz_sigmaz \
            for ix_site in range(systemsize-1)]

    hamiltonian = 0.0 * qutip.qip.operations.expand_operator(
                        sigma0, N=systemsize, targets=(0,))
    
    for ix_site in range(systemsize):
        h = bperp_terms[ix_site]
        hamiltonian = hamiltonian + \
                qutip.qip.operations.expand_operator(
                        h, N=systemsize, targets=(ix_site,))

        h = bparallel_terms[ix_site]
        hamiltonian = hamiltonian + \
                qutip.qip.operations.expand_operator(
                        h, N=systemsize, targets=(ix_site,))

    for ix_site in range(systemsize-1):
        h = interaction_zz_terms[ix_site]
        hamiltonian = hamiltonian + \
                qutip.qip.operations.expand_operator(
                        h, N=systemsize, targets=(ix_site, ix_site+1))

    ## Diagonalizing Hamiltonian

    eigenvalues, eigenvectors = hamiltonian.eigenstates()

    eigenvalues_unitary = np.exp(-1j * eigenvalues * tduration)

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
    plt.savefig(
        "mbl_sfim_N=%02d" % (systemsize,) + \
        "_tduration=%g_jIntMean=%g_jIntStd=%g_bFieldMean=%g_bFieldStd=%g_%s.pdf" \
        % (tduration, jIntMean, jIntStd, bFieldMean, bFieldStd, uuid.uuid4()))
    plt.close()
