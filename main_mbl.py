import numpy as np
import scipy
import scipy.stats
import itertools
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
        scale=(anglePolarPiMax - anglePolarPiMin)* np.pi,)

    ## Generating Hamiltonian 
    bperp_terms = [
         bField_samples[ix_site] * np.sin(theta_samples[ix_site]) * sigmax \
            for ix_site in range(systemsize)]

    bparallel_terms = [
        bField_samples[ix_site] * np.cos(theta_samples[ix_site]) * sigmaz \
            for ix_site in range(systemsize)]

    interaction_zz_terms = [
        jInt_samples[ix_site] * sigmaz_sigmaz \
            for ix_site in range(systemsize-1)]

    hamiltonian = 0.0 * qutip.qip.operations.expand_operator(
                        sigma0, N=systemsize, targets=(0,))

    print("bField_samples = %s" % bField_samples)
    print("theta_samples = %s" % theta_samples)
    print("jInt_Samples = %s" % jInt_samples)
    
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

    eigenvector_entropies = np.array([
            qutip.entropy_vn(qutip.ptrace(v, [ix for ix in range(systemsize >> 1)]))
                 for v in eigenvectors])

    print(eigenvector_entropies.shape)

    ## Eigenvector entropy plot for the Ising Hamiltonian
    fig, ax = plt.subplots(1, 1, figsize=(18.0/2.54, 12.0/2.54))

    ax.set_xlabel(r'Energy')
    ax.set_ylabel(r'$\mathcal{S}_1$')

    ax.plot(eigenvalues, 
            eigenvector_entropies,
            marker='o', markeredgecolor='k', markerfacecolor='blue',
            ls='',
    )

    ax.axhline(np.log(2) * (systemsize >> 1), ls='dotted', color='k')

    plotfilename = "mbl_sfim_N=%02d_anglePolarPiMin=%g_anglePolarPiMax=%g" \
        % (systemsize, anglePolarPiMin, anglePolarPiMax) + \
        "_jIntMean=%g_jIntStd=%g_bFieldMean=%g_bFieldStd=%g_%s.pdf" \
        % (jIntMean, jIntStd, bFieldMean, bFieldStd, uuid.uuid4())

    fig.savefig(plotfilename)
    plt.close()
