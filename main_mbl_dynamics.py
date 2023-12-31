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
    print("jInt_samples = %s" % jInt_samples)
    
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

    ## Time evolution
    ket_initial = qutip.ket('1' * systemsize)
    print(ket_initial)

    amplitude_eigenvectors = [(v.dag() * ket_initial).tr() for v in eigenvectors]
    print(amplitude_eigenvectors)

    times_array = np.arange(0.0, 10.0625, 0.0625)

    amplitude_return_array = np.empty_like(times_array, dtype=complex)

    for ix_time, t in enumerate(times_array):
        a = np.sum([
            np.exp(-1j * t * eigenvalues[k]) * \
             np.abs(amplitude_eigenvectors[k])**2
            for k in range(len(eigenvalues))])

        amplitude_return_array[ix_time] = a

    print(amplitude_return_array)
  
    ## Eigenvector entropy plot for the Ising Hamiltonian
    fig, ax = plt.subplots(1, 1, figsize=(18.0/2.54, 12.0/2.54))

    ax.plot(times_array, -np.log(np.abs(amplitude_return_array)**2) / systemsize)
    ax.set_ylabel(r"$\lambda(t)$")
    ax.set_xlabel(r"$B_{\mathrm{mean}} t$")

    plotfilename = "mbl_sfim_dynamics_N=%02d_anglePolarPiMin=%g_anglePolarPiMax=%g" \
        % (systemsize, anglePolarPiMin, anglePolarPiMax) + \
        "_jIntMean=%g_jIntStd=%g_bFieldMean=%g_bFieldStd=%g_%s.pdf" \
        % (jIntMean, jIntStd, bFieldMean, bFieldStd, uuid.uuid4())

    fig.savefig(plotfilename)
    plt.close()

