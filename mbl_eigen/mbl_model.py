from dataclasses import dataclass

import numpy as np
import qutip
import scipy.stats
from qutip.qip.operations import expand_operator


@dataclass
class MBLModel:
    hamiltonian: object
    sigma0: object
    sigmax: object
    sigmay: object
    sigmaz: object
    jInt_samples: np.ndarray
    bField_samples: np.ndarray
    theta_samples: np.ndarray


def spin_operators():
    sigma0 = qutip.qeye(2)
    sigmax = qutip.sigmax()
    sigmay = qutip.sigmay()
    sigmaz = qutip.sigmaz()
    return sigma0, sigmax, sigmay, sigmaz


def sample_mbl_disorder(
        systemsize,
        jIntMean,
        jIntStd,
        bFieldMean,
        bFieldStd,
        anglePolarPiMin,
        anglePolarPiMax):
    jInt_samples = scipy.stats.norm.rvs(
        size=(systemsize - 1), loc=jIntMean, scale=jIntStd)

    bField_samples = scipy.stats.norm.rvs(
        size=systemsize, loc=bFieldMean, scale=bFieldStd)

    theta_samples = scipy.stats.uniform.rvs(
        size=systemsize,
        loc=anglePolarPiMin * np.pi,
        scale=(anglePolarPiMax - anglePolarPiMin) * np.pi,)

    return jInt_samples, bField_samples, theta_samples


def build_mbl_hamiltonian(
        systemsize,
        jInt_samples,
        bField_samples,
        theta_samples,
        sigma0,
        sigmax,
        sigmaz):
    sigmaz_sigmaz = qutip.tensor(sigmaz, sigmaz)

    bperp_terms = [
         bField_samples[ix_site] * np.sin(theta_samples[ix_site]) * sigmax
            for ix_site in range(systemsize)]

    bparallel_terms = [
        bField_samples[ix_site] * np.cos(theta_samples[ix_site]) * sigmaz
            for ix_site in range(systemsize)]

    interaction_zz_terms = [
        jInt_samples[ix_site] * sigmaz_sigmaz
            for ix_site in range(systemsize - 1)]

    hamiltonian = 0.0 * expand_operator(
                        sigma0, N=systemsize, targets=(0,))

    for ix_site in range(systemsize):
        h = bperp_terms[ix_site]
        hamiltonian = hamiltonian + \
                expand_operator(
                        h, N=systemsize, targets=(ix_site,))

        h = bparallel_terms[ix_site]
        hamiltonian = hamiltonian + \
                expand_operator(
                        h, N=systemsize, targets=(ix_site,))

    for ix_site in range(systemsize - 1):
        h = interaction_zz_terms[ix_site]
        hamiltonian = hamiltonian + \
                expand_operator(
                        h, N=systemsize, targets=(ix_site, ix_site + 1))

    return hamiltonian


def build_mbl_model(
        systemsize,
        jIntMean,
        jIntStd,
        bFieldMean,
        bFieldStd,
        anglePolarPiMin,
        anglePolarPiMax):
    sigma0, sigmax, sigmay, sigmaz = spin_operators()
    jInt_samples, bField_samples, theta_samples = sample_mbl_disorder(
        systemsize=systemsize,
        jIntMean=jIntMean,
        jIntStd=jIntStd,
        bFieldMean=bFieldMean,
        bFieldStd=bFieldStd,
        anglePolarPiMin=anglePolarPiMin,
        anglePolarPiMax=anglePolarPiMax,
    )

    hamiltonian = build_mbl_hamiltonian(
        systemsize=systemsize,
        jInt_samples=jInt_samples,
        bField_samples=bField_samples,
        theta_samples=theta_samples,
        sigma0=sigma0,
        sigmax=sigmax,
        sigmaz=sigmaz,
    )

    return MBLModel(
        hamiltonian=hamiltonian,
        sigma0=sigma0,
        sigmax=sigmax,
        sigmay=sigmay,
        sigmaz=sigmaz,
        jInt_samples=jInt_samples,
        bField_samples=bField_samples,
        theta_samples=theta_samples,
    )
