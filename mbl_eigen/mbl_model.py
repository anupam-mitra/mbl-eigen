"""Many-body localized (MBL) random-field spin-chain model builder."""

from dataclasses import dataclass

import numpy as np
import scipy.stats

from .operators import (
    spin_operators,
    sum_single_site_terms,
    sum_two_site_terms,
)


@dataclass
class MBLModel:
    """Container for one random-field MBL realization."""

    hamiltonian: object
    sigma0: object
    sigmax: object
    sigmay: object
    sigmaz: object
    jInt_samples: np.ndarray
    bField_samples: np.ndarray
    theta_samples: np.ndarray


def sample_mbl_disorder(
        systemsize,
        jIntMean,
        jIntStd,
        bFieldMean,
        bFieldStd,
        anglePolarPiMin,
        anglePolarPiMax,
        rng=None):
    """Draw one disorder realization for the MBL Hamiltonian."""
    jInt_samples = scipy.stats.norm.rvs(
        size=systemsize - 1,
        loc=jIntMean,
        scale=jIntStd,
        random_state=rng,
    )
    bField_samples = scipy.stats.norm.rvs(
        size=systemsize,
        loc=bFieldMean,
        scale=bFieldStd,
        random_state=rng,
    )
    theta_samples = scipy.stats.uniform.rvs(
        size=systemsize,
        loc=anglePolarPiMin * np.pi,
        scale=(anglePolarPiMax - anglePolarPiMin) * np.pi,
        random_state=rng,
    )
    return jInt_samples, bField_samples, theta_samples


def build_mbl_hamiltonian(
        systemsize,
        jInt_samples,
        bField_samples,
        theta_samples,
        sigma0,
        sigmax,
        sigmaz):
    """Assemble the random-field MBL Hamiltonian."""
    import qutip

    sigmaz_sigmaz = qutip.tensor(sigmaz, sigmaz)
    bperp_terms = [
        bField_samples[i] * np.sin(theta_samples[i]) * sigmax
        for i in range(systemsize)
    ]
    bparallel_terms = [
        bField_samples[i] * np.cos(theta_samples[i]) * sigmaz
        for i in range(systemsize)
    ]
    interaction_zz_terms = [
        jInt_samples[i] * sigmaz_sigmaz
        for i in range(systemsize - 1)
    ]

    hamiltonian = sum_single_site_terms(bperp_terms, systemsize)
    hamiltonian = hamiltonian + sum_single_site_terms(
        bparallel_terms, systemsize)
    return hamiltonian + sum_two_site_terms(
        interaction_zz_terms, systemsize)


def build_mbl_model(
        systemsize,
        jIntMean,
        jIntStd,
        bFieldMean,
        bFieldStd,
        anglePolarPiMin,
        anglePolarPiMax,
        rng=None):
    """Sample disorder and build a full :class:`MBLModel`."""
    sigma0, sigmax, sigmay, sigmaz = spin_operators()
    jInt_samples, bField_samples, theta_samples = sample_mbl_disorder(
        systemsize=systemsize,
        jIntMean=jIntMean,
        jIntStd=jIntStd,
        bFieldMean=bFieldMean,
        bFieldStd=bFieldStd,
        anglePolarPiMin=anglePolarPiMin,
        anglePolarPiMax=anglePolarPiMax,
        rng=rng,
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


__all__ = [
    "MBLModel",
    "build_mbl_hamiltonian",
    "build_mbl_model",
    "sample_mbl_disorder",
]
