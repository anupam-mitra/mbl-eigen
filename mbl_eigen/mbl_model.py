"""Many-body localized (MBL) random-field spin-chain model builder."""

from dataclasses import dataclass

import numpy as np
import scipy.stats

from .operators import (
    build_site_operator_array,
    spin_operators,
    sum_single_site_terms,
    sum_two_site_terms,
)


@dataclass
class MBLModel:
    """Container for a single realisation of the random-field MBL model.

    Attributes
    ----------
    hamiltonian : qutip.Qobj
        Full-system Hamiltonian.
    sigma0, sigmax, sigmay, sigmaz : qutip.Qobj
        Single-site Pauli matrices (and identity).
    jInt_samples : numpy.ndarray, shape (systemsize - 1,)
        Drawn nearest-neighbour ZZ coupling strengths.
    bField_samples : numpy.ndarray, shape (systemsize,)
        Drawn local field magnitudes.
    theta_samples : numpy.ndarray, shape (systemsize,)
        Drawn polar angles of each local field (in radians).
    """

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
):
    """Draw a single disorder realisation for the MBL Hamiltonian.

    Parameters
    ----------
    systemsize : int
        Number of spin-1/2 sites.
    jIntMean, jIntStd : float
        Mean and standard deviation of the normal distribution for the
        nearest-neighbour ZZ coupling strengths.
    bFieldMean, bFieldStd : float
        Mean and standard deviation of the normal distribution for the
        local field magnitudes.
    anglePolarPiMin, anglePolarPiMax : float
        Lower and upper bounds of the uniform distribution for polar field
        angles, in units of π.

    Returns
    -------
    jInt_samples : numpy.ndarray, shape (systemsize - 1,)
    bField_samples : numpy.ndarray, shape (systemsize,)
    theta_samples : numpy.ndarray, shape (systemsize,)
    """
    jInt_samples = scipy.stats.norm.rvs(
        size=(systemsize - 1), loc=jIntMean, scale=jIntStd
    )
    bField_samples = scipy.stats.norm.rvs(
        size=systemsize, loc=bFieldMean, scale=bFieldStd
    )
    theta_samples = scipy.stats.uniform.rvs(
        size=systemsize,
        loc=anglePolarPiMin * np.pi,
        scale=(anglePolarPiMax - anglePolarPiMin) * np.pi,
    )
    return jInt_samples, bField_samples, theta_samples


def build_mbl_hamiltonian(
    systemsize,
    jInt_samples,
    bField_samples,
    theta_samples,
    sigma0,
    sigmax,
    sigmaz,
):
    """Assemble the random-field MBL Hamiltonian.

    The Hamiltonian is::

        H = Σ_i b_i sin(θ_i) X_i  +  Σ_i b_i cos(θ_i) Z_i
            +  Σ_<ij> J_ij Z_i Z_j

    Parameters
    ----------
    systemsize : int
        Number of spin-1/2 sites.
    jInt_samples : numpy.ndarray, shape (systemsize - 1,)
        Nearest-neighbour ZZ couplings.
    bField_samples : numpy.ndarray, shape (systemsize,)
        Local field magnitudes.
    theta_samples : numpy.ndarray, shape (systemsize,)
        Polar angles.
    sigma0, sigmax, sigmaz : qutip.Qobj
        Single-site identity, X, and Z operators.

    Returns
    -------
    qutip.Qobj
        Full-system Hamiltonian.
    """
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
        jInt_samples[i] * sigmaz_sigmaz for i in range(systemsize - 1)
    ]

    hamiltonian = sum_single_site_terms(bperp_terms, systemsize)
    hamiltonian = hamiltonian + sum_single_site_terms(bparallel_terms, systemsize)
    hamiltonian = hamiltonian + sum_two_site_terms(interaction_zz_terms, systemsize)
    return hamiltonian


def build_mbl_model(
    systemsize,
    jIntMean,
    jIntStd,
    bFieldMean,
    bFieldStd,
    anglePolarPiMin,
    anglePolarPiMax,
):
    """Sample disorder and build a full :class:`MBLModel` in one call.

    Parameters
    ----------
    systemsize : int
        Number of spin-1/2 sites.
    jIntMean, jIntStd : float
        Normal distribution parameters for nearest-neighbour ZZ couplings.
    bFieldMean, bFieldStd : float
        Normal distribution parameters for local field magnitudes.
    anglePolarPiMin, anglePolarPiMax : float
        Uniform distribution bounds for polar field angles, in units of π.

    Returns
    -------
    MBLModel
    """
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
