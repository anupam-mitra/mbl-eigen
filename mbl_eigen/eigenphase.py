"""Eigenvalue-to-eigenphase conversion utilities."""

import numpy as np


def eigenvalues_to_unitary(eigenvalues, tduration):
    """Convert real energy eigenvalues to unitary eigenvalues via time evolution.

    Computes ``exp(-i * eigenvalues * tduration)``.

    Parameters
    ----------
    eigenvalues : numpy.ndarray
        Real Hamiltonian eigenvalues.
    tduration : float
        Evolution time.

    Returns
    -------
    numpy.ndarray of complex
        Unit-circle eigenvalues.
    """
    return np.exp(-1j * np.asarray(eigenvalues, dtype=float) * float(tduration))


def extract_sorted_eigenphases(eigenvalues_unitary):
    """Extract eigenphases in ``[0, 2π)`` from complex unit-circle eigenvalues.

    Parameters
    ----------
    eigenvalues_unitary : numpy.ndarray of complex
        Complex numbers on the unit circle.

    Returns
    -------
    numpy.ndarray of float
        Sorted eigenphases in radians, each in ``[0, 2π)``.
    """
    return np.sort(np.angle(np.asarray(eigenvalues_unitary)) % (2.0 * np.pi))


def build_time_propagator_phases(energies, times):
    """Build a 2-D array of propagator phases ``exp(-i E t)``.

    Parameters
    ----------
    energies : numpy.ndarray, shape (D,)
        Energy eigenvalues.
    times : numpy.ndarray, shape (T,)
        Time points.

    Returns
    -------
    numpy.ndarray of complex, shape (T, D)
        ``phases[ix_t, ix_e] = exp(-i * energies[ix_e] * times[ix_t])``.
    """
    energies = np.asarray(energies, dtype=float)
    times = np.asarray(times, dtype=float)
    return np.exp(-1j * np.outer(times, energies))


__all__ = [
    "build_time_propagator_phases",
    "eigenvalues_to_unitary",
    "extract_sorted_eigenphases",
]
