"""Shared matplotlib plotting routines for MBL eigenvalue analysis."""

import numpy as np
import matplotlib.pyplot as plt


# Default figure size used consistently across all eigenphase plots.
_SQUARE_FIGSIZE = (12.0 / 2.54, 12.0 / 2.54)
_WIDE_FIGSIZE = (18.0 / 2.54, 12.0 / 2.54)


def plot_eigenphases_unit_circle(eigenvalues_unitary, filename, *, figsize=None):
    """Plot complex eigenvalues and their conjugates on the unit circle.

    Saves a PDF to *filename* and closes the figure.

    Parameters
    ----------
    eigenvalues_unitary : numpy.ndarray of complex
        Eigenvalues lying on (or near) the complex unit circle.
    filename : str
        Output path (typically a ``.pdf`` file).
    figsize : tuple of float, optional
        ``(width, height)`` in inches.  Defaults to a 12×12 cm square.
    """
    if figsize is None:
        figsize = _SQUARE_FIGSIZE

    re = np.real(eigenvalues_unitary)
    im = np.imag(eigenvalues_unitary)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xlabel(r"$\mathrm{Re}(\eta)$")
    ax.set_ylabel(r"$\mathrm{Im}(\eta)$")

    ax.plot(re, im, marker=".", ls="", label=r"$\eta$")
    ax.plot(re, -im, marker="o", fillstyle="none", ls="", label=r"$\eta^*$")

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.legend(loc="center")

    plt.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def plot_eigenvector_entropy(
    eigenvalues, entropies, page_entropy, filename, *, figsize=None
):
    """Plot half-chain von Neumann entanglement entropy vs energy.

    Saves a PDF to *filename* and closes the figure.

    Parameters
    ----------
    eigenvalues : numpy.ndarray of float
        Hamiltonian eigenvalues (x-axis).
    entropies : numpy.ndarray of float
        Half-chain entanglement entropies (y-axis), one per eigenstate.
    page_entropy : float
        Expected Page-limit entropy; drawn as a horizontal dotted line.
    filename : str
        Output path.
    figsize : tuple of float, optional
        ``(width, height)`` in inches.  Defaults to 18×12 cm.
    """
    if figsize is None:
        figsize = _WIDE_FIGSIZE

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xlabel(r"Energy")
    ax.set_ylabel(r"$\mathcal{S}_1$")

    ax.plot(
        eigenvalues,
        entropies,
        marker="o",
        markeredgecolor="k",
        markerfacecolor="blue",
        ls="",
    )
    ax.axhline(page_entropy, ls="dotted", color="k")

    fig.savefig(filename)
    plt.close(fig)


def plot_return_rate(times, amplitudes, systemsize, filename, *, figsize=None):
    """Plot the normalised return rate  λ(t) = −log|A(t)|² / N  vs time.

    Saves a PDF to *filename* and closes the figure.

    Parameters
    ----------
    times : numpy.ndarray of float
        Time grid.
    amplitudes : numpy.ndarray of complex
        Return amplitudes ``A(t)``.
    systemsize : int
        Number of sites (used for normalisation).
    filename : str
        Output path.
    figsize : tuple of float, optional
        ``(width, height)`` in inches.  Defaults to 18×12 cm.
    """
    if figsize is None:
        figsize = _WIDE_FIGSIZE

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(times, -np.log(np.abs(amplitudes) ** 2) / systemsize)
    ax.set_ylabel(r"$\lambda(t)$")
    ax.set_xlabel(r"$B_{\mathrm{mean}} t$")

    fig.savefig(filename)
    plt.close(fig)


def plot_magnetization_z(times, magnetization_z, filename, *, figsize=None):
    """Plot site-resolved ⟨Z_i⟩(t) vs time.

    Saves a PDF to *filename* and closes the figure.

    Parameters
    ----------
    times : numpy.ndarray of float
        Time grid.
    magnetization_z : numpy.ndarray of float
        Site-resolved magnetization ``(N, T)``.
    filename : str
        Output path.
    figsize : tuple of float, optional
        ``(width, height)`` in inches.  Defaults to 18×12 cm.
    """
    if figsize is None:
        figsize = _WIDE_FIGSIZE

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    systemsize = magnetization_z.shape[0]
    for i in range(systemsize):
        ax.plot(times, magnetization_z[i, :], label=rf"Site {i}")
        
    ax.set_ylabel(r"$\langle Z_i \rangle(t)$")
    ax.set_xlabel(r"$t$")
    ax.legend(loc="best")

    fig.savefig(filename)
    plt.close(fig)


__all__ = [
    "plot_eigenphases_unit_circle",
    "plot_eigenvector_entropy",
    "plot_magnetization_z",
    "plot_return_rate",
]
