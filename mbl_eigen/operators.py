"""Shared spin-operator construction and manipulation utilities."""

import numpy as np
import qutip
from qutip.qip.operations import expand_operator


def spin_operators():
    """Return the four single-site Pauli matrices as QuTiP Qobj.

    Returns
    -------
    sigma0 : qutip.Qobj
        2×2 identity.
    sigmax : qutip.Qobj
        Pauli X.
    sigmay : qutip.Qobj
        Pauli Y.
    sigmaz : qutip.Qobj
        Pauli Z.
    """
    sigma0 = qutip.qeye(2)
    sigmax = qutip.sigmax()
    sigmay = qutip.sigmay()
    sigmaz = qutip.sigmaz()
    return sigma0, sigmax, sigmay, sigmaz


def zero_operator(systemsize):
    """Return the zero operator on a chain of *systemsize* spin-1/2 sites.

    Parameters
    ----------
    systemsize : int
        Number of sites.

    Returns
    -------
    qutip.Qobj
        Zero operator with the correct tensor-product structure.
    """
    return 0.0 * expand_operator(qutip.qeye(2), N=systemsize, targets=(0,))


def sum_single_site_terms(terms, systemsize):
    """Sum a sequence of single-site operators into the full Hilbert space.

    Each element of *terms* is a single-site ``qutip.Qobj``; it is expanded to
    site ``ix_site`` (the element's position in the sequence) and accumulated.

    Parameters
    ----------
    terms : sequence of qutip.Qobj
        Single-site operators, one per site (length must equal *systemsize*).
    systemsize : int
        Total number of sites.

    Returns
    -------
    qutip.Qobj
        Sum expanded to the full Hilbert space.
    """
    result = zero_operator(systemsize)
    for ix_site, term in enumerate(terms):
        result = result + expand_operator(term, N=systemsize, targets=(ix_site,))
    return result


def sum_two_site_terms(terms, systemsize):
    """Sum nearest-neighbour two-site operators into the full Hilbert space.

    Element ``ix_site`` of *terms* acts on sites ``(ix_site, ix_site + 1)``.

    Parameters
    ----------
    terms : sequence of qutip.Qobj
        Two-site operators (length must equal ``systemsize - 1``).
    systemsize : int
        Total number of sites.

    Returns
    -------
    qutip.Qobj
        Sum expanded to the full Hilbert space.
    """
    result = zero_operator(systemsize)
    for ix_site, term in enumerate(terms):
        result = result + expand_operator(
            term, N=systemsize, targets=(ix_site, ix_site + 1)
        )
    return result


def build_site_operator_array(operator, systemsize, *, normalize=False):
    """Expand a single-site operator to every site and collect in an array.

    Parameters
    ----------
    operator : qutip.Qobj
        Single-site operator.
    systemsize : int
        Total number of sites.
    normalize : bool, optional
        When *True*, divide by ``sqrt(2**systemsize)`` before expansion
        (used in ``run_mbl_propagator``).

    Returns
    -------
    numpy.ndarray of qutip.Qobj, shape (systemsize,)
        Full-system operator at each site.
    """
    op = operator / np.sqrt(1 << systemsize) if normalize else operator
    return np.asarray(
        [
            expand_operator(op, N=systemsize, targets=(ix_site,))
            for ix_site in range(systemsize)
        ],
        dtype=object,
    )


def rotate_to_eigenbasis(operators, basis_changer):
    """Rotate a sequence of operators into a new basis via ``U† O U``.

    Parameters
    ----------
    operators : sequence of qutip.Qobj
        Operators in the original basis.
    basis_changer : qutip.Qobj
        Unitary ``U`` whose columns are the new basis vectors.

    Returns
    -------
    list of qutip.Qobj
        Operators in the new basis.
    """
    u = basis_changer
    return [u.dag() * op * u for op in operators]


__all__ = [
    "build_site_operator_array",
    "rotate_to_eigenbasis",
    "spin_operators",
    "sum_single_site_terms",
    "sum_two_site_terms",
    "zero_operator",
]
