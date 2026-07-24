"""QMBS Hamiltonian builders for the Rydberg / PXP model."""

import qutip

from .operators import spin_operators, sum_single_site_terms, sum_two_site_terms, zero_operator


def build_qmbs_ising_hamiltonian(systemsize, Omega, Delta, Vrr):
    """Build the Rydberg-blockade Ising Hamiltonian.

    The Hamiltonian is::

        H = Ω/2 Σ_i X_i  +  Δ/2 Σ_i Z_i  +  V_rr Σ_<ij> P_r^i P_r^j

    where ``P_r = (I - Z) / 2`` is the projector onto the Rydberg state.

    Parameters
    ----------
    systemsize : int
        Number of spin-1/2 sites.
    Omega : float
        Rabi frequency.
    Delta : float
        Detuning (already in energy units, not multiples of Omega).
    Vrr : float
        Rydberg–Rydberg interaction strength.

    Returns
    -------
    qutip.Qobj
        Full-system Hamiltonian.
    """
    sigma0, sigmax, _, sigmaz = spin_operators()

    projector_r = (sigma0 - sigmaz) * 0.5
    projector_rr = qutip.tensor(projector_r, projector_r)

    drive_x_terms = [Omega * 0.5 * sigmax for _ in range(systemsize)]
    detuning_z_terms = [Delta * 0.5 * sigmaz for _ in range(systemsize)]
    interaction_rr_terms = [Vrr * projector_rr for _ in range(systemsize - 1)]

    hamiltonian = sum_single_site_terms(drive_x_terms, systemsize)
    hamiltonian = hamiltonian + sum_single_site_terms(detuning_z_terms, systemsize)
    hamiltonian = hamiltonian + sum_two_site_terms(interaction_rr_terms, systemsize)
    return hamiltonian


def build_pxp_hamiltonian(systemsize, Omega, Delta):
    """Build the constrained PXP Hamiltonian.

    Each site's drive is projected by the ground-state projectors on its
    neighbours so that consecutive Rydberg excitations are forbidden.

    The Hamiltonian is::

        H_PXP = Ω/2 Σ_i P_g^{i-1} X_i P_g^{i+1}  +  Δ/2 Σ_i Z_i

    with open boundary conditions (edge sites only have one neighbour projector).

    Parameters
    ----------
    systemsize : int
        Number of spin-1/2 sites.
    Omega : float
        Rabi frequency.
    Delta : float
        Detuning (already in energy units).

    Returns
    -------
    qutip.Qobj
        Full-system PXP Hamiltonian.
    """
    from qutip.qip.operations import expand_operator

    sigma0, sigmax, _, sigmaz = spin_operators()

    projector_g = (sigma0 + sigmaz) * 0.5

    # Build site-local PXP terms (variable-rank operators: 2-site at edges,
    # 3-site in the bulk) then expand to the full chain.
    pxp_terms = (
        [0.5 * Omega * qutip.tensor(sigmax, projector_g)]
        + [
            0.5 * Omega * qutip.tensor(projector_g, sigmax, projector_g)
            for _ in range(systemsize - 2)
        ]
        + [0.5 * Omega * qutip.tensor(projector_g, sigmax)]
    )

    hamiltonian_pxp = zero_operator(systemsize)
    for ix_site, term in enumerate(pxp_terms):
        if ix_site == 0:
            targets = (ix_site, ix_site + 1)
        elif ix_site == systemsize - 1:
            targets = (ix_site - 1, ix_site)
        else:
            targets = (ix_site - 1, ix_site, ix_site + 1)
        hamiltonian_pxp = hamiltonian_pxp + expand_operator(
            term, N=systemsize, targets=targets
        )

    # Add detuning
    detuning_z_terms = [Delta * 0.5 * sigmaz for _ in range(systemsize)]
    hamiltonian_pxp = hamiltonian_pxp + sum_single_site_terms(
        detuning_z_terms, systemsize
    )
    return hamiltonian_pxp


__all__ = [
    "build_pxp_hamiltonian",
    "build_qmbs_ising_hamiltonian",
]
