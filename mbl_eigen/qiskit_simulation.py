"""Qiskit simulation engine for MBL propagator circuits.

This module bridges :mod:`mbl_eigen.qiskit_propagators` (which builds
``QuantumCircuit`` objects) with actual simulation backends, returning
physically meaningful observables (Loschmidt echo and site-resolved
magnetisation ⟨Z_i⟩).

Three backends are supported, selected via the ``backend`` parameter:

``"statevector"``
    Uses :class:`qiskit.quantum_info.Statevector` for exact state-vector
    simulation.  Requires only the base ``qiskit`` package — no Aer needed.

``"aer"``
    Uses :class:`qiskit_aer.AerSimulator` in ``statevector`` or shot-based
    mode.  Requires the optional ``qiskit-aer`` package.  Pass ``shots``
    to enable shot-based sampling; ``shots=None`` (default) runs in exact
    statevector mode via Aer.

``"fake_backend"``
    Uses a Qiskit Fake Provider backend (IBM hardware noise models) together
    with ``qiskit_aer``.  Requires ``qiskit-aer`` and
    ``qiskit-ibm-runtime`` (for the fake provider).  Always shot-based.

Example
-------
>>> from mbl_eigen.mbl_model import build_mbl_model
>>> from mbl_eigen.qiskit_simulation import run_mbl_qiskit_simulation
>>> import numpy as np
>>> model = build_mbl_model(
...     systemsize=4, jIntMean=1, jIntStd=1,
...     bFieldMean=1, bFieldStd=1,
...     anglePolarPiMin=0, anglePolarPiMax=1,
... )
>>> result = run_mbl_qiskit_simulation(
...     model, np.linspace(0, 2, 9),
...     trotter_steps=8, backend="statevector",
... )
>>> result.return_rate.shape
(9,)
>>> result.magnetization_z.shape
(4, 9)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .qiskit_propagators import build_mbl_trotter_circuit_from_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

QISKIT_SIM_BACKENDS = ("statevector", "aer", "fake_backend")

# Default fake backend name used when backend="fake_backend".
DEFAULT_FAKE_BACKEND_NAME = "FakeManilaV2"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """Observables extracted from a Qiskit MBL circuit simulation.

    Attributes
    ----------
    times : numpy.ndarray, shape (T,)
        Time grid (matches the input ``times_array``).
    return_rate : numpy.ndarray, shape (T,)
        Loschmidt echo  ``|⟨ψ₀|ψ(t)⟩|²``.  Equal to 1 at t = 0 (up to
        Trotter error) and decays toward zero for ergodic systems.
    magnetization_z : numpy.ndarray, shape (N, T)
        Site-resolved expectation value ``⟨Z_i⟩(t)`` for each qubit ``i``
        and each time step.  In the |1…1⟩ initial state (computational
        all-ones), every site starts at ``⟨Z⟩ = -1`` (Qiskit's Z
        eigenvalue for |1⟩).
    backend : str
        Which backend was used (one of :data:`QISKIT_SIM_BACKENDS`).
    shots : int or None
        Number of measurement shots, or ``None`` for exact simulation.
    """

    times: np.ndarray
    return_rate: np.ndarray
    magnetization_z: np.ndarray
    backend: str
    shots: int | None


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def run_mbl_qiskit_simulation(
        model,
        times_array,
        *,
        trotter_steps: int = 10,
        trotter_order: int = 2,
        backend: str = "statevector",
        shots: int | None = None,
        noise_model=None,
        fake_backend_name: str = DEFAULT_FAKE_BACKEND_NAME,
        insert_barriers: bool = False,
) -> SimulationResult:
    """Simulate MBL time evolution via Trotterized Qiskit circuits.

    For each time in *times_array* the function:

    1. Prepares the |1…1⟩ initial state (all qubits flipped from |0…0⟩).
    2. Appends a Trotterized MBL propagator for that time via
       :func:`~mbl_eigen.qiskit_propagators.build_mbl_trotter_circuit_from_model`.
    3. Runs the circuit on the selected *backend*.
    4. Extracts the Loschmidt echo and site-resolved ``⟨Z_i⟩``.

    Parameters
    ----------
    model : MBLModel
        A disorder realisation built by
        :func:`~mbl_eigen.mbl_model.build_mbl_model`.
    times_array : array-like of float
        1-D array of time points to simulate.
    trotter_steps : int, default 10
        Number of Trotter steps per unit time.
    trotter_order : {1, 2}, default 2
        Suzuki–Trotter order.
    backend : {"statevector", "aer", "fake_backend"}, default "statevector"
        Simulation backend.
    shots : int or None, default None
        Number of measurement shots.  ``None`` → exact statevector (for
        ``"statevector"`` and ``"aer"`` backends).  Must be set to a
        positive integer for ``"fake_backend"``.
    noise_model : qiskit_aer.noise.NoiseModel or None, default None
        Optional custom noise model for the ``"aer"`` backend.  Ignored for
        other backends.
    fake_backend_name : str, default "FakeManilaV2"
        Name of the IBM fake backend to instantiate when
        ``backend="fake_backend"``.
    insert_barriers : bool, default False
        Whether to insert barrier instructions between circuit layers.
        Useful for visualisation but ignored by most simulators.

    Returns
    -------
    SimulationResult
    """
    if backend not in QISKIT_SIM_BACKENDS:
        raise ValueError(
            "unsupported simulation backend %r; expected one of %s"
            % (backend, QISKIT_SIM_BACKENDS)
        )

    times_array = np.asarray(times_array, dtype=float)
    if times_array.ndim != 1:
        raise ValueError("times_array must be a 1-D array")

    systemsize = len(model.bField_samples)
    n_times = len(times_array)

    return_rate = np.empty(n_times, dtype=float)
    magnetization_z = np.empty((systemsize, n_times), dtype=float)

    logger.info(
        "run_mbl_qiskit_simulation: N=%d, T=%d time points, backend=%r, shots=%s",
        systemsize, n_times, backend, shots,
    )

    simulator = _build_simulator(backend, shots, noise_model, fake_backend_name)

    for ix_t, t in enumerate(times_array):
        logger.debug("Simulating t = %g (%d/%d)", t, ix_t + 1, n_times)

        circuit = _build_full_circuit(
            model=model,
            time=t,
            trotter_steps=max(1, int(round(trotter_steps * max(t, 1e-12)))),
            trotter_order=trotter_order,
            insert_barriers=insert_barriers,
        )

        sv = simulator(circuit)

        return_rate[ix_t] = _loschmidt_echo(sv, systemsize)
        magnetization_z[:, ix_t] = _site_magnetization_z(sv, systemsize)

    return SimulationResult(
        times=times_array,
        return_rate=return_rate,
        magnetization_z=magnetization_z,
        backend=backend,
        shots=shots,
    )


# ---------------------------------------------------------------------------
# Circuit preparation helpers
# ---------------------------------------------------------------------------

def _build_full_circuit(model, time, trotter_steps, trotter_order, insert_barriers):
    """Return state-prep + Trotter circuit (no measurements appended)."""
    from qiskit import QuantumCircuit

    systemsize = len(model.bField_samples)

    # --- Initial state: |1…1⟩ (flip all qubits from |0…0⟩) ---
    prep = QuantumCircuit(systemsize, name="state_prep")
    prep.x(range(systemsize))

    # --- Trotter propagator ---
    if time == 0.0:
        # Zero time → identity; return just the state-prep circuit
        return prep

    trotter = build_mbl_trotter_circuit_from_model(
        model=model,
        time=time,
        trotter_steps=trotter_steps,
        trotter_order=trotter_order,
        insert_barriers=insert_barriers,
    )

    return prep.compose(trotter)


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

def _build_simulator(backend, shots, noise_model, fake_backend_name):
    """Return a callable ``sim(circuit) -> Statevector | ndarray``."""
    if backend == "statevector":
        return _StatevectorSim()

    if backend == "aer":
        return _AerSim(shots=shots, noise_model=noise_model)

    # backend == "fake_backend"
    return _FakeBackendSim(
        fake_backend_name=fake_backend_name,
        shots=shots if shots is not None else 1024,
    )


class _StatevectorSim:
    """Exact simulation via ``qiskit.quantum_info.Statevector``."""

    def __call__(self, circuit):
        try:
            from qiskit.quantum_info import Statevector
        except ImportError as exc:
            raise ImportError(
                "statevector backend requires 'qiskit'; install with "
                "'python -m pip install .[qiskit]'"
            ) from exc

        sv = Statevector.from_instruction(circuit)
        return sv.data  # numpy complex128 array, shape (2**N,)


class _AerSim:
    """Simulation via ``qiskit_aer.AerSimulator``."""

    def __init__(self, shots, noise_model):
        try:
            from qiskit_aer import AerSimulator
        except ImportError as exc:
            raise ImportError(
                "aer backend requires 'qiskit-aer'; install with "
                "'python -m pip install qiskit-aer'"
            ) from exc

        if shots is None:
            # Exact statevector mode in Aer
            self._sim = AerSimulator(method="statevector", noise_model=noise_model)
            self._shots = None
        else:
            self._sim = AerSimulator(noise_model=noise_model)
            self._shots = shots

    def __call__(self, circuit):
        from qiskit import transpile
        from qiskit.quantum_info import Statevector

        if self._shots is None:
            # Save statevector and retrieve it
            c = circuit.copy()
            c.save_statevector()
            tc = transpile(c, self._sim)
            job = self._sim.run(tc)
            result = job.result()
            sv_data = np.asarray(result.get_statevector(tc), dtype=np.complex128)
            return sv_data
        else:
            # Shot-based: transpile, measure all, run
            c = circuit.copy()
            c.measure_all()
            tc = transpile(c, self._sim)
            job = self._sim.run(tc, shots=self._shots)
            counts = job.result().get_counts()
            return _counts_to_statevector_proxy(counts, circuit.num_qubits, self._shots)


class _FakeBackendSim:
    """Noisy simulation using an IBM Fake Provider backend."""

    def __init__(self, fake_backend_name, shots):
        try:
            from qiskit_aer import AerSimulator
        except ImportError as exc:
            raise ImportError(
                "fake_backend requires 'qiskit-aer'"
            ) from exc

        fake_backend = _load_fake_backend(fake_backend_name)
        self._sim = AerSimulator.from_backend(fake_backend)
        self._shots = shots

    def __call__(self, circuit):
        from qiskit import transpile

        c = circuit.copy()
        c.measure_all()
        tc = transpile(c, self._sim)
        job = self._sim.run(tc, shots=self._shots)
        counts = job.result().get_counts()
        return _counts_to_statevector_proxy(counts, circuit.num_qubits, self._shots)


def _load_fake_backend(name):
    """Attempt to load a Qiskit fake backend by name from known locations."""
    # qiskit-ibm-runtime >= 0.20 puts fakes here
    providers = [
        "qiskit_ibm_runtime.fake_provider",
        "qiskit.providers.fake_provider",
    ]
    for module_path in providers:
        try:
            import importlib
            module = importlib.import_module(module_path)
            cls = getattr(module, name, None)
            if cls is not None:
                logger.info("Loaded fake backend %r from %s", name, module_path)
                return cls()
        except ImportError:
            continue

    raise ImportError(
        "Could not load fake backend %r from any of %s. "
        "Install 'qiskit-ibm-runtime' or 'qiskit' (>=1.0) and ensure "
        "the backend name is correct." % (name, providers)
    )


# ---------------------------------------------------------------------------
# Observable extraction
# ---------------------------------------------------------------------------

def _loschmidt_echo(sv_data, systemsize):
    """Return |⟨ψ₀|ψ(t)⟩|² where ψ₀ = |1…1⟩.

    In Qiskit's little-endian convention the all-ones computational basis
    state |1…1⟩ corresponds to the last index of the statevector
    (index 2**N − 1).
    """
    if isinstance(sv_data, _ShotProxy):
        # Shot-based: estimate from empirical probability
        all_ones = "1" * systemsize
        return sv_data.prob(all_ones)

    # Exact statevector
    all_ones_index = (1 << systemsize) - 1
    return float(np.abs(sv_data[all_ones_index]) ** 2)


def _site_magnetization_z(sv_data, systemsize):
    """Compute ⟨Z_i⟩ for each site i.

    Uses the exact statevector when available, or the empirical
    probability distribution from shot-based counts otherwise.

    In Qiskit's computational basis: Z|0⟩ = +1, Z|1⟩ = −1.
    Qubit i contributes −1 if bit i of the basis state index is 1.
    """
    if isinstance(sv_data, _ShotProxy):
        return sv_data.magnetization_z(systemsize)

    # Exact: probabilities from statevector amplitudes
    probs = np.abs(sv_data) ** 2  # shape (2**N,)
    indices = np.arange(len(probs), dtype=np.int64)
    mag = np.empty(systemsize, dtype=float)
    for i in range(systemsize):
        # bit i set → eigenvalue −1; bit i clear → +1
        bits_i = ((indices >> i) & 1).astype(float)
        eigenvalues_i = 1.0 - 2.0 * bits_i  # +1 or −1
        mag[i] = float(np.dot(probs, eigenvalues_i))
    return mag


# ---------------------------------------------------------------------------
# Shot-based helper
# ---------------------------------------------------------------------------

class _ShotProxy:
    """Lightweight wrapper around Qiskit ``counts`` for expectation values."""

    def __init__(self, counts: dict, num_qubits: int, total_shots: int):
        self._counts = counts
        self._num_qubits = num_qubits
        self._total_shots = total_shots

    def prob(self, bitstring: str) -> float:
        """Empirical probability of *bitstring* (e.g. '1111')."""
        return self._counts.get(bitstring, 0) / self._total_shots

    def magnetization_z(self, systemsize: int) -> np.ndarray:
        """Site-resolved ⟨Z_i⟩ from shot counts.

        Qiskit returns bitstrings in big-endian order for the *printed*
        counts key (leftmost character = highest-index qubit), so we
        reverse when indexing by site.
        """
        mag = np.zeros(systemsize, dtype=float)
        for bitstring, count in self._counts.items():
            # Qiskit count keys may contain spaces; strip them
            bits = bitstring.replace(" ", "")
            # Reverse: bits[0] is the highest-index qubit in Qiskit
            bits_reversed = bits[::-1]
            for i in range(systemsize):
                b = int(bits_reversed[i]) if i < len(bits_reversed) else 0
                mag[i] += (1.0 - 2.0 * b) * count
        return mag / self._total_shots


def _counts_to_statevector_proxy(counts, num_qubits, shots):
    return _ShotProxy(counts, num_qubits, shots)


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    "DEFAULT_FAKE_BACKEND_NAME",
    "QISKIT_SIM_BACKENDS",
    "SimulationResult",
    "run_mbl_qiskit_simulation",
]
