"""Qiskit simulation engine for MBL propagator circuits.

This module bridges :mod:`mbl_eigen.qiskit_propagators` (which builds
``QuantumCircuit`` objects) with actual simulation backends, returning
physically meaningful observables (Loschmidt echo and site-resolved
magnetisation ⟨Z_i⟩).

Three backends are supported, selected via the ``backend`` parameter:

``"statevector"``
    Uses :class:`qiskit.quantum_info.Statevector` for exact state-vector
    simulation.  Evolution is incremental: the prepared state is advanced
    once per time interval ``Δt`` instead of re-simulating the full
    circuit from scratch at every time point (full per-point rebuilds are
    used when the time grid is not monotonically increasing).  Requires
    only the base ``qiskit`` package — no Aer needed.

``"aer"``
    Uses :class:`qiskit_aer.AerSimulator` in ``statevector`` or shot-based
    mode.  Requires the optional ``qiskit-aer`` package.  Pass ``shots``
    to enable shot-based sampling; ``shots=None`` (default) runs in exact
    statevector mode via Aer.

``"fake_backend"``
    Uses a Qiskit Fake Provider backend (IBM hardware noise models) together
    with ``qiskit_aer``.  Requires ``qiskit-aer`` and
    ``qiskit-ibm-runtime`` (for the fake provider).  Always shot-based.
    Local readout-error mitigation (tensor-product confusion-matrix
    inversion from all-zeros/all-ones calibration circuits) is applied by
    default; disable with ``mitigate_readout=False``.  Calibration runs
    once per simulation run, and the confusion matrix follows the
    transpiled virtual→physical qubit layout.

``steps_per_unit_time`` in :func:`run_mbl_qiskit_simulation` counts
Trotter steps per unit time (scaled per time point or per evolution
interval ``Δt``), whereas ``n_steps`` in
:mod:`mbl_eigen.qiskit_propagators` is the total step count for the
given evolution time.

Shot-based runs (``"aer"`` with ``shots`` and ``"fake_backend"``) also
populate binomial standard-error fields ``return_rate_stderr`` and
``magnetization_stderr``, computed from the raw (pre-mitigation) counts
so they measure sampling error only; exact runs return zeros.

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
...     steps_per_unit_time=8, backend="statevector",
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

# Epsilon floor for time-scaled Trotter step counts (guards t ≈ 0).
_TROTTER_TIME_EPSILON = 1e-12

# Per-job timeout (seconds) passed to Aer ``run``; guards hung jobs.
_JOB_TIMEOUT_SECONDS = 300.0


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
    return_rate_stderr : numpy.ndarray, shape (T,)
        Binomial standard error ``sqrt(p (1 − p) / shots)`` of the return
        rate for shot-based runs, with ``p`` the raw (pre-mitigation)
        measured probability of the all-ones bitstring.  This is the
        pre-mitigation sampling error: the reported point values use the
        mitigated probabilities, the standard error does not.  Zeros for
        exact simulation.
    magnetization_stderr : numpy.ndarray, shape (N, T)
        Standard error ``2 sqrt(p (1 − p) / shots)`` of ``⟨Z_i⟩`` for
        shot-based runs, with ``p`` the raw (pre-mitigation) measured
        probability of |1⟩ at site ``i`` (pre-mitigation sampling error;
        see ``return_rate_stderr``).  Zeros for exact simulation.
    backend : str
        Which backend was used (one of :data:`QISKIT_SIM_BACKENDS`).
    shots : int or None
        Number of measurement shots, or ``None`` for exact simulation.
    """

    times: np.ndarray
    return_rate: np.ndarray
    magnetization_z: np.ndarray
    return_rate_stderr: np.ndarray
    magnetization_stderr: np.ndarray
    backend: str
    shots: int | None


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def run_mbl_qiskit_simulation(
        model,
        times_array,
        *,
        steps_per_unit_time: int = 10,
        trotter_order: int = 2,
        backend: str = "statevector",
        shots: int | None = None,
        noise_model=None,
        fake_backend_name: str = DEFAULT_FAKE_BACKEND_NAME,
        insert_barriers: bool = False,
        mitigate_readout: bool = True,
        seed: int | None = None,
) -> SimulationResult:
    """Simulate MBL time evolution via Trotterized Qiskit circuits.

    The |1…1⟩ initial state is prepared and evolved through the times in
    *times_array*:

    1. Prepares the |1…1⟩ initial state (all qubits flipped from |0…0⟩).
    2. Appends a Trotterized MBL propagator via
       :func:`~mbl_eigen.qiskit_propagators.build_mbl_trotter_circuit_from_model`.
    3. Runs the circuit on the selected *backend*.
    4. Extracts the Loschmidt echo and site-resolved ``⟨Z_i⟩``.

    The exact ``"statevector"`` backend evolves incrementally: the
    prepared :class:`~qiskit.quantum_info.Statevector` is advanced from
    one time point to the next by a per-interval ``Δt = t_i − t_{i−1}``
    Trotter circuit (``n_steps`` scaled by ``steps_per_unit_time`` per
    interval), avoiding a full re-simulation per point.  If *times_array*
    is not monotonically increasing, the statevector backend falls back
    to full per-point rebuilds.  The shot-based backends always rebuild
    the full circuit per point because measurement collapses the state.

    Parameters
    ----------
    model : MBLModel
        A disorder realisation built by
        :func:`~mbl_eigen.mbl_model.build_mbl_model`.
    times_array : array-like of float
        1-D array of time points to simulate.
    steps_per_unit_time : int, default 10
        Number of Trotter steps per unit time.  For each evolution
        interval ``Δt`` (and for each full-circuit time point on
        non-incremental paths) the propagator is built with
        ``max(1, round(steps_per_unit_time * abs(Δt)))`` total steps.
        This differs from ``n_steps`` in
        :func:`~mbl_eigen.qiskit_propagators.build_mbl_trotter_circuit`,
        which is the total step count for the given evolution time.
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
    mitigate_readout : bool, default True
        Apply local readout-error mitigation for the ``"fake_backend"``
        backend: run all-zeros and all-ones calibration circuits (once
        per simulation run), build the tensor-product readout confusion
        matrix using the transpiled virtual→physical qubit layout,
        invert it, and map the measured counts to corrected
        probabilities (negative values clipped to zero, then
        renormalised; a singular confusion matrix falls back to the raw
        counts).  Standard errors are always computed from the raw
        pre-mitigation counts.  Ignored by the other backends.
    seed : int or None, default None
        Random seed forwarded as ``seed_simulator`` and
        ``seed_transpiler`` to the Aer and fake-backend simulators.
        ``None`` → unseeded runs.  Ignored by the exact ``"statevector"``
        backend.

    Returns
    -------
    SimulationResult
    """
    if backend not in QISKIT_SIM_BACKENDS:
        raise ValueError(
            "unsupported simulation backend %r; expected one of %s"
            % (backend, QISKIT_SIM_BACKENDS)
        )

    if np.iscomplexobj(times_array):
        raise ValueError("times_array must contain real values")
    times_array = np.asarray(times_array, dtype=float)
    if times_array.ndim != 1:
        raise ValueError("times_array must be a 1-D array")
    if len(times_array) == 0:
        raise ValueError("times_array must not be empty")
    if not np.isfinite(times_array).all():
        raise ValueError("times_array must contain only finite values")
    if (isinstance(steps_per_unit_time, bool)
            or not isinstance(steps_per_unit_time, (int, np.integer))
            or steps_per_unit_time <= 0):
        raise ValueError("steps_per_unit_time must be a positive integer")
    if trotter_order not in (1, 2):
        raise ValueError("trotter_order must be 1 or 2")
    if shots is not None and (
            isinstance(shots, bool)
            or not isinstance(shots, (int, np.integer))
            or shots <= 0
    ):
        raise ValueError("shots must be a positive integer or None")
    if backend == "statevector" and shots is not None:
        raise ValueError("shots are only supported by aer and fake_backend")
    if backend == "fake_backend" and shots is None:
        raise ValueError("fake_backend requires a positive shots value")
    if seed is not None and (
            isinstance(seed, bool)
            or not isinstance(seed, (int, np.integer))
            or seed < 0
    ):
        raise ValueError("seed must be a non-negative integer or None")
    if not isinstance(mitigate_readout, bool):
        raise ValueError("mitigate_readout must be a boolean")

    systemsize = len(model.bField_samples)
    n_times = len(times_array)
    incremental = backend == "statevector" and bool(
        np.all(np.diff(times_array) >= 0.0)
    )

    return_rate = np.empty(n_times, dtype=float)
    magnetization_z = np.empty((systemsize, n_times), dtype=float)
    return_rate_stderr = np.zeros(n_times, dtype=float)
    magnetization_stderr = np.zeros((systemsize, n_times), dtype=float)

    logger.info(
        "run_mbl_qiskit_simulation: N=%d, T=%d time points, backend=%r, "
        "shots=%s, incremental=%s, mitigate_readout=%s",
        systemsize, n_times, backend, shots, incremental, mitigate_readout,
    )

    simulator = _build_simulator(
        backend, shots, noise_model, fake_backend_name, seed, systemsize,
        mitigate_readout,
    )

    state = None
    for ix_t, t in enumerate(times_array):
        logger.debug("Simulating t = %g (%d/%d)", t, ix_t + 1, n_times)

        if incremental:
            if ix_t == 0:
                base_circuit = _build_full_circuit(
                    model=model,
                    time=t,
                    n_steps=max(
                        1,
                        int(round(
                            steps_per_unit_time
                            * max(abs(t), _TROTTER_TIME_EPSILON)
                        )),
                    ),
                    trotter_order=trotter_order,
                    insert_barriers=insert_barriers,
                )
                state = _circuit_statevector(base_circuit)
            else:
                dt = float(t - times_array[ix_t - 1])
                if dt != 0.0:
                    dt_circuit = build_mbl_trotter_circuit_from_model(
                        model=model,
                        time=dt,
                        n_steps=max(
                            1,
                            int(round(
                                steps_per_unit_time
                                * max(abs(dt), _TROTTER_TIME_EPSILON)
                            )),
                        ),
                        trotter_order=trotter_order,
                        insert_barriers=insert_barriers,
                    )
                    state = state.evolve(dt_circuit)
            sv = state.data
        else:
            circuit = _build_full_circuit(
                model=model,
                time=t,
                n_steps=max(
                    1,
                    int(round(
                        steps_per_unit_time * max(abs(t), _TROTTER_TIME_EPSILON)
                    )),
                ),
                trotter_order=trotter_order,
                insert_barriers=insert_barriers,
            )

            sv = simulator(circuit)

        return_rate[ix_t] = _loschmidt_echo(sv, systemsize)
        magnetization_z[:, ix_t] = _site_magnetization_z(sv, systemsize)
        if isinstance(sv, _ShotProxy):
            return_rate_stderr[ix_t] = sv.prob_stderr("1" * systemsize)
            magnetization_stderr[:, ix_t] = sv.magnetization_z_stderr(
                systemsize
            )

    return SimulationResult(
        times=times_array,
        return_rate=return_rate,
        magnetization_z=magnetization_z,
        return_rate_stderr=return_rate_stderr,
        magnetization_stderr=magnetization_stderr,
        backend=backend,
        shots=shots,
    )


# ---------------------------------------------------------------------------
# Circuit preparation helpers
# ---------------------------------------------------------------------------

def _build_full_circuit(model, time, n_steps, trotter_order, insert_barriers):
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
        n_steps=n_steps,
        trotter_order=trotter_order,
        insert_barriers=insert_barriers,
    )

    return prep.compose(trotter)


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

def _build_simulator(backend, shots, noise_model, fake_backend_name, seed,
                     systemsize, mitigate_readout):
    """Return a callable ``sim(circuit) -> Statevector | ndarray``.

    ``seed`` is forwarded to the seeded simulators; ``systemsize``
    enables the fake-backend qubit-count guard; ``mitigate_readout``
    enables fake-backend readout-error mitigation.
    """
    if backend == "statevector":
        return _StatevectorSim(seed=seed)

    if backend == "aer":
        return _AerSim(shots=shots, noise_model=noise_model, seed=seed)

    # backend == "fake_backend"
    return _FakeBackendSim(
        fake_backend_name=fake_backend_name,
        shots=shots,
        systemsize=systemsize,
        seed=seed,
        mitigate_readout=mitigate_readout,
    )


def _circuit_statevector(circuit):
    """Return the exact :class:`~qiskit.quantum_info.Statevector` of *circuit*.

    Raises ``ImportError`` when the base ``qiskit`` package is missing.
    """
    try:
        from qiskit.quantum_info import Statevector
    except ImportError as exc:
        raise ImportError(
            "statevector backend requires 'qiskit'; install with "
            "'python -m pip install .[qiskit]'"
        ) from exc

    return Statevector.from_instruction(circuit)


class _StatevectorSim:
    """Exact simulation via ``qiskit.quantum_info.Statevector``.

    ``seed`` is accepted for a uniform factory signature and ignored;
    exact state-vector evolution is deterministic.
    """

    def __init__(self, seed):
        self._seed = seed

    def __call__(self, circuit):
        sv = _circuit_statevector(circuit)
        return sv.data  # numpy complex128 array, shape (2**N,)


class _AerSim:
    """Simulation via ``qiskit_aer.AerSimulator``."""

    def __init__(self, shots, noise_model, seed):
        try:
            from qiskit_aer import AerSimulator
        except ImportError as exc:
            raise ImportError(
                "aer backend requires 'qiskit-aer'; install with "
                "'python -m pip install qiskit-aer'"
            ) from exc

        if noise_model is not None and shots is None:
            raise ValueError(
                "noise_model requires a shot-based run: pass shots > 0"
            )

        self._seed = seed
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
            tc = transpile(c, self._sim, seed_transpiler=self._seed)
            job = self._sim.run(
                tc,
                shots=1,
                seed_simulator=self._seed,
                timeout=_JOB_TIMEOUT_SECONDS,
            )
            result = job.result()
            if not result.success:
                raise RuntimeError(
                    "Aer statevector job failed (timeout=%ss)"
                    % _JOB_TIMEOUT_SECONDS
                )
            sv_data = np.asarray(result.get_statevector(tc), dtype=np.complex128)
            return sv_data
        else:
            # Shot-based: transpile, measure all, run
            c = circuit.copy()
            c.measure_all()
            tc = transpile(c, self._sim, seed_transpiler=self._seed)
            job = self._sim.run(
                tc,
                shots=self._shots,
                seed_simulator=self._seed,
                timeout=_JOB_TIMEOUT_SECONDS,
            )
            result = job.result()
            if not result.success:
                raise RuntimeError(
                    "Aer counts job failed (timeout=%ss)" % _JOB_TIMEOUT_SECONDS
                )
            counts = result.get_counts()
            return _counts_to_statevector_proxy(counts, circuit.num_qubits, self._shots)


class _FakeBackendSim:
    """Noisy simulation using an IBM Fake Provider backend.

    When ``mitigate_readout`` is set, measured counts are corrected for
    readout errors via local confusion-matrix calibration before
    observable extraction.  Calibration circuits run once per instance
    and the per-physical-qubit bit probabilities are cached; the
    confusion matrix is ordered by the transpiled virtual→physical
    qubit layout of each main circuit, and standard errors are computed
    from the raw pre-mitigation counts.
    """

    def __init__(self, fake_backend_name, shots, systemsize, seed,
                 mitigate_readout=True):
        try:
            from qiskit_aer import AerSimulator
        except ImportError as exc:
            raise ImportError(
                "fake_backend requires 'qiskit-aer'"
            ) from exc

        fake_backend = _load_fake_backend(fake_backend_name)
        if systemsize > fake_backend.num_qubits:
            raise ValueError(
                "fake_backend %s provides %d qubits; systemsize %d exceeds it"
                % (fake_backend_name, fake_backend.num_qubits, systemsize)
            )
        self._sim = AerSimulator.from_backend(fake_backend)
        self._shots = shots
        self._seed = seed
        self._mitigate_readout = mitigate_readout
        self._n_cal_qubits = fake_backend.num_qubits
        self._cal_probs = None

    def _calibration_probs(self):
        """Return cached ``(p0_given_0, p1_given_1)`` per physical qubit.

        Runs the all-zeros/all-ones calibration circuits on the first
        call and caches the per-physical-qubit bit probabilities for
        reuse across time points (calibration clbit index == physical
        qubit index).
        """
        if self._cal_probs is None:
            self._cal_probs = _readout_calibration_probs(
                self._sim, self._n_cal_qubits, self._shots, self._seed,
            )
        return self._cal_probs

    def __call__(self, circuit):
        from qiskit import transpile

        c = circuit.copy()
        c.measure_all()
        tc = transpile(c, self._sim, seed_transpiler=self._seed)
        job = self._sim.run(
            tc,
            shots=self._shots,
            seed_simulator=self._seed,
            timeout=_JOB_TIMEOUT_SECONDS,
        )
        result = job.result()
        if not result.success:
            raise RuntimeError(
                "Fake-backend counts job failed (timeout=%ss)"
                % _JOB_TIMEOUT_SECONDS
            )
        counts = result.get_counts()
        if self._mitigate_readout:
            p0_given_0, p1_given_1 = self._calibration_probs()
            mitigated = _mitigate_counts(
                counts, p0_given_0, p1_given_1, self._shots,
                _final_index_layout(tc),
            )
            return _ShotProxy(
                mitigated, circuit.num_qubits, self._shots,
                stderr_counts=counts,
            )
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


def _readout_calibration_probs(backend, num_qubits, shots, seed):
    """Run readout calibration and return per-physical-qubit probabilities.

    Prepares all-zeros and all-ones *num_qubits*-qubit calibration
    circuits on *backend* (same ``shots``, seeded like the main jobs)
    and returns ``(p0_given_0, p1_given_1)`` where index ``q`` holds the
    probability for physical qubit ``q``.  The calibration circuits have
    trivial layout, so calibration clbit index == physical qubit index.
    """
    from qiskit import QuantumCircuit, transpile

    p0_given_0 = None
    p1_given_1 = None
    for prep_ones in (False, True):
        cal = QuantumCircuit(num_qubits, num_qubits)
        if prep_ones:
            cal.x(range(num_qubits))
        cal.measure(range(num_qubits), range(num_qubits))
        tc = transpile(cal, backend, seed_transpiler=seed)
        job = backend.run(
            tc,
            shots=shots,
            seed_simulator=seed,
            timeout=_JOB_TIMEOUT_SECONDS,
        )
        result = job.result()
        if not result.success:
            raise RuntimeError(
                "Readout calibration job failed (timeout=%ss)"
                % _JOB_TIMEOUT_SECONDS
            )
        cal_counts = result.get_counts()
        if prep_ones:
            p1_given_1 = _per_qubit_bit_probabilities(
                cal_counts, num_qubits, 1
            )
        else:
            p0_given_0 = _per_qubit_bit_probabilities(
                cal_counts, num_qubits, 0
            )
    return p0_given_0, p1_given_1


def _final_index_layout(tc):
    """Return the virtual→physical qubit list of transpiled *tc*.

    ``L[q]`` is the physical qubit holding virtual qubit ``q`` at the
    end of the transpiled circuit.  Returns ``None`` when the layout is
    unavailable (callers fall back to the identity mapping).
    """
    try:
        if tc.layout is None:
            return None
        return list(tc.layout.final_index_layout())
    except Exception:
        return None


def _mitigate_counts(counts, p0_given_0, p1_given_1, shots, layout):
    """Return readout-mitigated pseudo-counts for measured *counts*.

    Builds the tensor-product readout confusion matrix from the cached
    per-physical-qubit calibration probabilities *p0_given_0* /
    *p1_given_1* and inverts it to map the measured probability vector
    onto corrected probabilities.  *layout* maps each classical bit
    ``q`` of *counts* to the physical qubit ``layout[q]`` whose readout
    error applies (``None`` → identity mapping).  Negative values are
    clipped to zero and the result is renormalised; pseudo-counts are
    returned scaled back to *shots*.  A singular confusion matrix falls
    back to the raw (unmitigated) counts with a warning.
    """
    num_qubits = max(len(key.replace(" ", "")) for key in counts)
    if layout is not None and len(layout) < num_qubits:
        layout = None

    confusion = np.array([[1.0]])
    for q in reversed(range(num_qubits)):
        phys = layout[q] if layout is not None else q
        a_q = np.array([
            [p0_given_0[phys], 1.0 - p1_given_1[phys]],
            [1.0 - p0_given_0[phys], p1_given_1[phys]],
        ])
        confusion = np.kron(confusion, a_q)

    measured = np.zeros(1 << num_qubits, dtype=float)
    total_shots = sum(counts.values())
    for key, count in counts.items():
        bits_reversed = key.replace(" ", "")[::-1]
        index = 0
        for q, char in enumerate(bits_reversed):
            if char == "1":
                index |= 1 << q
        measured[index] += count
    measured /= total_shots

    try:
        mitigated = np.linalg.solve(confusion, measured)
    except np.linalg.LinAlgError:
        logger.warning(
            "readout confusion matrix is singular; returning raw counts"
        )
        return {
            key.replace(" ", ""): float(count)
            for key, count in counts.items()
        }
    mitigated = np.clip(mitigated, 0.0, None)
    total = float(mitigated.sum())
    if total <= 0.0:
        return {
            key.replace(" ", ""): float(count)
            for key, count in counts.items()
        }
    mitigated /= total

    bit_format = "0%db" % num_qubits
    return {
        format(index, bit_format): float(prob * shots)
        for index, prob in enumerate(mitigated)
        if prob > 0.0
    }


def _per_qubit_bit_probabilities(counts, num_qubits, bit_value):
    """Return P(measure *bit_value*) per classical bit from calibration counts."""
    totals = np.zeros(num_qubits, dtype=float)
    for key, count in counts.items():
        bits_reversed = key.replace(" ", "")[::-1]
        for q in range(num_qubits):
            b = int(bits_reversed[q]) if q < len(bits_reversed) else 0
            if b == bit_value:
                totals[q] += count
    return totals / sum(counts.values())


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
    Model site i maps to qubit ``systemsize − 1 − i`` (little-endian),
    so bit ``systemsize − 1 − i`` of the basis index sets ⟨Z_i⟩.
    """
    if isinstance(sv_data, _ShotProxy):
        return sv_data.magnetization_z(systemsize)

    # Exact: probabilities from statevector amplitudes
    probs = np.abs(sv_data) ** 2  # shape (2**N,)
    indices = np.arange(len(probs), dtype=np.int64)
    mag = np.empty(systemsize, dtype=float)
    for i in range(systemsize):
        # model site i ↔ qubit N−1−i; bit set → eigenvalue −1, clear → +1
        bits_i = ((indices >> (systemsize - 1 - i)) & 1).astype(float)
        eigenvalues_i = 1.0 - 2.0 * bits_i  # +1 or −1
        mag[i] = float(np.dot(probs, eigenvalues_i))
    return mag


# ---------------------------------------------------------------------------
# Shot-based helper
# ---------------------------------------------------------------------------

class _ShotProxy:
    """Lightweight wrapper around Qiskit ``counts`` for expectation values.

    ``stderr_counts`` optionally supplies raw (pre-mitigation) counts
    used by the binomial standard-error methods; point-value methods
    always read ``counts``.
    """

    def __init__(self, counts: dict, num_qubits: int, total_shots: int,
                 stderr_counts: dict | None = None):
        self._counts = {
            key.replace(" ", ""): value for key, value in counts.items()
        }
        self._num_qubits = num_qubits
        self._total_shots = total_shots
        if stderr_counts is None:
            self._stderr_counts = None
        else:
            self._stderr_counts = {
                key.replace(" ", ""): value
                for key, value in stderr_counts.items()
            }

    def prob(self, bitstring: str) -> float:
        """Empirical probability of *bitstring* (e.g. '1111')."""
        bits = bitstring.replace(" ", "")
        return self._counts.get(bits, 0) / self._total_shots

    def magnetization_z(self, systemsize: int) -> np.ndarray:
        """Site-resolved ⟨Z_i⟩ from shot counts.

        Qiskit returns bitstrings in big-endian order for the *printed*
        counts key (leftmost character = highest-index qubit).  After
        reversing, model site i lives at index ``systemsize − 1 − i``.
        """
        mag = np.zeros(systemsize, dtype=float)
        for bitstring, count in self._counts.items():
            # Qiskit count keys may contain spaces; strip them
            bits = bitstring.replace(" ", "")
            # Reverse: bits[0] is the highest-index qubit in Qiskit
            bits_reversed = bits[::-1]
            for i in range(systemsize):
                bit_index = systemsize - 1 - i
                b = (
                    int(bits_reversed[bit_index])
                    if bit_index < len(bits_reversed)
                    else 0
                )
                mag[i] += (1.0 - 2.0 * b) * count
        return mag / self._total_shots

    def prob_stderr(self, bitstring: str) -> float:
        """Binomial standard error of :meth:`prob` for *bitstring*.

        Uses the raw ``stderr_counts`` when supplied, so the error is
        the pre-mitigation sampling error even for mitigated proxies.
        """
        counts = (
            self._stderr_counts
            if self._stderr_counts is not None
            else self._counts
        )
        bits = bitstring.replace(" ", "")
        p = counts.get(bits, 0) / self._total_shots
        return float(np.sqrt(p * (1.0 - p) / self._total_shots))

    def magnetization_z_stderr(self, systemsize: int) -> np.ndarray:
        """Standard error of :meth:`magnetization_z` from shot counts.

        Each site's ⟨Z_i⟩ estimator averages ±1 outcomes; with ``p`` the
        raw-counts empirical probability of measuring |1⟩ at a site the
        standard error is ``2 sqrt(p (1 − p) / shots)``.
        """
        p1 = np.zeros(systemsize, dtype=float)
        counts = (
            self._stderr_counts
            if self._stderr_counts is not None
            else self._counts
        )
        for bitstring, count in counts.items():
            bits = bitstring.replace(" ", "")
            bits_reversed = bits[::-1]
            for i in range(systemsize):
                bit_index = systemsize - 1 - i
                b = (
                    int(bits_reversed[bit_index])
                    if bit_index < len(bits_reversed)
                    else 0
                )
                if b == 1:
                    p1[i] += count
        p1 /= self._total_shots
        return 2.0 * np.sqrt(p1 * (1.0 - p1) / self._total_shots)


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
