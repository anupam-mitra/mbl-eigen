"""Qiskit circuit builders for hardware-suitable propagators."""

import numpy as np


def sample_mbldtc_angles(systemsize, rng=None):
    """Sample the random angles used by the MBL-DTC Floquet circuit."""
    _validate_systemsize(systemsize)

    if rng is None:
        rng = np.random

    if not hasattr(rng, "random"):
        raise TypeError("rng must provide a random(size=...) method")

    phi_z = np.asarray(rng.random(systemsize), dtype=float) * np.pi
    phi_zz = np.asarray(rng.random(systemsize - 1), dtype=float) * np.pi
    _validate_finite_array(phi_z, "phi_z")
    _validate_finite_array(phi_zz, "phi_zz")
    return phi_z, phi_zz


def build_mbldtc_floquet_circuit(
        systemsize,
        theta_x,
        phi_z,
        phi_zz,
        cycles=1,
        insert_barriers=False):
    """Build the exact MBL-DTC Floquet propagator as a Qiskit circuit.

    The circuit implements the same gate order as ``mbl_eigen.mbldtc_app``:
    a global X rotation layer, then on-site Z rotations, then nearest-neighbor
    ZZ interactions. This is already gate-native and does not require
    Trotterization.
    """
    _validate_systemsize(systemsize)
    _validate_positive_integer(cycles, "cycles")

    phi_z = _as_real_vector(phi_z, systemsize, "phi_z")
    phi_zz = _as_real_vector(phi_zz, systemsize - 1, "phi_zz")
    theta_x = _as_finite_real(theta_x, "theta_x")
    QuantumCircuit = _require_quantum_circuit()

    circuit = QuantumCircuit(systemsize, name="mbldtc_floquet")

    for ix_cycle in range(cycles):
        for ix_site in range(systemsize):
            circuit.rx(theta_x, _qiskit_qubit(systemsize, ix_site))

        if insert_barriers:
            circuit.barrier()

        for ix_site in range(systemsize):
            circuit.rz(
                phi_z[ix_site],
                _qiskit_qubit(systemsize, ix_site),
            )

        if insert_barriers and systemsize > 1:
            circuit.barrier()

        for ix_site in range(systemsize - 1):
            circuit.rzz(
                phi_zz[ix_site],
                _qiskit_qubit(systemsize, ix_site),
                _qiskit_qubit(systemsize, ix_site + 1),
            )

        if insert_barriers and ix_cycle != cycles - 1:
            circuit.barrier()

    return circuit


def build_mbl_trotter_step_circuit(
        systemsize,
        jInt_samples,
        bField_samples,
        theta_samples,
        time_step,
        trotter_order=2,
        insert_barriers=False):
    """Build one Trotter step for the random-field MBL Hamiltonian.

    The Hamiltonian matches ``mbl_eigen.mbl_model.build_mbl_hamiltonian(...)``:

    H = sum_i hx[i] X_i + sum_i hz[i] Z_i + sum_i J[i] Z_i Z_{i+1}

    The diagonal Z/ZZ sector is implemented with ``rz`` and ``rzz`` gates, and
    the transverse X sector is implemented with ``rx`` gates.

    Notes
    -----
    ``time_step`` is the per-step evolution time ``dt``. First-order Suzuki
    (``trotter_order=1``) has local error O(dt**2) and global error O(t * dt);
    second-order (``trotter_order=2``) has local error O(dt**3) and global
    error O(t * dt**2).
    """
    _validate_systemsize(systemsize)
    _validate_trotter_order(trotter_order)

    jInt_samples = _as_real_vector(jInt_samples, systemsize - 1, "jInt_samples")
    bField_samples = _as_real_vector(bField_samples, systemsize, "bField_samples")
    theta_samples = _as_real_vector(theta_samples, systemsize, "theta_samples")
    time_step = _as_finite_real(time_step, "time_step")

    hx_terms = bField_samples * np.sin(theta_samples)
    hz_terms = bField_samples * np.cos(theta_samples)
    _validate_finite_array(hx_terms, "transverse field terms")
    _validate_finite_array(hz_terms, "longitudinal field terms")
    QuantumCircuit = _require_quantum_circuit()

    circuit = QuantumCircuit(systemsize, name="mbl_trotter_step")

    if trotter_order == 1:
        _append_mbl_diagonal_layer(circuit, hz_terms, jInt_samples, time_step)
        if insert_barriers:
            circuit.barrier()
        _append_mbl_x_layer(circuit, hx_terms, time_step)
        return circuit

    _append_mbl_diagonal_layer(circuit, hz_terms, jInt_samples, time_step * 0.5)
    if insert_barriers:
        circuit.barrier()
    _append_mbl_x_layer(circuit, hx_terms, time_step)
    if insert_barriers:
        circuit.barrier()
    _append_mbl_diagonal_layer(circuit, hz_terms, jInt_samples, time_step * 0.5)
    return circuit


def build_mbl_trotter_circuit(
        systemsize,
        jInt_samples,
        bField_samples,
        theta_samples,
        time,
        n_steps=1,
        trotter_order=2,
        insert_barriers=False):
    """Build a Trotterized Qiskit circuit for the MBL time-evolution operator.

    ``n_steps`` is the total number of Trotter steps dividing ``time``:
    ``time_step = time / n_steps``. With ``dt = time / n_steps``, first-order
    Suzuki has local error O(dt**2) and global error O(t * dt); second-order
    has local error O(dt**3) and global error O(t * dt**2).
    """
    _validate_systemsize(systemsize)
    _validate_positive_integer(n_steps, "n_steps")
    _validate_trotter_order(trotter_order)
    time = _as_finite_real(time, "time")
    jInt_samples = _as_real_vector(jInt_samples, systemsize - 1, "jInt_samples")
    bField_samples = _as_real_vector(bField_samples, systemsize, "bField_samples")
    theta_samples = _as_real_vector(theta_samples, systemsize, "theta_samples")

    QuantumCircuit = _require_quantum_circuit()
    circuit = QuantumCircuit(systemsize, name="mbl_time_evolution")
    time_step = time / n_steps

    for ix_step in range(n_steps):
        step_circuit = build_mbl_trotter_step_circuit(
            systemsize=systemsize,
            jInt_samples=jInt_samples,
            bField_samples=bField_samples,
            theta_samples=theta_samples,
            time_step=time_step,
            trotter_order=trotter_order,
            insert_barriers=insert_barriers,
        )
        circuit.compose(step_circuit, inplace=True)
        if insert_barriers and ix_step != n_steps - 1:
            circuit.barrier()

    return circuit


def build_mbl_trotter_circuit_from_model(
        model,
        time,
        n_steps=1,
        trotter_order=2,
        insert_barriers=False):
    """Build a Trotterized MBL circuit directly from ``MBLModel`` samples.

    ``n_steps`` is the total number of Trotter steps dividing ``time``:
    ``time_step = time / n_steps``.
    """
    systemsize = len(model.bField_samples)
    return build_mbl_trotter_circuit(
        systemsize=systemsize,
        jInt_samples=model.jInt_samples,
        bField_samples=model.bField_samples,
        theta_samples=model.theta_samples,
        time=time,
        n_steps=n_steps,
        trotter_order=trotter_order,
        insert_barriers=insert_barriers,
    )


def _append_mbl_diagonal_layer(circuit, hz_terms, jInt_samples, time_step):
    systemsize = circuit.num_qubits
    rz_angles = 2.0 * hz_terms * time_step
    rzz_angles = 2.0 * jInt_samples * time_step
    _validate_finite_array(rz_angles, "rz angles")
    _validate_finite_array(rzz_angles, "rzz angles")

    for ix_site, angle in enumerate(rz_angles):
        circuit.rz(
            angle,
            _qiskit_qubit(systemsize, ix_site),
        )

    for ix_site, angle in enumerate(rzz_angles):
        circuit.rzz(
            angle,
            _qiskit_qubit(systemsize, ix_site),
            _qiskit_qubit(systemsize, ix_site + 1),
        )


def _append_mbl_x_layer(circuit, hx_terms, time_step):
    systemsize = circuit.num_qubits
    rx_angles = 2.0 * hx_terms * time_step
    _validate_finite_array(rx_angles, "rx angles")

    for ix_site, angle in enumerate(rx_angles):
        circuit.rx(
            angle,
            _qiskit_qubit(systemsize, ix_site),
        )


def _qiskit_qubit(systemsize, site):
    """Map QuTiP tensor site order to Qiskit's little-endian qubit order."""
    return systemsize - 1 - site


def _require_quantum_circuit():
    try:
        from qiskit import QuantumCircuit
    except ImportError as exc:
        raise ImportError(
            "Qiskit propagators require the optional 'qiskit' extra; install with 'python3 -m pip install .[qiskit]'"
        ) from exc

    return QuantumCircuit


def _validate_systemsize(systemsize):
    _validate_positive_integer(systemsize, "systemsize")


def _validate_positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError("%s must be a positive integer" % name)


def _validate_trotter_order(trotter_order):
    if trotter_order not in (1, 2):
        raise ValueError("trotter_order must be 1 or 2")


def _as_real_vector(values, expected_length, name):
    array = np.asarray(values)
    if array.shape != (expected_length,):
        raise ValueError(
            "%s must have shape (%d,), got %s"
            % (name, expected_length, array.shape)
        )
    if np.iscomplexobj(array) or any(np.iscomplexobj(value) for value in array.flat):
        raise ValueError("%s must contain real values" % name)

    try:
        array = np.asarray(array, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must contain real numeric values" % name) from exc

    _validate_finite_array(array, name)
    return array


def _as_finite_real(value, name):
    array = np.asarray(value)
    if array.ndim != 0 or np.iscomplexobj(array):
        raise ValueError("%s must be a finite real scalar" % name)

    try:
        value = float(array)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be a finite real scalar" % name) from exc

    if not np.isfinite(value):
        raise ValueError("%s must be a finite real scalar" % name)
    return value


def _validate_finite_array(array, name):
    if not np.isfinite(array).all():
        raise ValueError("%s must contain only finite values" % name)


__all__ = [
    "build_mbldtc_floquet_circuit",
    "build_mbl_trotter_circuit",
    "build_mbl_trotter_circuit_from_model",
    "build_mbl_trotter_step_circuit",
    "sample_mbldtc_angles",
]
