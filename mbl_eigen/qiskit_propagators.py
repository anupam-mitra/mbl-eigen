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
    QuantumCircuit = _require_quantum_circuit()
    _validate_systemsize(systemsize)
    _validate_positive_integer(cycles, "cycles")

    phi_z = _as_real_vector(phi_z, systemsize, "phi_z")
    phi_zz = _as_real_vector(phi_zz, systemsize - 1, "phi_zz")
    theta_x = float(theta_x)

    circuit = QuantumCircuit(systemsize, name="mbldtc_floquet")

    for ix_cycle in range(cycles):
        for ix_site in range(systemsize):
            circuit.rx(theta_x, ix_site)

        if insert_barriers:
            circuit.barrier()

        for ix_site in range(systemsize):
            circuit.rz(phi_z[ix_site], ix_site)

        if insert_barriers and systemsize > 1:
            circuit.barrier()

        for ix_site in range(systemsize - 1):
            circuit.rzz(phi_zz[ix_site], ix_site, ix_site + 1)

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
    """
    QuantumCircuit = _require_quantum_circuit()
    _validate_systemsize(systemsize)
    _validate_trotter_order(trotter_order)

    jInt_samples = _as_real_vector(jInt_samples, systemsize - 1, "jInt_samples")
    bField_samples = _as_real_vector(bField_samples, systemsize, "bField_samples")
    theta_samples = _as_real_vector(theta_samples, systemsize, "theta_samples")
    time_step = float(time_step)

    hx_terms = bField_samples * np.sin(theta_samples)
    hz_terms = bField_samples * np.cos(theta_samples)

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
        trotter_steps=1,
        trotter_order=2,
        insert_barriers=False):
    """Build a Trotterized Qiskit circuit for the MBL time-evolution operator."""
    QuantumCircuit = _require_quantum_circuit()
    _validate_systemsize(systemsize)
    _validate_positive_integer(trotter_steps, "trotter_steps")

    circuit = QuantumCircuit(systemsize, name="mbl_time_evolution")
    time_step = float(time) / trotter_steps

    for ix_step in range(trotter_steps):
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
        if insert_barriers and ix_step != trotter_steps - 1:
            circuit.barrier()

    return circuit


def build_mbl_trotter_circuit_from_model(
        model,
        time,
        trotter_steps=1,
        trotter_order=2,
        insert_barriers=False):
    """Build a Trotterized MBL circuit directly from ``MBLModel`` samples."""
    systemsize = len(model.bField_samples)
    return build_mbl_trotter_circuit(
        systemsize=systemsize,
        jInt_samples=model.jInt_samples,
        bField_samples=model.bField_samples,
        theta_samples=model.theta_samples,
        time=time,
        trotter_steps=trotter_steps,
        trotter_order=trotter_order,
        insert_barriers=insert_barriers,
    )


def _append_mbl_diagonal_layer(circuit, hz_terms, jInt_samples, time_step):
    for ix_site, hz_term in enumerate(hz_terms):
        circuit.rz(2.0 * hz_term * time_step, ix_site)

    for ix_site, coupling in enumerate(jInt_samples):
        circuit.rzz(2.0 * coupling * time_step, ix_site, ix_site + 1)


def _append_mbl_x_layer(circuit, hx_terms, time_step):
    for ix_site, hx_term in enumerate(hx_terms):
        circuit.rx(2.0 * hx_term * time_step, ix_site)


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
    if int(value) != value or int(value) <= 0:
        raise ValueError("%s must be a positive integer" % name)


def _validate_trotter_order(trotter_order):
    if trotter_order not in (1, 2):
        raise ValueError("trotter_order must be 1 or 2")


def _as_real_vector(values, expected_length, name):
    array = np.asarray(values, dtype=float)
    if array.shape != (expected_length,):
        raise ValueError(
            "%s must have shape (%d,), got %s"
            % (name, expected_length, array.shape)
        )
    return array


__all__ = [
    "build_mbldtc_floquet_circuit",
    "build_mbl_trotter_circuit",
    "build_mbl_trotter_circuit_from_model",
    "build_mbl_trotter_step_circuit",
    "sample_mbldtc_angles",
]
