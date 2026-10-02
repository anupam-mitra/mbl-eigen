import logging
import os
import uuid

import numpy as np
import pandas as pd

from .mbl_app import _mbl_model_title, _rng_from_args, time_grid
from .mbl_model import build_mbl_model
from .output_names import mbl_qiskit_plot_name
from .plotting import plot_magnetization_z, plot_return_rate
from .qiskit_propagators import (
    build_mbldtc_floquet_circuit,
    sample_mbldtc_angles,
)
from .qiskit_simulation import run_mbl_qiskit_simulation


logger = logging.getLogger(__name__)


def run_mbl_qiskit(args):
    """Run the MBL Qiskit simulation workflow and save plots.

    Raw observables are persisted to a CSV file next to the
    return-rate plot.  ``--seed`` seeds both the disorder
    realisation and the simulator.
    """
    model = build_mbl_model(
        systemsize=args.systemsize,
        jIntMean=args.jIntMean,
        jIntStd=args.jIntStd,
        bFieldMean=args.bFieldMean,
        bFieldStd=args.bFieldStd,
        anglePolarPiMin=args.anglePolarPiMin,
        anglePolarPiMax=args.anglePolarPiMax,
        rng=_rng_from_args(args, None),
    )

    logger.info("bField_samples = %s", model.bField_samples)
    logger.info("theta_samples = %s", model.theta_samples)
    logger.info("jInt_samples = %s", model.jInt_samples)

    times_array = time_grid(args.tduration)

    result = run_mbl_qiskit_simulation(
        model=model,
        times_array=times_array,
        seed=args.seed,
        steps_per_unit_time=args.trotterSteps,
        trotter_order=args.trotterOrder,
        backend=args.simBackend,
        shots=args.shots,
        fake_backend_name=args.fakeBackend,
        insert_barriers=args.insertBarriers,
    )

    ret_rate_file = mbl_qiskit_plot_name(
        kind="ret",
        systemsize=args.systemsize,
        anglePolarPiMin=args.anglePolarPiMin,
        anglePolarPiMax=args.anglePolarPiMax,
        jIntMean=args.jIntMean,
        jIntStd=args.jIntStd,
        bFieldMean=args.bFieldMean,
        bFieldStd=args.bFieldStd,
        backend=args.simBackend,
        trotter_steps=args.trotterSteps,
    )

    mag_file = mbl_qiskit_plot_name(
        kind="mag",
        systemsize=args.systemsize,
        anglePolarPiMin=args.anglePolarPiMin,
        anglePolarPiMax=args.anglePolarPiMax,
        jIntMean=args.jIntMean,
        jIntStd=args.jIntStd,
        bFieldMean=args.bFieldMean,
        bFieldStd=args.bFieldStd,
        backend=args.simBackend,
        trotter_steps=args.trotterSteps,
    )

    qiskit_title = "MBL qiskit (%s, tsteps=%s): %s" % (
        args.simBackend,
        args.trotterSteps,
        _mbl_model_title(args),
    )

    plot_return_rate(
        times=result.times,
        amplitudes=np.sqrt(result.return_rate),
        systemsize=args.systemsize,
        filename=ret_rate_file,
        title=qiskit_title,
    )
    logger.info("Saved return rate plot to %s", ret_rate_file)

    plot_magnetization_z(
        times=result.times,
        magnetization_z=result.magnetization_z,
        filename=mag_file,
        title=qiskit_title,
    )
    logger.info("Saved magnetization plot to %s", mag_file)

    ret_rate_csv = os.path.splitext(ret_rate_file)[0] + ".csv"
    csv_dirname = os.path.dirname(ret_rate_csv)
    if csv_dirname:
        os.makedirs(csv_dirname, exist_ok=True)
    csv_data = {
        "time": result.times,
        "return_rate": result.return_rate,
    }
    return_rate_stderr = getattr(result, "return_rate_stderr", None)
    if return_rate_stderr is not None:
        csv_data["return_rate_stderr"] = return_rate_stderr
    for i in range(args.systemsize):
        csv_data["site_%d" % i] = result.magnetization_z[i]
    magnetization_stderr = getattr(result, "magnetization_stderr", None)
    if magnetization_stderr is not None:
        for i in range(args.systemsize):
            csv_data["site_%d_stderr" % i] = magnetization_stderr[i]
    pd.DataFrame(csv_data).to_csv(ret_rate_csv, index=False)
    logger.info("Saved raw observables to %s", ret_rate_csv)


def run_mbldtc_qiskit(args):
    """Simulate MBL-DTC Floquet dynamics via exact statevector and save plots.

    Raw observables are persisted to a CSV file next to the echo plot.
    Exact statevector simulation; no shots are taken.  The state is
    evolved incrementally, one Floquet cycle at a time.  ``tduration``
    is the number of drive periods (Floquet cycles) to simulate.  The
    initial state is |1…1⟩, matching
    :func:`mbl_eigen.mbldtc_app.run_mbldtc`.
    """
    systemsize = args.systemsize
    theta_x = np.pi * args.thetaXPi
    if args.tduration != int(args.tduration):
        raise ValueError(
            "tduration must be an integer number of drive periods "
            "for the DTC workflow"
        )
    n_cycles = int(args.tduration)

    phi_z, phi_zz = sample_mbldtc_angles(
        systemsize, rng=_rng_from_args(args, None)
    )

    logger.info("phi_z = %s", phi_z)
    logger.info("phi_zz = %s", phi_zz)

    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
    except ImportError as exc:
        raise ImportError(
            "statevector simulation requires 'qiskit'; install with "
            "'python -m pip install .[qiskit]'"
        ) from exc

    logger.info(
        "run_mbldtc_qiskit: N=%d, cycles=%d, thetax=%gpi",
        systemsize, n_cycles, args.thetaXPi,
    )

    prep = QuantumCircuit(systemsize, name="state_prep")
    prep.x(range(systemsize))

    one_cycle = build_mbldtc_floquet_circuit(
        systemsize=systemsize,
        theta_x=theta_x,
        phi_z=phi_z,
        phi_zz=phi_zz,
        cycles=1,
    )

    echo = np.empty(n_cycles + 1, dtype=float)
    magnetization_z = np.empty((systemsize, n_cycles + 1), dtype=float)
    all_ones_index = (1 << systemsize) - 1
    indices = np.arange(1 << systemsize, dtype=np.int64)

    state = Statevector.from_instruction(prep)
    for n in range(n_cycles + 1):
        if n > 0:
            state = state.evolve(one_cycle)

        sv = state.data
        probs = np.abs(sv) ** 2

        echo[n] = float(probs[all_ones_index])
        for i in range(systemsize):
            bits_i = ((indices >> (systemsize - 1 - i)) & 1).astype(float)
            eigenvalues_i = 1.0 - 2.0 * bits_i
            magnetization_z[i, n] = float(np.dot(probs, eigenvalues_i))

    cycles = np.arange(n_cycles + 1, dtype=float)

    dtc_title = r"MBL-DTC qiskit: $N=%d$, $\theta_x=%g\pi$" % (
        systemsize, args.thetaXPi,
    )

    echo_file = "mbl_qiskit_dtc_echo_N=%02d_thetax=%gpi_%s.pdf" % (
        systemsize, args.thetaXPi, uuid.uuid4(),
    )
    mag_file = "mbl_qiskit_dtc_mag_N=%02d_thetax=%gpi_%s.pdf" % (
        systemsize, args.thetaXPi, uuid.uuid4(),
    )

    plot_return_rate(
        times=cycles,
        amplitudes=np.sqrt(echo),
        systemsize=systemsize,
        filename=echo_file,
        title=dtc_title,
    )
    logger.info("Saved DTC echo plot to %s", echo_file)

    plot_magnetization_z(
        times=cycles,
        magnetization_z=magnetization_z,
        filename=mag_file,
        title=dtc_title,
    )
    logger.info("Saved DTC magnetization plot to %s", mag_file)

    echo_csv = os.path.splitext(echo_file)[0] + ".csv"
    csv_dirname = os.path.dirname(echo_csv)
    if csv_dirname:
        os.makedirs(csv_dirname, exist_ok=True)
    csv_data = {
        "cycle": cycles,
        "return_rate": echo,
    }
    for i in range(systemsize):
        csv_data["site_%d" % i] = magnetization_z[i]
    pd.DataFrame(csv_data).to_csv(echo_csv, index=False)
    logger.info("Saved raw observables to %s", echo_csv)
