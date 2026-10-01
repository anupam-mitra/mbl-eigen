import logging

import numpy as np

from .mbl_app import _mbl_model_title, _rng_from_args, time_grid
from .mbl_model import build_mbl_model
from .output_names import mbl_qiskit_plot_name
from .plotting import plot_magnetization_z, plot_return_rate
from .qiskit_simulation import run_mbl_qiskit_simulation


logger = logging.getLogger(__name__)


def run_mbl_qiskit(args):
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
        trotter_steps=args.trotterSteps,
        trotter_order=args.trotterOrder,
        backend=args.simBackend,
        shots=args.shots,
        fake_backend_name=args.fakeBackend,
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
