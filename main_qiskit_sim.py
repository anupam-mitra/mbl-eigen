import logging

import numpy as np

from mbl_eigen.cli import build_qiskit_sim_parser
from mbl_eigen.mbl_model import build_mbl_model
from mbl_eigen.output_names import mbl_qiskit_plot_name
from mbl_eigen.plotting import plot_magnetization_z, plot_return_rate
from mbl_eigen.qiskit_simulation import run_mbl_qiskit_simulation


def main():
    argument_parser = build_qiskit_sim_parser()
    args = argument_parser.parse_args()

    model = build_mbl_model(
        systemsize=args.systemsize,
        jIntMean=args.jIntMean,
        jIntStd=args.jIntStd,
        bFieldMean=args.bFieldMean,
        bFieldStd=args.bFieldStd,
        anglePolarPiMin=args.anglePolarPiMin,
        anglePolarPiMax=args.anglePolarPiMax,
    )

    logging.info("bField_samples = %s", model.bField_samples)
    logging.info("theta_samples = %s", model.theta_samples)
    logging.info("jInt_samples = %s", model.jInt_samples)

    # Time grid for dynamics
    times_array = np.arange(0.0, 10.0625, 0.0625)

    result = run_mbl_qiskit_simulation(
        model=model,
        times_array=times_array,
        trotter_steps=args.trotterSteps,
        trotter_order=args.trotterOrder,
        backend=args.simBackend,
        shots=args.shots,
        fake_backend_name=args.fakeBackend,
    )

    # Output filenames
    ret_rate_file = mbl_qiskit_plot_name(
        systemsize=args.systemsize,
        anglePolarPiMin=args.anglePolarPiMin,
        anglePolarPiMax=args.anglePolarPiMax,
        jIntMean=args.jIntMean,
        jIntStd=args.jIntStd,
        bFieldMean=args.bFieldMean,
        bFieldStd=args.bFieldStd,
        backend=args.simBackend,
        trotter_steps=args.trotterSteps,
    ).replace("mbl_qiskit_sim_", "mbl_qiskit_ret_")

    mag_file = mbl_qiskit_plot_name(
        systemsize=args.systemsize,
        anglePolarPiMin=args.anglePolarPiMin,
        anglePolarPiMax=args.anglePolarPiMax,
        jIntMean=args.jIntMean,
        jIntStd=args.jIntStd,
        bFieldMean=args.bFieldMean,
        bFieldStd=args.bFieldStd,
        backend=args.simBackend,
        trotter_steps=args.trotterSteps,
    ).replace("mbl_qiskit_sim_", "mbl_qiskit_mag_")

    plot_return_rate(
        times=result.times,
        amplitudes=np.sqrt(result.return_rate),  # plot_return_rate expects amplitudes A(t) where |A(t)|^2 is the return probability
        systemsize=args.systemsize,
        filename=ret_rate_file,
    )
    logging.info("Saved return rate plot to %s", ret_rate_file)

    plot_magnetization_z(
        times=result.times,
        magnetization_z=result.magnetization_z,
        filename=mag_file,
    )
    logging.info("Saved magnetization plot to %s", mag_file)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
