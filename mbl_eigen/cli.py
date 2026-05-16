import argparse

from .eigensolver import EIGENSOLVER_DEVICE_CHOICES
from .eigensolver import GENERAL_EIGEN_BACKENDS
from .eigensolver import HERMITIAN_EIGEN_BACKENDS


COMMON_DESCRIPTION = "Plots eigenphases of the time evolution operator under a PXP like Hamiltonian"


def build_qmbs_parser():
    argument_parser = argparse.ArgumentParser(
        prog="main_qmbs.py",
        description=COMMON_DESCRIPTION,
        epilog="",
    )

    argument_parser.add_argument("--systemsize", type=int)
    argument_parser.add_argument("--tduration", type=float)
    argument_parser.add_argument("--Delta", type=float)
    argument_parser.add_argument(
        "--eigenBackend",
        choices=HERMITIAN_EIGEN_BACKENDS,
        default="qobj",
    )
    argument_parser.add_argument(
        "--eigenDevice",
        choices=EIGENSOLVER_DEVICE_CHOICES,
        default="auto",
    )

    return argument_parser


def build_mbldtc_parser():
    argument_parser = argparse.ArgumentParser(
        prog="main_mbldtc.py",
        description=COMMON_DESCRIPTION,
        epilog="",
    )

    argument_parser.add_argument("--systemsize", type=int)
    argument_parser.add_argument("--thetaXPi", type=float)
    argument_parser.add_argument(
        "--eigenBackend",
        choices=GENERAL_EIGEN_BACKENDS,
        default="qobj",
    )
    argument_parser.add_argument(
        "--eigenDevice",
        choices=EIGENSOLVER_DEVICE_CHOICES,
        default="auto",
    )

    return argument_parser


def build_mbl_parser(
        prog="main_mbl.py",
        description=COMMON_DESCRIPTION,
        default_eigen_backend="qobj"):
    argument_parser = argparse.ArgumentParser(
        prog=prog,
        description=description,
        epilog="",
    )

    argument_parser.add_argument("--systemsize", type=int)
    argument_parser.add_argument("--tduration", type=float)
    argument_parser.add_argument("--jIntMean", type=float)
    argument_parser.add_argument("--bFieldMean", type=float)
    argument_parser.add_argument("--jIntStd", type=float)
    argument_parser.add_argument("--bFieldStd", type=float)
    argument_parser.add_argument("--anglePolarPiMin", type=float)
    argument_parser.add_argument("--anglePolarPiMax", type=float)
    argument_parser.add_argument(
        "--eigenBackend",
        choices=HERMITIAN_EIGEN_BACKENDS,
        default=default_eigen_backend,
    )
    argument_parser.add_argument(
        "--eigenDevice",
        choices=EIGENSOLVER_DEVICE_CHOICES,
        default="auto",
    )

    return argument_parser


def build_qiskit_sim_parser():
    from .qiskit_simulation import QISKIT_SIM_BACKENDS, DEFAULT_FAKE_BACKEND_NAME

    argument_parser = build_mbl_parser(
        prog="main_qiskit_sim.py",
        description="MBL time evolution via Qiskit simulation backends.",
    )

    argument_parser.add_argument(
        "--simBackend",
        choices=QISKIT_SIM_BACKENDS,
        default="statevector",
    )
    argument_parser.add_argument("--shots", type=int, default=None)
    argument_parser.add_argument(
        "--fakeBackend", type=str, default=DEFAULT_FAKE_BACKEND_NAME
    )
    argument_parser.add_argument("--trotterSteps", type=int, default=10)
    argument_parser.add_argument("--trotterOrder", type=int, choices=[1, 2], default=2)

    return argument_parser
