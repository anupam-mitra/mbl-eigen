import argparse

from .eigensolver import EIGENSOLVER_DEVICE_CHOICES
from .eigensolver import GENERAL_EIGEN_BACKENDS
from .eigensolver import HERMITIAN_EIGEN_BACKENDS


COMMON_DESCRIPTION = "Plots eigenphases of the time evolution operator under a PXP like Hamiltonian"


def _integer_at_least(minimum):
    def parse(value):
        try:
            parsed = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "expected an integer"
            ) from exc

        if parsed < minimum:
            raise argparse.ArgumentTypeError(
                "expected an integer >= %d" % minimum
            )

        return parsed

    return parse


def build_qmbs_parser():
    argument_parser = argparse.ArgumentParser(
        prog="main_qmbs.py",
        description=COMMON_DESCRIPTION,
        epilog="",
    )

    argument_parser.add_argument(
        "--systemsize", type=_integer_at_least(2), required=True)
    argument_parser.add_argument("--tduration", type=float, required=True)
    argument_parser.add_argument("--Delta", type=float, required=True)
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

    argument_parser.add_argument(
        "--systemsize", type=_integer_at_least(1), required=True)
    argument_parser.add_argument("--thetaXPi", type=float, required=True)
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
    argument_parser.add_argument("--seed", type=_integer_at_least(0))

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

    argument_parser.add_argument(
        "--systemsize", type=_integer_at_least(1), required=True)
    argument_parser.add_argument("--tduration", type=float, required=True)
    argument_parser.add_argument("--jIntMean", type=float, required=True)
    argument_parser.add_argument("--bFieldMean", type=float, required=True)
    argument_parser.add_argument("--jIntStd", type=float, required=True)
    argument_parser.add_argument("--bFieldStd", type=float, required=True)
    argument_parser.add_argument(
        "--anglePolarPiMin", type=float, required=True)
    argument_parser.add_argument(
        "--anglePolarPiMax", type=float, required=True)
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
    argument_parser.add_argument("--seed", type=_integer_at_least(0))

    return argument_parser
