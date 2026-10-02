import argparse
import math

from .eigensolver import EIGENSOLVER_DEVICE_CHOICES
from .eigensolver import GENERAL_EIGEN_BACKENDS
from .eigensolver import HERMITIAN_EIGEN_BACKENDS


QMBS_DESCRIPTION = "PXP/SFIM eigenvalue + eigenphase analysis"
MBLDTC_DESCRIPTION = "MBL-DTC Floquet spectrum analysis"
MBL_DESCRIPTION = "MBL spectrum + eigenvector entropy analysis"


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


def _finite_positive_float(value):
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected a float"
        ) from exc

    if math.isnan(parsed) or math.isinf(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError(
            "expected a finite float > 0"
        )

    return parsed


def _finite_float(value):
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected a float"
        ) from exc

    if math.isnan(parsed) or math.isinf(parsed):
        raise argparse.ArgumentTypeError(
            "expected a finite float"
        )

    return parsed


def _float_at_least(minimum):
    def parse(value):
        try:
            parsed = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "expected a float"
            ) from exc

        if math.isnan(parsed) or math.isinf(parsed) or parsed < minimum:
            raise argparse.ArgumentTypeError(
                "expected a finite float >= %g" % minimum
            )

        return parsed

    return parse


def _integer_bounded(minimum, maximum):
    def parse(value):
        try:
            parsed = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "expected an integer"
            ) from exc

        if parsed < minimum or parsed > maximum:
            raise argparse.ArgumentTypeError(
                "expected an integer between %d and %d "
                "(dense eigensolvers allocate O(2^L) memory; "
                "check available RAM before large L)"
                % (minimum, maximum)
            )

        return parsed

    return parse


class _PolarAngleRangeAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        minimum = getattr(namespace, "anglePolarPiMin", None)
        maximum = getattr(namespace, "anglePolarPiMax", None)
        if minimum is None or maximum is None:
            return

        if minimum > maximum:
            raise argparse.ArgumentError(
                self,
                "--anglePolarPiMin (%s) must be <= --anglePolarPiMax (%s)"
                % (minimum, maximum),
            )


def build_qmbs_parser():
    argument_parser = argparse.ArgumentParser(
        prog="main_qmbs.py",
        description=QMBS_DESCRIPTION,
        epilog="",
    )

    argument_parser.add_argument(
        "--systemsize", type=_integer_bounded(2, 22), required=True)
    argument_parser.add_argument(
        "--tduration", type=_finite_positive_float, required=True)
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
        description=MBLDTC_DESCRIPTION,
        epilog="",
    )

    argument_parser.add_argument(
        "--systemsize", type=_integer_bounded(1, 22), required=True)
    argument_parser.add_argument(
        "--thetaXPi", type=_finite_float, required=True)
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
        description=MBL_DESCRIPTION,
        default_eigen_backend="qobj"):
    argument_parser = argparse.ArgumentParser(
        prog=prog,
        description=description,
        epilog="",
    )

    argument_parser.add_argument(
        "--systemsize", type=_integer_bounded(1, 22), required=True)
    argument_parser.add_argument(
        "--tduration", type=_finite_positive_float, required=True)
    argument_parser.add_argument("--jIntMean", type=float, required=True)
    argument_parser.add_argument("--bFieldMean", type=float, required=True)
    argument_parser.add_argument(
        "--jIntStd", type=_float_at_least(0), required=True)
    argument_parser.add_argument(
        "--bFieldStd", type=_float_at_least(0), required=True)
    argument_parser.add_argument(
        "--anglePolarPiMin", type=_finite_float, required=True,
        action=_PolarAngleRangeAction)
    argument_parser.add_argument(
        "--anglePolarPiMax", type=_finite_float, required=True,
        action=_PolarAngleRangeAction)
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
    argument_parser.add_argument(
        "--shots", type=_integer_at_least(1), default=None)
    argument_parser.add_argument(
        "--fakeBackend", type=str, default=DEFAULT_FAKE_BACKEND_NAME
    )
    argument_parser.add_argument(
        "--trotterSteps", type=_integer_at_least(1), default=10)
    argument_parser.add_argument("--trotterOrder", type=int, choices=[1, 2], default=2)
    argument_parser.add_argument(
        "--workflow", choices=["mbl", "dtc"], default="mbl")
    argument_parser.add_argument(
        "--thetaXPi", type=_finite_float, default=None)
    argument_parser.add_argument("--insertBarriers", action="store_true")

    return argument_parser


def _dispatch(run, args):
    import logging
    import sys

    logging.basicConfig(level=logging.INFO)

    try:
        run(args)
    except Exception as exc:
        print("error: %s" % exc, file=sys.stderr)
        sys.exit(1)


def main_qmbs():
    from .qmbs_app import run_qmbs

    _dispatch(run_qmbs, build_qmbs_parser().parse_args())


def main_mbldtc():
    from .mbldtc_app import run_mbldtc

    _dispatch(run_mbldtc, build_mbldtc_parser().parse_args())


def main_mbl():
    from .mbl_app import run_mbl

    _dispatch(run_mbl, build_mbl_parser().parse_args())


def main_mbl_dynamics():
    from .mbl_app import run_mbl_dynamics

    args = build_mbl_parser(
        prog="main_mbl_dynamics.py",
        description="MBL return-rate dynamics analysis.",
    ).parse_args()
    _dispatch(run_mbl_dynamics, args)


def main_mbl_propagator():
    from .mbl_app import run_mbl_propagator

    args = build_mbl_parser(
        prog="main_mbl_propagator.py",
        description="",
        default_eigen_backend="numpy",
    ).parse_args()
    _dispatch(run_mbl_propagator, args)


def main_qiskit_sim():
    from .qiskit_app import run_mbldtc_qiskit, run_mbl_qiskit

    parser = build_qiskit_sim_parser()
    args = parser.parse_args()

    if args.workflow == "dtc":
        if args.shots is not None:
            parser.error("dtc workflow takes no shots (exact statevector)")
        if args.simBackend != "statevector":
            parser.error(
                "dtc workflow runs exact statevector; "
                "simBackend must be statevector"
            )
        if args.thetaXPi is None:
            parser.error("dtc workflow requires --thetaXPi")
    else:
        if args.simBackend == "statevector" and args.shots is not None:
            parser.error("shots requires simBackend aer or fake_backend")
        if args.simBackend == "fake_backend" and args.shots is None:
            parser.error("fake_backend requires --shots")

    run = run_mbldtc_qiskit if args.workflow == "dtc" else run_mbl_qiskit
    _dispatch(run, args)
