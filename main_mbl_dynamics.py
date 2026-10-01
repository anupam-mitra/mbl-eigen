import logging
import sys

from mbl_eigen.cli import build_mbl_parser
from mbl_eigen.mbl_app import run_mbl_dynamics


def main():
    argument_parser = build_mbl_parser(
        prog="main_mbl_dynamics.py",
        description="MBL return-rate dynamics analysis.",
    )
    try:
        run_mbl_dynamics(argument_parser.parse_args())
    except Exception as exc:
        print("Error: %s" % exc, file=sys.stderr)
        print(
            "Please check your inputs and environment, then try again.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
