import logging

from mbl_eigen.cli import build_mbl_parser
from mbl_eigen.mbl_app import run_mbl_dynamics


def main():
    argument_parser = build_mbl_parser(
        prog="main_mbl_dynamics.py",
        description="MBL return-rate dynamics analysis.",
    )
    run_mbl_dynamics(argument_parser.parse_args())


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
