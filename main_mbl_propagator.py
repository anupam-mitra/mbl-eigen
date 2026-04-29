import logging

from mbl_eigen.cli import build_mbl_parser
from mbl_eigen.mbl_app import run_mbl_propagator


def main():
    argument_parser = build_mbl_parser(
        prog="main_mbl_propagator.py",
        description="",
        default_eigen_backend="numpy",
    )
    run_mbl_propagator(argument_parser.parse_args())


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
