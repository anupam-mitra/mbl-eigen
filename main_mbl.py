import logging
from mbl_eigen.cli import build_mbl_parser
from mbl_eigen.mbl_app import run_mbl


def main():
    argument_parser = build_mbl_parser()
    run_mbl(argument_parser.parse_args())


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
