import logging
from mbl_eigen.cli import build_mbldtc_parser
from mbl_eigen.mbldtc_app import run_mbldtc


def main():
    argument_parser = build_mbldtc_parser()
    run_mbldtc(argument_parser.parse_args())


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
