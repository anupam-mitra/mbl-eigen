import logging
from mbl_eigen.cli import build_qmbs_parser
from mbl_eigen.qmbs_app import run_qmbs


def main():
    argument_parser = build_qmbs_parser()
    run_qmbs(argument_parser.parse_args())


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
