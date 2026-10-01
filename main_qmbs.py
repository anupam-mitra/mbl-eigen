import logging
import sys

from mbl_eigen.cli import build_qmbs_parser
from mbl_eigen.qmbs_app import run_qmbs


def main():
    argument_parser = build_qmbs_parser()
    try:
        run_qmbs(argument_parser.parse_args())
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
