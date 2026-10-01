import logging
import sys

from mbl_eigen.cli import build_mbldtc_parser
from mbl_eigen.mbldtc_app import run_mbldtc


def main():
    argument_parser = build_mbldtc_parser()
    try:
        run_mbldtc(argument_parser.parse_args())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
