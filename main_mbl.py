import logging
import sys

from mbl_eigen.cli import build_mbl_parser
from mbl_eigen.mbl_app import run_mbl


def main():
    argument_parser = build_mbl_parser(
        prog="main_mbl.py",
        description="MBL spectrum + eigenvector entropy analysis.",
    )
    try:
        run_mbl(argument_parser.parse_args())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
