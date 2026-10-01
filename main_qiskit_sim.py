import logging
import sys

from mbl_eigen.cli import build_qiskit_sim_parser
from mbl_eigen.qiskit_app import run_mbl_qiskit


def main():
    args = build_qiskit_sim_parser().parse_args()

    try:
        run_mbl_qiskit(args)
    except Exception as exc:
        print("mbl-qiskit-sim failed: %s" % exc, file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
