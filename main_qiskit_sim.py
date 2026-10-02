import logging

from mbl_eigen.cli import main_qiskit_sim


def main():
    main_qiskit_sim()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
