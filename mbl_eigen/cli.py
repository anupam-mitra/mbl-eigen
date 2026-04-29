import argparse


COMMON_DESCRIPTION = "Plots eigenphases of the time evolution operator under a PXP like Hamiltonian"


def build_qmbs_parser():
    argument_parser = argparse.ArgumentParser(
        prog="main_qmbs.py",
        description=COMMON_DESCRIPTION,
        epilog="",
    )

    argument_parser.add_argument("--systemsize", type=int)
    argument_parser.add_argument("--tduration", type=float)
    argument_parser.add_argument("--Delta", type=float)

    return argument_parser


def build_mbldtc_parser():
    argument_parser = argparse.ArgumentParser(
        prog="main_qmbs.py",
        description=COMMON_DESCRIPTION,
        epilog="",
    )

    argument_parser.add_argument("--systemsize", type=int)
    argument_parser.add_argument("--thetaXPi", type=float)

    return argument_parser


def build_mbl_parser(
        prog="main_qmbs.py",
        description=COMMON_DESCRIPTION):
    argument_parser = argparse.ArgumentParser(
        prog=prog,
        description=description,
        epilog="",
    )

    argument_parser.add_argument("--systemsize", type=int)
    argument_parser.add_argument("--tduration", type=float)
    argument_parser.add_argument("--jIntMean", type=float)
    argument_parser.add_argument("--bFieldMean", type=float)
    argument_parser.add_argument("--jIntStd", type=float)
    argument_parser.add_argument("--bFieldStd", type=float)
    argument_parser.add_argument("--anglePolarPiMin", type=float)
    argument_parser.add_argument("--anglePolarPiMax", type=float)

    return argument_parser
