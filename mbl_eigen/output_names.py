import uuid

import numpy as np


def qmbs_sfim_plot_name(systemsize, tduration, Vrr, Omega, Delta):
    return "qmbs_sfim_N=%02d_tduration=%g_Vrr=%g_Omega=%g_Delta=%g.pdf" % (
        systemsize, tduration, Vrr, Omega, Delta)


def qmbs_pxp_plot_name(systemsize, tduration, Omega, Delta):
    return "qmbs_pxp_N=%02d_tduration=%g_Omega=%g_Delta=%g.pdf" % (
        systemsize, tduration, Omega, Delta)


def mbldtc_plot_name(systemsize, theta_x):
    return "mbldtc_N=%02d_thetax=%gpi_%s.pdf" % (
        systemsize, theta_x / np.pi, uuid.uuid4())


def mbl_entropy_plot_name(
        systemsize,
        anglePolarPiMin,
        anglePolarPiMax,
        jIntMean,
        jIntStd,
        bFieldMean,
        bFieldStd):
    return "mbl_sfim_N=%02d_anglePolarPiMin=%g_anglePolarPiMax=%g" % (
        systemsize, anglePolarPiMin, anglePolarPiMax) + \
        "_jIntMean=%g_jIntStd=%g_bFieldMean=%g_bFieldStd=%g_%s.pdf" % (
        jIntMean, jIntStd, bFieldMean, bFieldStd, uuid.uuid4())


def mbl_dynamics_plot_name(
        systemsize,
        anglePolarPiMin,
        anglePolarPiMax,
        jIntMean,
        jIntStd,
        bFieldMean,
        bFieldStd):
    return "mbl_sfim_dynamics_N=%02d_anglePolarPiMin=%g_anglePolarPiMax=%g" % (
        systemsize, anglePolarPiMin, anglePolarPiMax) + \
        "_jIntMean=%g_jIntStd=%g_bFieldMean=%g_bFieldStd=%g_%s.pdf" % (
        jIntMean, jIntStd, bFieldMean, bFieldStd, uuid.uuid4())
