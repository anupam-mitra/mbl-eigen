"""Centralized output filename formatters for all MBL analysis workflows."""

import uuid

import numpy as np


def qmbs_sfim_plot_name(systemsize, tduration, Vrr, Omega, Delta):
    """Return a stable filename for the QMBS Ising (SFIM) eigenphase plot."""
    return "qmbs_sfim_N=%02d_tduration=%g_Vrr=%g_Omega=%g_Delta=%g.pdf" % (
        systemsize,
        tduration,
        Vrr,
        Omega,
        Delta,
    )


def qmbs_pxp_plot_name(systemsize, tduration, Omega, Delta):
    """Return a stable filename for the QMBS PXP eigenphase plot."""
    return "qmbs_pxp_N=%02d_tduration=%g_Omega=%g_Delta=%g.pdf" % (
        systemsize,
        tduration,
        Omega,
        Delta,
    )


def mbldtc_plot_name(systemsize, theta_x):
    """Return a UUID-suffixed filename for an MBL-DTC eigenphase plot."""
    return "mbldtc_N=%02d_thetax=%gpi_%s.pdf" % (
        systemsize,
        theta_x / np.pi,
        uuid.uuid4(),
    )


def _mbl_plot_name(prefix, systemsize, anglePolarPiMin, anglePolarPiMax,
                   jIntMean, jIntStd, bFieldMean, bFieldStd):
    """Shared formatter for MBL plot filenames (internal helper)."""
    return (
        "%s_N=%02d_anglePolarPiMin=%g_anglePolarPiMax=%g"
        "_jIntMean=%g_jIntStd=%g_bFieldMean=%g_bFieldStd=%g_%s.pdf"
    ) % (
        prefix,
        systemsize,
        anglePolarPiMin,
        anglePolarPiMax,
        jIntMean,
        jIntStd,
        bFieldMean,
        bFieldStd,
        uuid.uuid4(),
    )


def mbl_entropy_plot_name(
    systemsize, anglePolarPiMin, anglePolarPiMax, jIntMean, jIntStd,
    bFieldMean, bFieldStd,
):
    """Return a UUID-suffixed filename for an MBL eigenvector entropy plot."""
    return _mbl_plot_name(
        "mbl_sfim", systemsize, anglePolarPiMin, anglePolarPiMax,
        jIntMean, jIntStd, bFieldMean, bFieldStd,
    )


def mbl_dynamics_plot_name(
    systemsize, anglePolarPiMin, anglePolarPiMax, jIntMean, jIntStd,
    bFieldMean, bFieldStd,
):
    """Return a UUID-suffixed filename for an MBL return-rate dynamics plot."""
    return _mbl_plot_name(
        "mbl_sfim_dynamics", systemsize, anglePolarPiMin, anglePolarPiMax,
        jIntMean, jIntStd, bFieldMean, bFieldStd,
    )
