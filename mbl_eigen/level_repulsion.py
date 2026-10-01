import logging

import numpy as np


_logger = logging.getLogger(__name__)


def calc_mean_adjacent_level_spacing_ratio(
        eigenvalues: np.ndarray,
        fraction_cutoff=0.02,
        use_spacing=True,
        circular_period=None):
    """
    Calculates the mean adjacent level spacing ratio

    Parameters
    ----------
    eigenvalues: numpy.ndarray
    Eigenvalue spectrum from which to calculate mean adjacent level spacing
    ratio

    fraction_cutoff: float
    Fraction of eigenvalues to skip

    use_spacing: bool, optional
    If True, compute adjacent spacings from the bulk eigenvalues. If
    False, the internal `spacings` variable holds filtered bulk
    eigenvalues (not spacings); branch unused by current callers

    circular_period: float, optional
    Period used to include wraparound spacing for circular spectra

    Returns
    ----------
    ratio_mean: float
    Mean adjacent level spacing ratio
    """
    num_eigenvalues: int = len(eigenvalues)
    _logger.debug("num_eigenvalues = %d" % (num_eigenvalues,))

    ix_start = int(fraction_cutoff * num_eigenvalues)
    ix_end = int((1.0 - fraction_cutoff) * num_eigenvalues)

    _logger.debug("ix_start = %d, ix_end = %g" % (ix_start, ix_end))
    eigenvalues_bulk = eigenvalues[ix_start: ix_end]

    if len(eigenvalues_bulk) == 0:
        raise ValueError(
            "Computing adjacent level spacing ratios requires at least "
            "3 bulk eigenvalues (2 spacings); got 0")

    if use_spacing:
        if circular_period is None:
            spacings: np.ndarray = np.diff(eigenvalues_bulk)
        else:
            spacings = np.diff(np.concatenate((
                eigenvalues_bulk,
                [eigenvalues_bulk[0] + circular_period],
            )))
        _logger.debug("spacings = %s" % spacings)
    else:
        spacings: np.ndarray = eigenvalues_bulk[np.abs(eigenvalues_bulk) > 1e-12]
        _logger.debug("spacings = %s" % spacings)

    if len(spacings) < 2:
        raise ValueError(
            "Computing adjacent level spacing ratios requires at least "
            "3 bulk eigenvalues (2 spacings); got %d" % len(spacings))

    spacing_max: np.ndarray = \
        np.asarray([max(spacings[n], spacings[n + 1])
            for n in range(len(spacings) - 1)])
    _logger.debug("spacing_max = %s" % spacing_max)

    if np.any(spacing_max == 0):
        raise ValueError(
            "Degenerate spectrum: zero adjacent spacing encountered; "
            "spacing ratio undefined")

    spacing_min: np.ndarray = \
        np.asarray([min(spacings[n], spacings[n + 1])
            for n in range(len(spacings) - 1)])
    _logger.debug("spacing_min = %s" % spacing_min)

    ratios: np.ndarray = spacing_min / spacing_max
    _logger.debug("ratios = %s" % ratios)

    ratio_mean: float = np.nanmean(ratios)
    _logger.debug("ratio_mean = %g" % (ratio_mean))

    return ratio_mean


def calc_mean_adjacent_level_spacing_ratio_lenient(*args, **kwargs) -> float:
    """Compute the mean spacing ratio, mapping degenerate input to NaN.

    Wraps ``calc_mean_adjacent_level_spacing_ratio``: inputs rejected by
    that function (fewer than three bulk eigenvalues, degenerate
    spectrum) log a warning and yield ``nan`` instead of raising, so
    workflows can continue past degenerate models.
    """
    try:
        return calc_mean_adjacent_level_spacing_ratio(*args, **kwargs)
    except ValueError as exc:
        _logger.warning("level-spacing ratio undefined: %s", exc)
        return float("nan")
