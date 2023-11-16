import numpy as np
import logging

################################################################################
def calc_mean_adjacent_level_spacing_ratio(
        eigenvalues:np.ndarray,
        fraction_cutoff=0.02,
        use_spacing=True):
    """
    Calculates the mean adjacent level spacing ratio

    Parameters
    ----------
    eigenvalues: numpy.ndarray
    Eigenvalue spectrum from which to calculate mean adjacent level spacing 
    ratio

    fraction_cutoff: float
    Fraction of eigenvalues to skip

    Returns
    ------
    ratio_mean: float
    Mean adjacent level spacing ratio
    """
    num_eigenvalues:int = len(eigenvalues)
    logging.debug("num_eigenvalues = %d" % (num_eigenvalues,))
    
    ix_start = int(fraction_cutoff * num_eigenvalues) 
    ix_end = int((1.0-fraction_cutoff) * num_eigenvalues) + 1
    
    logging.debug("ix_start = %d, ix_end = %g" % (ix_start, ix_end))
    eigenvalues_bulk = eigenvalues[ix_start : ix_end]

    if use_spacing:
        spacings:np.ndarray = np.diff(eigenvalues_bulk)
        logging.debug("spacings = %s" % spacings)
    else:
        spacings:np.nd_array = eigenvalues_bulk[np.abs(eigenvalues_bulk) > 1e-12]
        logging.debug("spacings = %s" % spacings)
    
    spacing_max:np.ndarray = \
        np.asarray([max(spacings[n], spacings[n+1]) \
            for n in range(len(spacings)-1)])
    logging.debug("spacing_max = %s" % spacing_max)
    
    spacing_min:np.ndarray = \
        np.asarray([min(spacings[n], spacings[n+1]) \
            for n in range(len(spacings)-1)])
    logging.debug("spacing_min = %s" % spacing_min)
    
    ratios:np.ndarray = spacing_min / spacing_max
    logging.debug("ratios = %s" % ratios)
    
    ratio_mean:float = np.nanmean(ratios)
    logging.debug("ratio_mean = %g" % (ratio_mean))
    
    return ratio_mean
################################################################################
