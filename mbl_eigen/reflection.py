import numpy as np
import qutip
from qutip.qip.operations import expand_operator


def reflection_about_center(
        n_sites: int, swap: qutip.Qobj,
):
    """
    Calculates a reflection operator about the center in a one-dimensional array

    Parameters
    ----------
    n_sites: int
    Number of sites in the one-dimensional array

    swap: qutip.Qobj
    Swap operator, wchih swaps two sites

    Returns
    -------
    reflection_op: qutip.Qobj
    Reflection operator about the center
    """
    if isinstance(n_sites, bool) or not isinstance(n_sites, (int, np.integer)):
        raise ValueError("n_sites must be a positive integer")
    if n_sites < 1:
        raise ValueError("n_sites must be a positive integer")
    if n_sites == 1:
        return qutip.qeye(swap.dims[0][0])

    swap_op_array = np.empty((n_sites // 2,), dtype=object)

    for ix_site in range(n_sites // 2):
        op = expand_operator(
            swap, n_sites, targets=(ix_site, n_sites - 1 - ix_site),)

        swap_op_array[ix_site] = op

    reflection_op = swap_op_array[0]

    for ix_site in range(1, n_sites // 2):
        reflection_op = reflection_op * swap_op_array[ix_site]

    return reflection_op

