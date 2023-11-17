import numpy as np
import qutip

import symmetry
################################################################################
class ReflectionAboutCenter(symmetry.Involution):
    """
    Implements reflection symmetry about the center of a one-dimensional array.

    For an array with sites `0, 1, ..., n-1`,
    swap `0`, `n-1`
    swap `1`, `n-2
    and so on

    If `n` is even, `n/2-1` and `n/2` can be swapped.

    If `n` is odd, `n//2` is the middle and is not swapped with any other site.

    Parameters
    ----------
    n_sites: int
    Number of sites in the one-dimensional array

    swap: operator
    Swap operator that swaps two sites.
    """

    def __init__(self, n_sites:int, swap:qutip.qobj.Qobj):
        """
        Initializes the object
        """
        self.n_sites = n_sites
        self.swap = swap
        self.gen_reflection_op()

    def gen_reflection_op(self):
        """
        Generates the reflection operator which implements a reflection
        about the center.
        """

        self.op = reflection_about_center(
            self.n_sites, self.swap)

    def diagonalize_op(self):
        """
        Diagaonlizes the reflection operato
        """

        if not hasattr(self, "eigenvalues"):
            self.eigenvalues, self.eigenvectors = \
                self.op.eigenstates()

    def get_even_projector_eigen(self):
        """
        Calculates the projector onto the even eigenvalue subspace
        of the reflection operator by calculating the eigenvalues
        and eigenvectors.
        """

        if not hasattr(self, "even_projector_eigen"):

            self.diagonalize_op()

            self.indices_even_reflection = \
                 np.where(self.eigenvalues > 0)

            self.even_projector_eigen = sum([qutip.ket2dm(k) \
                for k in \
                    self.eigenvectors[self.indices_even_reflection]])

        return self.even_projector_eigen
################################################################################
################################################################################
def reflection_about_center(
        n_sites:int, swap:qutip.qobj.Qobj,
):
    """
    Calculates a reflection operator about the center in a one-dimensional array

    Parameters
    ----------
    n_sites: int
    Number of sites in the one-dimensional array

    swap: qutip.qobj.Qobj
    Swap operator, wchih swaps two sites

    Returns
    -------
    reflection_op: qutip.qobj.Qobj
    Reflection operator about the center  
    """
    swap_op_array = np.empty((n_sites//2,), dtype=object)

    for ix_site in range(n_sites//2):
        op = qutip.qip.operations.expand_operator(
            swap, n_sites, targets=(ix_site, n_sites - 1 - ix_site),)

        swap_op_array[ix_site] = op

    reflection_op = swap_op_array[0]

    for ix_site in range(1, n_sites//2):
        reflection_op = reflection_op * swap_op_array[ix_site]

    return reflection_op
################################################################################
