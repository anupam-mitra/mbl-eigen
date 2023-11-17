import numpy as np
import qutip

################################################################################
class Symmetry:
    """
    Implementation of a symmetry operator
    """
    def __init__(self):
        pass


class Involution:
    """
    Implmentation of a symmetry which it its own inverse, that is whose
    eigenvalues are +1 and -1
    """

    def __init__(self):
        pass

    def get_even_projector(self):
        """
        Calculates the projector onto the even eigenvalue subspace
        of the reflection operator by using the fact that the reflection
        operator has eigenvalues `+1` and `-1. The projector onto the
        `+1` eigenbasis is `(identity + operator)/2`
        """

        if not hasattr(self, "even_projector"):
            self.even_projector = \
                (qutip.identity(dims=self.op.dims[0]) +
                self.op) * 0.5

        return self.even_projector

    def get_odd_projector(self):
        """
        Calculates the projector onto the odd eigenvalue subspace
        of the reflection operator by using the fact that the reflection
        operator has eigenvalues `+1` and `-1. The projector onto the
        `+1` eigenbasis is `(identity - operator)/2`
        """

        if not hasattr(self, "odd_projector"):
            self.odd_projector = \
                (qutip.identity(dims=self.op.dims[0]) -
                self.op) * 0.5

        return self.odd_projector
