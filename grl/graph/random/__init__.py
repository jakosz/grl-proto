import numba as _numba
import numpy as _np

from ._igraph import *


@_numba.njit()
def seed(seed=None):
    """ Set seed for numba jitted functions. 
    """
    if seed is not None:
        _np.random.seed(seed)
