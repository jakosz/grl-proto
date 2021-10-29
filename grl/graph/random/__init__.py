import numba as numba
import numpy as np

from ._igraph import *


__all__ = [
    "seed"
]


@numba.njit(cache=True)
def seed(seed=None):
    """ Set seed for numba jitted functions. 
    """
    if seed is not None:
        np.random.seed(seed)
