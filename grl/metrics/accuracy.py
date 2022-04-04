import numba

from grl.numby import *


@numba.njit(cache=True)
def accuracy(y, yhat):
    return (round(yhat) == y).mean()
