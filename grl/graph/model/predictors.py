import numba
import numpy as np

from grl import numby


def expand_dims(f):
    def wrap(*args):
        x, args = args[0], args[1:]
        x = np.expand_dims(x, 0) if x.ndim == 1 else x
        return f(x, *args)
    return wrap


@expand_dims
@numba.njit(cache=True)
def asymmetric(x, L, R, activation):
    xL = L[x[:, 0]]
    xR = R[x[:, 1]]
    return activation(numby.sum1(xL*xR))


@expand_dims
@numba.njit(cache=True)
def diagonal(x, L, D, activation):
    xL = L[x[:, 0]]
    xR = L[x[:, 1]]
    return activation(numby.sum1(xL*xR*D))


@expand_dims
@numba.njit(cache=True)
def symmetric(x, L, activation):
    xL = L[x[:, 0]]
    xR = L[x[:, 1]]
    return activation(numby.sum1(xL*xR))
