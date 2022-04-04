import numba
import numpy as np

from grl import config
from grl import graph
from grl.numby import *


@numba.njit(cache=True)
def accuracy(g, L, R=None):
    if R is None:
        raise NotImplementedError("symmetric model not supported")
    if L.shape != R.shape:
        raise NotImplementedError("diagonal model not supported")
    return asymmetric(g, L, R)


@numba.njit(cache=True, parallel=True)
def asymmetric(g, L, R):
    acc = np.empty(config.CORES, dtype=np.float32)
    for i in numba.prange(config.CORES):
        x, y = graph.sample.nce(g, 8192)
        yhat = sigmoid(sum1(L[x[:, 0]]*R[x[:, 1]])).astype(np.float32)
        acc[i] = (round(yhat) == y).mean()
    return np.mean(acc)


@numba.njit(cache=True, parallel=True)
def diagonal(g, L, D):
    acc = np.empty(config.CORES, dtype=np.float32)
    for i in numba.prange(config.CORES):
        x, y = graph.sample.nce(g, 8192)
        yhat = sigmoid(sum1(L[x[:, 0]]*L[x[:, 1]]*D)).astype(np.float32)
        acc[i] = (round(yhat) == y).mean()
    return np.mean(acc)
