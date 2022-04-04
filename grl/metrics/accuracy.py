import numba
import numpy as np

from grl import config
from grl import graph
from grl.numby import *


@numba.njit(cache=True)
def accuracy(g, L, R, activation):
    if np.all(L[0] == R[0]):
        return symmetric(g, L, activation)
    if L.shape != R.shape:
        return diagonal(g, L, R, activation) 
    return asymmetric(g, L, R, activation)


@numba.njit(cache=True, parallel=True)
def asymmetric(g, L, R, activation):
    acc = np.empty(config.CORES, dtype=np.float32)
    for i in numba.prange(config.CORES):
        x, y = graph.sample.nce(g, 8192)
        yhat = activation(sum1(L[x[:, 0]]*R[x[:, 1]])).astype(np.float32)
        acc[i] = (round(yhat) == y).mean()
    return np.mean(acc)


@numba.njit(cache=True, parallel=True)
def diagonal(g, L, D, activation):
    acc = np.empty(config.CORES, dtype=np.float32)
    for i in numba.prange(config.CORES):
        x, y = graph.sample.nce(g, 8192)
        yhat = activation(sum1(L[x[:, 0]]*L[x[:, 1]]*D)).astype(np.float32)
        acc[i] = (round(yhat) == y).mean()
    return np.mean(acc)


@numba.njit(cache=True, parallel=True)
def symmetric(g, L, activation):
    acc = np.empty(config.CORES, dtype=np.float32)
    for i in numba.prange(config.CORES):
        x, y = graph.sample.nce(g, 8192)
        yhat = activation(sum1(L[x[:, 0]]*L[x[:, 1]])).astype(np.float32)
        acc[i] = (round(yhat) == y).mean()
    return np.mean(acc)
