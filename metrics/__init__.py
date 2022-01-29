import numba
import numpy as np


@numba.njit()
def accuracy(g, L, R=None):
    if R is None:
        raise NotImplementedError("symmetric model not supported")
    if L.shape != R.shape:
        raise NotImplementedError("diagonal model not supported")
    return accuracy_asymmetric(g, L, R)


@numba.njit(parallel=True)
def accuracy_asymmetric(g, L, R):
    acc = np.empty(32, dtype=np.float32)
    for i in numba.prange(32):
        x, y = grl.graph.sample.nce(g, 8192)
        yhat = grl.sigmoid(grl.sum1(L[x[:, 0]]*R[x[:, 1]])).astype(np.float32)
        acc[i] = (grl.round(yhat) == y).mean()
    return np.mean(acc)
