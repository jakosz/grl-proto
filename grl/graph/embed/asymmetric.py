import numba
import numpy as np

import grl


@numba.njit(fastmath=True, parallel=True)
def encode(graph, dim, lr, steps):
    
    cores = numba.config.NUMBA_NUM_THREADS
    L = np.random.randn(grl.vcount(graph)+1, dim)/dim  # @indexing
    R = np.random.randn(grl.vcount(graph)+1, dim)/dim  # @indexing

    for i in numba.prange(cores):
        
        x, y = grl.graph.sample.neg(graph, steps//cores)
        
        for j in range(steps//cores):

            xL = L[x[j, 0]]
            xR = R[x[j, 1]]

            dy = grl.sigmoid(np.sum(xL*xR)) - y[j]

            cos = grl.cos_decay(j / (steps//cores))
            L[x[j, 0]] = L[x[j, 0]] - xR*dy*lr*cos
            R[x[j, 1]] = R[x[j, 1]] - xL*dy*lr*cos
            
    return L, R


@numba.njit()
def decode(L, R, dim=None):
    L = L[1:] if dim is None else L[1:, :dim]  # @indexing
    R = R[1:] if dim is None else R[1:, :dim]  # @indexing
    return grl.sigmoid(L.dot(R.T))
