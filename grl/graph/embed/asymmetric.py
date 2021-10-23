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

            dxL = xR*dy
            dxR = xL*dy
            grl.clip_1d_inplace(dxL, -grl.CLIP, grl.CLIP)
            grl.clip_1d_inplace(dxR, -grl.CLIP, grl.CLIP)

            cos = grl.cos_decay(j / (steps//cores))
            L[x[j, 0]] -= dxL*lr*cos
            R[x[j, 1]] -= dxR*lr*cos
            
    return L, R


@numba.njit()
def decode(L, R, dim=None):
    L = L[1:] if dim is None else L[1:, :dim]  # @indexing
    R = R[1:] if dim is None else R[1:, :dim]  # @indexing
    return grl.sigmoid(L.dot(R.T))
