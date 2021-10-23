import numba
import numpy as np

import grl


@numba.njit(fastmath=True, parallel=True)
def encode(graph, dim, lr, steps):
    
    cores = numba.config.NUMBA_NUM_THREADS
    model = np.random.randn(grl.vcount(graph)+1, dim)/dim  # @indexing

    for i in numba.prange(cores):
        
        x, y = grl.graph.sample.neg(graph, steps//cores)
        
        for j in range(steps//cores):

            xL = model[x[j, 0]]
            xR = model[x[j, 1]]

            dy = grl.sigmoid(np.sum(xL*xR)) - y[j]

            dxL = xR*dy
            dxR = xL*dy
            grl.clip_1d_inplace(dxL, -grl.CLIP, grl.CLIP)
            grl.clip_1d_inplace(dxR, -grl.CLIP, grl.CLIP)
            
            cos = grl.cos_decay(j / (steps//cores))
            model[x[j, 0]] -= dxL*lr*cos
            model[x[j, 1]] -= dxR*lr*cos
            
    return model


@numba.njit()
def decode(model, dim=None):
    model = model[1:] if dim is None else model[1:, :dim]  # @indexing
    return grl.sigmoid(model.dot(model.T))
