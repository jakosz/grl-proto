import numba
import numpy as np

import grl


@numba.njit(fastmath=True, parallel=True)
def encode(graph, dim, lr, steps):
    
    cores = numba.config.NUMBA_NUM_THREADS
    model = np.random.randn(grl.vcount(graph), dim)/dim

    for i in numba.prange(cores):
        
        x, y = grl.graph.sample.neg(graph, steps//cores)
        
        for j in range(steps//cores):

            xL = model[x[j, 0]]
            xR = model[x[j, 1]]

            dy = grl.sigmoid(np.sum(xL*xR)) - y[j]

            cos = grl.cos_decay(j / steps//cores)
            model[x[j, 0]] = model[x[j, 0]] - xR*dy*lr*cos
            model[x[j, 1]] = model[x[j, 1]] - xL*dy*lr*cos
            
    return model


@numba.njit()
def decode(model):
    return grl.sigmoid(model.dot(model.T))
