import numba
import numpy as np

import grl


@numba.njit(fastmath=True, parallel=True)
def _encode_diagonal(graph, dim, lr, steps):
    
    cores = numba.config.NUMBA_NUM_THREADS
    model = np.random.randn(grl.vcount(graph)+1, dim)/dim  # @indexing
    diag = np.random.randn(dim)/dim

    for i in numba.prange(cores):
        
        x, y = grl.graph.sample.neg(graph, steps//cores)
        
        for j in range(steps//cores):

            xL = model[x[j, 0]]
            xR = model[x[j, 1]]
            xLR = xL*xR
            xLRd = xLR*diag

            dy = grl.sigmoid(np.sum(xLRd)) - y[j]
            
            dxLR = diag*dy
            ddiag = xLR*dy

            cos = grl.cos_decay(j / (steps//cores))
            model[x[j, 0]] = model[x[j, 0]] - xR*dxLR*lr*cos
            model[x[j, 1]] = model[x[j, 1]] - xL*dxLR*lr*cos
            diag = diag - ddiag*lr*cos
            
    return model, diag


@numba.njit(fastmath=True, parallel=True)
def _encode_symmetric(graph, dim, lr, steps):
    
    cores = numba.config.NUMBA_NUM_THREADS
    model = np.random.randn(grl.vcount(graph)+1, dim)/dim  # @indexing

    for i in numba.prange(cores):
        
        x, y = grl.graph.sample.neg(graph, steps//cores)
        
        for j in range(steps//cores):

            xL = model[x[j, 0]]
            xR = model[x[j, 1]]

            dy = grl.sigmoid(np.sum(xL*xR)) - y[j]

            cos = grl.cos_decay(j / (steps//cores))
            model[x[j, 0]] = model[x[j, 0]] - xR*dy*lr*cos
            model[x[j, 1]] = model[x[j, 1]] - xL*dy*lr*cos
            
    return model


@numba.njit(fastmath=True, parallel=True)
def _encode_asymmetric(graph, dim, lr, steps):
    
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
def _decode_diagonal(model, diag, dim=None):
    model = model[1:] if dim is None else model[1:, :dim]  # @indexing
    diag = diag if dim is None else diag[:dim]
    model = model*diag
    return grl.sigmoid(model.dot(model.T))


@numba.njit()
def _decode_asymmetric(L, R, dim=None):
    L = L[1:] if dim is None else L[1:, :dim]  # @indexing
    R = R[1:] if dim is None else R[1:, :dim]  # @indexing
    return grl.sigmoid(L.dot(R.T))


@numba.njit()
def _decode_symmetric(model, dim=None):
    model = model[1:] if dim is None else model[1:, :dim]  # @indexing
    return grl.sigmoid(model.dot(model.T))
