from concurrent.futures import ProcessPoolExecutor

import numba
import numpy as np

import grl
from . import _utils 


@numba.njit(fastmath=True, cache=True)
def worker(x, y, L, R, lr):
    n = x.shape[0]
    for j in range(n):
        xL = L[x[j, 0]]
        xR = R[x[j, 1]]
        # compute gradients
        dy = grl.sigmoid(np.sum(xL*xR)) - y[j]
        dxL = xR*dy
        dxR = xL*dy
        grl.clip_1d_inplace(dxL, -grl.CLIP, grl.CLIP)
        grl.clip_1d_inplace(dxR, -grl.CLIP, grl.CLIP)
        # update embedding
        cos = grl.cos_decay(j/n) 
        L[x[j, 0]] -= dxL*lr*cos
        R[x[j, 1]] -= dxR*lr*cos


def worker_mp_wrapper(graph, steps, name_L, name_R, lr):
    x, y = grl.graph.sample.neg(graph, steps)
    return worker(x, y, grl.get(name_L), grl.get(name_R), lr)


def encode(graph, dim, steps, lr=.025):
    name_L = grl.utils.random_hex()
    name_R = grl.utils.random_hex()
    n = grl.graph.embed._utils.split_steps(steps, grl.CORES)  # number of steps per core
    L = grl.randn((grl.vcount(graph)+1, dim), name_L)  # @indexing
    R = grl.randn((grl.vcount(graph)+1, dim), name_R)  # @indexing
    
    with ProcessPoolExecutor(grl.CORES) as p:
        for core in range(grl.CORES):
            p.submit(worker_mp_wrapper, graph, n, name_L, name_R, lr)
    
    return grl.get(name_L), grl.get(name_R)


@numba.njit(cache=True)
def encode_st(graph, dim, steps, lr=.025):
    """ Single-threaded embedding. 
        Most efficient on graphs of up to 100 vertices. 
    """
    L = np.random.randn(grl.vcount(graph)+1, dim)/dim  # @indexing
    R = np.random.randn(grl.vcount(graph)+1, dim)/dim  # @indexing
    x, y = grl.graph.sample.neg(graph, steps)
    worker(x, y, L, R, lr)
    return L, R 


@numba.njit(cache=True)
def decode(L, R, dim=None):
    L = L[1:] if dim is None else L[1:, :dim]  # @indexing
    R = R[1:] if dim is None else R[1:, :dim]  # @indexing
    return grl.sigmoid(L.dot(R.T))
