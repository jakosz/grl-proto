from concurrent.futures import ProcessPoolExecutor

import numba
import numpy as np

import grl
from . import _utils 


@numba.njit(fastmath=True, cache=True)
def worker(x, y, E, lr):
    n = x.shape[0]    
    for i in range(n):
        xL = E[x[i, 0]]
        xR = E[x[i, 1]]
        # compute gradients
        dy = grl.sigmoid(np.sum(xL*xR)) - y[i]
        dxL = xR*dy
        dxR = xL*dy
        grl.clip_1d_inplace(dxL, -grl.CLIP, grl.CLIP)
        grl.clip_1d_inplace(dxR, -grl.CLIP, grl.CLIP)
        # update embedding
        cos = grl.cos_decay(i/n)
        E[x[i, 0]] = E[x[i, 0]] - dxL*lr*cos
        E[x[i, 1]] = E[x[i, 1]] - dxR*lr*cos
        

def worker_mp_wrapper(graph, steps, name, lr):
    x, y = grl.graph.sample.neg(graph, steps)
    return worker(x, y, grl.get(name), lr)


def encode(graph, dim, steps, lr=.025):
    name = np.random.randint(0, 2**63, 1).tobytes().hex()
    n = grl.graph.embed._utils.split_steps(steps, grl.CORES)  # number of steps per core
    model = grl.randn((grl.vcount(graph)+1, dim), name)  # @indexing
    
    with ProcessPoolExecutor(grl.CORES) as p:
        for core in range(grl.CORES):
            p.submit(worker_mp_wrapper, graph, n, name, lr)
    
    return grl.get(name)


@numba.njit(cache=True)
def encode_st(graph, dim, steps, lr=.025):
    """ Single-threaded embedding. 
        Most efficient on graphs of up to 100 vertices. 
    """
    E = np.random.randn(grl.vcount(graph)+1, dim)/dim  # @indexing
    x, y = grl.graph.sample.neg(graph, steps)
    worker(x, y, E, lr)
    return E 


@numba.njit(cache=True)
def decode(model, dim=None):
    model = model[1:] if dim is None else model[1:, :dim]  # @indexing
    return grl.sigmoid(model.dot(model.T))
