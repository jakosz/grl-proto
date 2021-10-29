from concurrent.futures import ProcessPoolExecutor

import numba
import numpy as np

import grl
from . import _utils


@numba.njit(fastmath=True, cache=True)
def worker(x, y, E, D, lr):
    n = x.shape[0]
    for j in range(n):
        xL = E[x[j, 0]]
        xR = E[x[j, 1]]
        # compute gradients
        dy = grl.sigmoid(np.sum(xL*xR*D)) - y[j]  # output
        dxLR = D*dy  # embedding product 
        dD = xL*xR*dy  # diagonal
        dxL = xR*dxLR  # left vector
        dxR = xL*dxLR  # right vector
        grl.clip_1d_inplace(dD, -grl.CLIP, grl.CLIP)
        grl.clip_1d_inplace(dxL, -grl.CLIP, grl.CLIP)
        grl.clip_1d_inplace(dxR, -grl.CLIP, grl.CLIP)
        # update parameters
        cos = grl.cos_decay(j/n)
        E[x[j, 0]] -= dxL*lr*cos
        E[x[j, 1]] -= dxR*lr*cos
        D[:] -= dD*lr*cos


def worker_mp_wrapper(graph, steps, name_E, name_D, lr):
    x, y = grl.graph.sample.neg(graph, steps)
    return worker(x, y, grl.get(name_E), grl.get(name_D), lr)


def encode(graph, dim, steps, lr=.025):
    """ Embed graph nodes in a symmetric matrix with diagonal.

        Parameters
        ----------
        graph : grl.graph
            Input graph.
        dim : int
            Embedding dimension.
        steps: int
            Number of iterations to perform. 
        lr : float
            Learning rate, optional, defaults to .025.
            Note that cosine decay is applied automatically.

        Returns
        -------
        E, D : 2darray, 1darray
            Embedding, diagonal. 
    """
    n = grl.graph.embed._utils.split_steps(steps, grl.CORES)  # number of steps per core
    # generate names for the shared memory object namespace
    name_E = grl.utils.random_hex()
    name_D = grl.utils.random_hex()
    E = grl.randn((grl.vcount(graph)+1, dim), name_E)  # node embedding @indexing
    D = grl.empty((dim,), name_D)  # diagonal
    D[:] = np.random.randn(dim)/dim
    
    # fit the model 
    with ProcessPoolExecutor(grl.CORES) as p:
        for core in range(grl.CORES):
            p.submit(worker_mp_wrapper, graph, n, name_E, name_D, lr)
    
    return grl.get(name_E), grl.get(name_D).ravel()


@numba.njit()
def encode_st(graph, dim, steps, lr=.025):
    """ Single-threaded embedding. 
        Most efficient on graphs of up to 100 vertices. 
    """
    E = np.random.randn(grl.vcount(graph)+1, dim)/dim  # node embedding @indexing
    D = np.random.randn(dim)  # diagonal
    x, y = grl.graph.sample.neg(graph, steps)
    worker(x, y, E, D, lr)
    return E, D 


@numba.njit()
def decode(model, diag, dim=None):
    model = model[1:] if dim is None else model[1:, :dim]  # @indexing
    diag = diag if dim is None else diag[:dim]
    return grl.sigmoid(model.dot(np.diag(diag)).dot(model.T))
