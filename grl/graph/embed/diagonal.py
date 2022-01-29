""" This module implements embedding graph nodes in a symmetric matrix and a diagonal:

$$ A \approx \sigma(XDX^{\top}) $$

where A is the adjacency matrix, X is the embedding, and D is the diagonal.  

"""

from concurrent.futures import ProcessPoolExecutor

import numba
import numpy as np

import grl
from . import _utils


@numba.njit(cache=True)
def worker(x, y, E, D, lr, loss):
    
    if loss == 'logistic':
        f = grl.sigmoid
    elif loss == 'mse':
        f = grl.identity
    else:
        return 1
    
    n = x.shape[0]
    for i in range(n):
        xL = E[x[i, 0]]
        xR = E[x[i, 1]]
        # compute gradients
        dy = f(np.sum(xL*xR*D)) - y[i]  # output
        dxLR = D*dy  # embedding product 
        dD = xL*xR*dy  # diagonal
        dxL = xR*dxLR  # left vector
        dxR = xL*dxLR  # right vector
        grl.clip_1d_inplace(dD, -grl.CLIP, grl.CLIP)
        grl.clip_1d_inplace(dxL, -grl.CLIP, grl.CLIP)
        grl.clip_1d_inplace(dxR, -grl.CLIP, grl.CLIP)
        # update parameters
        cos = grl.cos_decay(i/n)
        E[x[i, 0]] -= dxL*lr*cos
        E[x[i, 1]] -= dxR*lr*cos
        D[:] -= dD*lr*cos
    return 0


def worker_mp_wrapper(graph, steps, name_E, name_D, lr, loss):
    x, y = grl.graph.sample.neg(graph, steps)
    return worker(x, y, grl.get(name_E), grl.get(name_D), lr, loss)


def encode(graph, dim, steps, lr=.025, loss='logistic'):
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
            Learning rate. 
            Note that cosine decay is applied automatically.
        loss : str
            Loss function to optimize. 
            Supported values are 'logistic' and 'mse'. 

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
    D = grl.empty((dim,), np.float32, name_D)  # diagonal
    D[:] = np.random.randn(dim)/dim
    
    # fit the model 
    with ProcessPoolExecutor(grl.CORES) as p:
        for core in range(grl.CORES):
            p.submit(worker_mp_wrapper, graph, n, name_E, name_D, lr, loss)
    
    return grl.get(name_E), grl.get(name_D).ravel()


@numba.njit(cache=True)
def encode_st(graph, dim, steps, lr=.025, loss='logistic'):
    """ Single-threaded embedding. 
        Most efficient on graphs of up to 100 vertices. 
    """
    E = np.random.randn(grl.vcount(graph)+1, dim)/dim  # node embedding @indexing
    D = np.random.randn(dim)  # diagonal
    x, y = grl.graph.sample.neg(graph, steps)
    worker(x, y, E, D, lr, loss)
    return E, D 


@numba.njit(cache=True)
def decode(model, diag, dim=None):
    model = model[1:] if dim is None else model[1:, :dim]  # @indexing
    diag = diag if dim is None else diag[:dim]
    return grl.sigmoid(model.dot(np.diag(diag)).dot(model.T))
