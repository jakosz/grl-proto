import numba
import numpy as np

from . import core
from . import utils


@numba.njit(cache=True)
def get_random_anti_edge(graph):
    """ Sample a random nonexistent edge. """
    src = np.random.choice(graph[1])
    nbs = core.neighbours(src, graph)
    anti = utils.enumerate_without(graph, nbs)
    dst = np.random.choice(anti)
    return np.array([src, dst], dtype=graph[1].dtype)


@numba.njit(cache=True)
def get_random_edge(graph):
    """ Sample a random existing edge. """
    src = np.random.choice(graph[1])
    dst = np.random.choice(core.neighbours(src, graph))
    return np.array([src, dst], dtype=graph[1].dtype)


@numba.njit(cache=True)
def get_random_pair(graph):
    """ Sample a random pair of nodes. """
    return np.random.choice(core.vcount(graph), 2).astype(graph[1].dtype)+1  # @indexing


@numba.njit(cache=True)
def get_nce_sample(graph, n):
    """ Sample edges with noise contrast. 
    """
    n = n//2
    X = np.empty((n*2, 2), dtype=graph[1].dtype)
    Y = np.hstack((np.ones(n), np.zeros(n)))
    
    for i in range(n):
        X[i] = get_random_edge(graph)
        
    for i in range(n):
        X[n+i] = get_random_pair(graph)
        
    # shuffle
    srt = np.arange(n*2)
    np.random.shuffle(srt)
    X, Y = X[srt], Y[srt]

    return X, Y


@numba.njit(cache=True)
def get_neg_sample(graph, n):
    """ Sample edges with negative contrast. 
    """
    n = n//2
    X = np.empty((n*2, 2), dtype=graph[1].dtype)
    Y = np.hstack((np.ones(n), np.zeros(n)))
    
    for i in range(n):
        X[i] = get_random_edge(graph)
        
    for i in range(n):
        X[n+i] = get_random_anti_edge(graph)
    
    # shuffle
    srt = np.arange(n*2)
    np.random.shuffle(srt)
    X, Y = X[srt], Y[srt]
        
    return X, Y


@numba.njit(cache=True)
def _random_uniform_walk(vi, graph, length, data=None, step=0):
    if data is None:
        data = np.zeros(length, dtype=graph[0].dtype.type)
    if length == 0:
        return data
    data[step] = np.random.choice(core.neighbors(vi, graph))
    return _random_uniform_walk(data[step], graph, length-1, data, step+1)


@numba.njit(cache=True)
def random_walk(vi, graph, length):
    return _random_uniform_walk(vi, graph, length)


# API proto
nce = get_nce_sample
neg = get_neg_sample
