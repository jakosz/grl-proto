import numba
import numpy as np

from . import core


@numba.njit()
def enumerate_nodes(graph):
    return (np.arange(core.vcount(graph))+1).astype(graph[1].dtype)


@numba.njit()
def enumerate_without(graph, subset):
    """ Enumerate nodes and remove given subset from the enumeration. 
    """
    nodes = enumerate_nodes(graph)
    for i in subset-1: # @1-indexing
        nodes[i] = 0
    return nodes[nodes != 0]


@numba.njit()
def get_random_anti_edge(graph):
    """ Sample a random nonexistent edge. """
    src = np.random.choice(graph[1])
    nbs = core.neighbours(src, graph)
    anti = enumerate_without(graph, nbs)
    dst = np.random.choice(anti)
    return np.array([src, dst], dtype=graph[1].dtype)


@numba.njit()
def get_random_edge(graph):
    """ Sample a random existing edge. """
    src = np.random.choice(graph[1])
    dst = np.random.choice(core.neighbours(src, graph))
    return np.array([src, dst], dtype=graph[1].dtype)


@numba.njit()
def get_random_pair(graph):
    """ Sample a random pair of nodes. """
    return np.random.choice(enumerate_nodes(graph), 2).astype(graph[1].dtype)


@numba.njit()
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
        
    return X, Y


@numba.njit()
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
        
    return X, Y


# API proto
nce = get_nce_sample
neg = get_neg_sample
