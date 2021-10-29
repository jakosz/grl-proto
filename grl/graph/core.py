import numba
import numpy as np


@numba.njit(cache=True)
def degree(graph):
    """ Get degrees of nodes in a graph. 
    """
    v, e = graph
    return (v[1:] - v[:-1])[1:]


@numba.njit(cache=True)
def density(graph):
    """ Get density of a graph. 
    """
    return ecount(graph)/(vcount(graph)**2 - vcount(graph))  # @symmetry


@numba.njit(cache=True)
def ecount(graph):
    """ Get edge count of a graph.
    """
    v, e = graph
    return e.shape[0]  # @symmetry


@numba.njit(cache=True)
def neighbours(i, graph):
    """ Get neighbours of i-th node. 
    """
    v, e = graph
    if v[i+1] - v[i] > 0:
        return e[v[i]:v[i] + (v[i+1] - v[i])]
    else:
        return np.array([0], dtype=e.dtype)


@numba.njit(cache=True)
def vcount(graph):
    """ Get vertex count of a graph.
    """
    v, e = graph
    return v.shape[0] - 2  # first is empty, last is the length of edge array
