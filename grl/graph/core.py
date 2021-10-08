import numba
import numpy as np


@numba.njit()
def degree(graph):
    v, e = graph
    return (v[1:] - v[:-1])[1:]


@numba.njit()
def density(graph):
    return ecount(graph)/(vcount(graph)**2 - vcount(graph))


@numba.njit()
def ecount(graph):
    """ Graph's edge count.
    """
    v, e = graph
    return e.shape[0]


@numba.njit()
def neighbours(i, graph):
    """ Get neighbours of node i. """
    v, e = graph
    if v[i+1] - v[i] > 0:
        return e[v[i]:v[i] + (v[i+1] - v[i])]
    else:
        return np.array([0], dtype=e.dtype)


@numba.njit()
def vcount(graph):
    """ Graph's vertex count.
    """
    v, e = graph
    return v.shape[0] - 2  # first is empty, last is the length of edge array
