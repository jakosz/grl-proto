import numba
import numba.types as nt
import numpy as np

import grl.types as gt

from . import core
from . import utils
from .. import numby


def sampler(f_positives, f_negatives):
    """ Define graph sampler. 

        Parameters
        ----------
        f_positives : function(graph)
            A function taking graph and returning 1darray with a random example
            of an edge.  
        f_negatives : function(graph, vcount2)
            A function taking graph and number of nodes in secondary (non-indexed) 
            modality and returning 1darray with a random example of a missing edge. 

        Returns
        -------
        function
    """
    def wrap(f):
        def get_sample(graph, n, vcount2=0):
            n = n//2
            X = np.empty((n*2, 2), dtype=graph[1].dtype)
            Y = np.hstack((np.ones(n), np.zeros(n)))

            for i in range(n):
                X[i] = f_positives(graph)

            for i in range(n):
                X[n+i] = f_negatives(graph, vcount2)

            # shuffle
            srt = np.arange(n*2)
            np.random.shuffle(srt)
            X, Y = X[srt], Y[srt]

            return X, Y
        return get_sample
    return wrap


@numba.njit(cache=True)
def get_random_anti_edge(graph, vcount2=0):
    """ Sample a random nonexistent edge. """
    vs = np.random.choice(core.vcount(graph), 16) + 1  # @indexing
    src, dst = vs[0], vs[1:]
    nbs = core.neighbors(src, graph)
    for v in dst:
        if not numby.isin_1d(v, nbs):
            return np.array([src, v], dtype=graph[1].dtype)
    return get_random_anti_edge(graph)


@numba.njit(cache=True)
def get_random_edge(graph):
    """ Sample a random existing edge. """
    src = np.random.choice(core.vcount(graph)) + 1  # @indexing
    dst = np.random.choice(core.neighbors(src, graph))
    if not dst:
        return get_random_edge(graph)
    return np.array([src, dst], dtype=graph[1].dtype)


@numba.njit(cache=True)
def get_random_pair(graph, vcount2=0):
    """ Sample a random pair of nodes. 
        
        Parameters
        ----------
        graph : tuple
            grl.graph
        vcount2 : int, optional
            Number of nodes of the second (non-indexed) modality in the graph. 
            For unimodal graphs this value should be 0 (default).

        Returns
        -------
        1darray[2]
    """
    if not vcount2:
        res = np.random.randint(0, core.vcount(graph), 2) + 1  # @indexing
        return res.astype(graph[1].dtype.type)
    else:
        src = np.random.choice(core.vcount(graph)) + 1  # @indexing
        dst = np.random.choice(vcount2) + 1  # @indexing
        return np.array([src, dst], dtype=graph[1].dtype)
    

def get_random_walk_pair(graph):
    raise NotImplementedError


@numba.njit(cache=True)
@sampler(get_random_edge, get_random_pair)
def get_nce_sample(graph, n, vcount2=0):
    """ Sample edges with balanced noise contrast.
        
        Parameters
        ----------
        graph : tuple
            grl.graph
        n : int
            Sample size. 
        vcount2 : int, optional
            Number of nodes in the second (non-indexed) modality in the graph. 
            For unimodal graphs this value should be 0 (default).

        Returns
        -------
        2darray, 1darray
            Edgelist and targets. 
    """
    pass


@numba.njit(cache=True)
@sampler(get_random_edge, get_random_anti_edge)
def get_neg_sample(graph, n, vcount2=0):
    """ Sample edges with balanced negative contrast.
        
        Parameters
        ----------
        graph : tuple
            grl.graph
        n : int
            Sample size. 
        vcount2 : int, optional
            Number of nodes in the second (non-indexed) modality in the graph. 
            For unimodal graphs this value should be 0 (default).

        Returns
        -------
        2darray, 1darray
            Edgelist and targets. 
    """
    pass


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


# Alias
nce = get_nce_sample
neg = get_neg_sample
