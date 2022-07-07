# Notes
# -----
#
# As of 0.8.37 I have a lot of doubts whether decorators make the 
# whole shabang easier (given the kwargs limitation of numba jit compiler).  
#
import numba
import numpy as np

from . import core
from . import utils
from .. import numby


def sampler(f_positives, f_negatives):
    """ Define graph sampler. 
        
        Parameters
        ----------
        f_positives : function(graph, *pargs)
            A function taking graph and optional arguments and returning 1darray 
            with a random example of an edge.  
        f_negatives : function(graph, *nargs)
            A function taking graph and optional arguments and returning 1darray 
            with a random example of a non-edge. 
        
        Returns
        -------
        function
    """
    def wrap(f):
        def get_sample(graph, n, pargs=(), nargs=()):
            
            X = np.empty((n, 2), dtype=graph[1].dtype)
            n //= 2
            Y = np.hstack((np.ones(n), np.zeros(n)))

            for i in range(n):
                X[i] = f_positives(graph, *pargs)

            for i in range(n):
                X[n+i] = f_negatives(graph, *nargs)

            # shuffle
            srt = np.arange(n*2)
            np.random.shuffle(srt)
            X, Y = X[srt], Y[srt]

            return X, Y
        # propagate name & docstring of the wrapped function
        get_sample.__name__ = f.__name__
        get_sample.__doc__ = f.__doc__
        return get_sample
    return wrap


def with_mask(edge_sampler):
    """ Modify edge sampler to return unmasked examples.
        Mask should be the last positional argument to the wrapped function.
    """
    def wrap(dummy):
        @numba.njit()
        def wrapd(graph, *args):
            fargs, mask = args[:-1], args[-1]
            edge = edge_sampler(graph, *fargs)
            if utils.is_edge_masked(edge, graph, mask):
                return wrapd(graph, *args)
            else:
                return edge
        # propagate name & docstring of the wrapped function
        wrapd.__name__ = dummy.__name__
        wrapd.__doc__ = dummy.__doc__
        return wrapd
    return wrap


# caching is disabled on purpose; there's a weird bug somewhere - enabling caching 
# IN THIS SPECIFIC FUNCTION will sometimes make Python crash
@numba.njit()  
def get_random_anti_edge(graph, vcount2=0):
    """ Sample a random nonexistent edge. """
    if not vcount2:
        vs = np.random.choice(core.vcount(graph), 16) + 1  # @indexing
        src, dst = vs[0], vs[1:]
        nbs = core.neighbors(src, graph)
        for v in dst:
            if not numby.isin_1d(v, nbs):
                return np.array([src, v], dtype=graph[1].dtype)
    else:
        src = np.random.choice(core.vcount(graph)) + 1  # @indexing
        dst = np.random.choice(vcount2, 16) + 1  # @indexing
        nbs = core.neighbors(src, graph)
        for v in dst:
            if not numby.isin_1d(v, nbs):
                return np.array([src, v], dtype=graph[1].dtype)
    return get_random_anti_edge(graph, vcount2)


@numba.njit(cache=True)
def get_random_edge(graph):
    """ Sample a random existing edge. """
    src = np.random.choice(core.vcount(graph)) + 1  # @indexing
    dst = np.random.choice(core.neighbors(src, graph))
    if not dst:
        return get_random_edge(graph)
    return np.array([src, dst], dtype=graph[1].dtype)


@with_mask(get_random_edge)
def get_random_edge_with_mask():
    pass


@numba.njit(cache=True)
def _alt_get_random_edge_with_mask(graph, mask):
    """ Sample a random existing edge allowed by mask
        (i.e. if its corresponding mask value is 1).
    """
    v, e = graph
    #v = v.astype(np.int64)  # @indexing
    src = np.random.choice(core.vcount(graph)) + 1  # @indexing
    nb_addr = utils.addr_neighbors(src, graph)
    nb_addr = nb_addr[mask[nb_addr] == 1]
    if nb_addr.size == 0:
        return get_random_edge_with_mask(graph, mask)
    dst = np.random.choice(e[nb_addr])
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
    

@numba.njit(cache=True)
def get_random_walk_pair(graph, walk_length):
    v = np.random.choice(grl.vcount(graph)) + 1  # @indexing
    w = grl.graph.sample.random_walk(v, graph, walk_length).copy()
    np.random.shuffle(w)
    return w[:2]


@with_mask(get_random_walk_pair)
def get_random_walk_pair_with_mask():
    pass


@numba.njit(cache=True)
@sampler(get_random_edge, get_random_pair)
def get_nce_sample():
    """ Sample edges with balanced noise contrast.
        
        Parameters
        ----------
        graph : tuple
            grl.graph
        n : int
            Sample size. 
        nargs : tuple, optional
            Positional arguments to the contrastive sampler:
                (vcount2,)
            Defaults to:
                (0,)
            Details:
                vcount2 : int
                    Number of nodes in the second (non-indexed) modality in the graph. 
                    For unimodal graphs this value should be 0 (default).

        Returns
        -------
        2darray, 1darray
            Edgelist and targets. 
    """
    pass


@numba.njit(cache=True)
@sampler(get_random_edge_with_mask, get_random_pair)
def get_nce_sample_with_mask():
    """ Draw a graph sample where positive node pairs are 
        neighbors (k=1) allowed by the given mask, and negative pairs 
        are drawn at random (i.e. this sampler uses a uniform noise 
        contrast).
        
        Parameters
        ----------
        graph : tuple
            A grl graph.
        n : int
            Sample size.
        pargs : tuple
            Positional arguments to the random edge sampler:
                (mask,)
        nargs : tuple, optional
            Positional arguments to the contrastive sampler:
                (vcount2,)
            Defaults to:
                (0,)
            Details:
                vcount2 : int
                    Number of nodes in the second (non-indexed) modality in the graph. 
                    For unimodal graphs this value should be 0 (default).
                
        Returns
        -------
        result : (2darray, 1darray)
        
        Notes
        -----
        
    """    
    pass


@numba.njit(cache=True)
@sampler(get_random_edge, get_random_anti_edge)
def get_neg_sample():
    """ Sample edges with balanced negative contrast.
        
        Parameters
        ----------
        graph : tuple
            grl.graph
        n : int
            Sample size. 
        nargs : tuple, optional
            Positional arguments to the contrastive sampler:
                (vcount2,)
            Defaults to:
                (0,)
            Details:
                vcount2 : int
                    Number of nodes in the second (non-indexed) modality in the graph. 
                    For unimodal graphs this value should be 0 (default).

        Returns
        -------
        2darray, 1darray
            Edgelist and targets. 
    """
    pass


@numba.njit(cache=True)
@sampler(get_random_walk_pair, get_random_pair)
def get_random_walk_sample():
    """ Draw a graph sample where positive node pairs are taken
        from random walks of given length, and negative pairs are 
        drawn at random (i.e. this sampler uses a uniform noise 
        contrast).
        
        Parameters
        ----------
        graph : tuple
            A grl graph.
        n : int
            Sample size.
        pargs : tuple
            Positional arguments to the random walk sampler:
                (walk_length,)
        nargs : tuple, optional
            Positional arguments to the contrastive sampler:
                (vcount2,)
            Defaults to:
                (0,)
                
        Returns
        -------
        result : (2darray, 1darray)
        
        Notes
        -----
        
    """
    pass


@numba.njit(cache=True)
@sampler(get_random_walk_pair_with_mask, get_random_pair)
def get_random_walk_sample_with_mask():
    """ Draw a graph sample where positive node pairs are taken
        from random walks of given length, and negative pairs are 
        drawn at random (i.e. this sampler uses a uniform noise 
        contrast).
        
        Parameters
        ----------
        graph : tuple
            A grl graph.
        n : int
            Sample size.
        pargs : tuple
            Positional arguments to the random walk sampler:
                (walk_length, mask,)
        nargs : tuple, optional
            Positional arguments to the contrastive sampler:
                (vcount2,)
            Defaults to:
                (0,)
                
        Returns
        -------
        result : (2darray, 1darray)
        
        Notes
        -----
        
    """
    pass


@numba.njit(cache=True)
def _random_uniform_walk(vi, graph, length, data=None, step=0):
    if data is None:
        data = np.zeros(length, dtype=graph[1].dtype.type)
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
