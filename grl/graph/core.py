""" The ``graph.core`` module defines basic operations on grl's graph `data structure`.  
Functions defined here are exported at the package level. 

Graph structure representation
==============================

Grl represents graph as a 2-tuple of 1-dimensional numpy arrays, 
corresponding to nodes and edges.   

The `nodes` array is of type ``uint64`` and length `n+2`, where `n` is the number of nodes in the graph.
It contains indices of the `edges` array laid out as follows:
    - its `i`-th element contains the index in the `edges` array where the list of neighbours of `i`-th node `begins`
    - its `i+1`-th element marks the end of the neighbour list of `i`-th node
    - if `i`-th node has no neighbours, ``nodes[i] == nodes[i+1]``
    - its last element is the size of the `edges` array. 
    - its first element is always 0. 

The `edges` array is of type ``uint32`` and length `2m`, where `m` is the number 
of edges in the graph (currently only symmetric graphs are supported). 
It contains indices of the `nodes` array corresponding to node neighbourhoods as described above. 

Note that this implies that nodes are 1-indexed (in some scenarios it was useful 
to have a meaningless embedding at index 0; this might change at any time), and
the maximum supported number of nodes is 2^32. 

All functions in this module have to be compiled with numba in nopython mode, 
so that higher-level functions also compile this way.  
"""

import numba
import numpy as np


@numba.njit(cache=True)
def degree(graph):
    """ Get degrees of nodes in a graph.
        
        Parameters
        ----------
        graph : grl.Graph
            Input graph.

        Returns
        -------
        result : 1darray
            Degree vector of the graph. 
    """
    v, e = graph
    return (v[1:] - v[:-1])[1:]  # @indexing


@numba.njit(cache=True)
def density(graph):
    """ Get density of a graph. 

        Parameters
        ----------
        graph : grl.Graph
            Input graph.

        Returns
        -------
        result : float 
            Graph density.
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
