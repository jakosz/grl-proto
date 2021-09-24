import igraph
import numba
import numpy as np

from .mem import *


def from_adjacency(a):
    raise NotImplementedError


def from_igraph(g):
    """ Convert igraph.Graph to [[name]] graph representation.
        Note: currently it assumes a symmetric graph to be passed. 
    """
    nodes = zeros((g.vcount()+2,), dtype=np.uint64)
    edges = zeros((g.ecount()*2,), dtype=np.uint32)
    offset = 0
    for v in g.vs:
        i = v.index
        vnb = np.array(g.neighbors(v), dtype=np.uint32) + 1 # 1-index
        degree = vnb.shape[0]
        nodes[i+1] = offset # 1-index
        edges[offset:offset+degree] = vnb
        offset += degree
    nodes[-1] = offset
    return nodes, edges


def from_ogb(g):
    """ Convert Open Graph Benchmark library-agnostic format to [[name]]. 
    """
    raise NotImplementedError


@numba.njit(parallel=True)
def to_adjacency(graph):
    n = grl.graph.core.vcount(graph)
    A = np.zeros((n, n), dtype=np.uint8)
    for i in numba.prange(n):
        for j in grl.graph.core.neighbours(i+1, graph)-1:
            A[i, j] = 1
    return A


def to_igraph(g):
    raise NotImplementedError


def to_ogb(g):
    raise NotImplementedError
