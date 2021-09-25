import igraph
import numba
import numpy as np

from . import core
from .mem import *


@numba.njit()
def from_adjacency(A):
    """ Convert adjacency matrix to [[name]] graph representation.
    """
    nodes = np.zeros(A.shape[0]+2, dtype=np.uint64)
    edges = np.zeros(A.sum(), dtype=np.uint32)
    cnt = 0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j]:
                edges[cnt] = j + 1
                cnt += 1
        nodes[i+2] = cnt
    return nodes, edges


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
    n = core.vcount(graph)
    A = np.zeros((n, n), dtype=np.uint8)
    for i in numba.prange(n):
        for j in core.neighbours(i+1, graph):
            if j:
                A[i, j-1] = 1
    return A


def to_igraph(g):
    n = core.vcount(g)
    Graph = igraph.Graph(n)
    for src in range(n):
        for dst in core.neighbours(src+1, g):
            if dst > src:
                Graph.add_edge(src, dst-1)
    return Graph


def to_ogb(g):
    raise NotImplementedError
