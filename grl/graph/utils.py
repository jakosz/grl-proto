import hashlib
import pickle

import igraph
import numba
import numpy as np

from . import core
from ..shmem import *


def digest(graph):
    return hashlib.sha256(b"".join([e.tobytes() for e in graph])).digest()


@numba.njit(cache=True)
def enumerate_edges(graph):
    res = np.empty((core.ecount(graph), 2), dtype=graph[1].dtype)
    i = 0
    for src in enumerate_nodes(graph):
        nbs = core.neighbours(src, graph)
        if nbs[0]:  # core.neighbours returns array([0]) for isolates
            for dst in nbs:
                res[i, 0] = src
                res[i, 1] = dst
                i += 1
    return res


@numba.njit(cache=True)
def enumerate_nodes(graph):
    return (np.arange(core.vcount(graph))+1).astype(graph[1].dtype)


@numba.njit(cache=True)
def enumerate_without(graph, subset):
    """ Enumerate nodes and remove given subset from the enumeration. 
    """
    nodes = enumerate_nodes(graph)
    for i in subset-1:  # @indexing
        nodes[i] = 0
    return nodes[nodes != 0]


@numba.njit(cache=True)
def from_adjacency(A):
    """ Convert adjacency matrix to grl graph representation.
    """
    nodes = np.zeros(A.shape[0]+2, dtype=np.uint64)
    edges = np.zeros(A.sum(), dtype=np.uint32)
    cnt = 0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j]:
                edges[cnt] = j + 1  # @indexing
                cnt += 1
        nodes[i+2] = cnt
    return nodes, edges


def from_igraph(g):
    """ Convert igraph.Graph to grl graph representation.
        Note: currently it assumes a symmetric graph to be passed. 
    """
    nodes = np.zeros((g.vcount()+2,), dtype=np.uint64)
    edges = np.zeros((g.ecount()*2,), dtype=np.uint32)
    offset = 0
    for v in g.vs:
        i = v.index
        vnb = np.array(g.neighbors(v), dtype=np.uint32) + 1  # @indexing 
        degree = vnb.shape[0]
        nodes[i+1] = offset  # @indexing 
        edges[offset:offset+degree] = vnb
        offset += degree
    nodes[-1] = offset
    return nodes, edges


def from_ogb(dataset):
    """ Convert Open Graph Benchmark library-agnostic format to grl.
        @todo:numba
    """
    e0 = dataset.graph['edge_index'].T
    e1 = np.hstack([e0[:, 1:], e0[:, :1]])  # @symmetry
    es = np.unique(np.vstack([e0, e1]), axis=0) + 1  # @indexing @slow

    vcount = dataset.graph['num_nodes']
    ecount = es.shape[0]

    diff = np.diff(es[:, 0])
    assert np.all(np.unique(diff) == np.array([0, 1])),  "edgelist not sorted"
    ix = np.hstack([0, np.where(diff)[0] + 1, ecount])  # node boundaries

    nodes = np.zeros((vcount+2,), dtype=np.uint64)
    edges = np.zeros((ecount,), dtype=np.uint32)
    offset = 0

    for i in range(ix.shape[0]-1):
        nb = es[ix[i]:ix[i+1], 1]
        degree = nb.shape[0]
        nodes[i+1] = offset  # @indexing
        edges[offset:offset+degree] = nb
        offset += degree
    nodes[-1] = offset
    
    return nodes, edges 


def hexdigest(graph):
    return digest(graph).hex()


@numba.njit(cache=True, parallel=True)
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


# --- ?

def save(path, graph):
    with open(path, 'wb') as f:
        f.write(pickle.dumps(graph))


def load(path):
    with open(path, 'rb') as f:
        return pickle.loads(f.read())
