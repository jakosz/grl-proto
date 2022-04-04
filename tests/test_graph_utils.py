import igraph
import numpy as np

import grl

from common import igraphs, graphs, ogb_dataset


def test_enumerate_edges(graphs):
    for G in graphs():
        assert grl.graph.utils.enumerate_edges(G).shape[0] == grl.graph.core.ecount(G)


def test_enumerate_nodes(graphs):
    for G in graphs():
        assert grl.graph.utils.enumerate_nodes(G).shape[0] == grl.graph.core.vcount(G)


def test_to_from_adjacency(graphs):
    for G in graphs():
        A = grl.graph.utils.to_adjacency(G)
        H = grl.graph.utils.from_adjacency(A)
        assert np.all(G[0] == H[0])
        assert np.all(G[1] == G[1])


def test_to_from_igraph(igraphs):
    for G in igraphs():
        g = grl.graph.utils.from_igraph(G)
        H = grl.graph.utils.to_igraph(g)
        h = grl.graph.utils.from_igraph(H)
        assert np.all(g[0] == h[0])
        assert np.all(g[1] == h[1])


def test_from_ogb_degree(ogb_dataset):
    
    dataset = ogb_dataset('zachary')
    graph = grl.graph.utils.from_ogb(dataset)
    ogb_edges = dataset.graph['edge_index'].T
        
    es = ogb_edges
    se = np.hstack([es[:, 1:], es[:, :1]])  # @symmetry
    es = np.unique(np.vstack([es, se]), axis=0)  # @indexing @slow

    assert np.all(grl.degree(graph) == np.bincount(es.ravel())//2), "degree vectors do not match"


def test_from_ogb_igraph(ogb_dataset):
    
    dataset = ogb_dataset('zachary')
    graph = igraph.Graph.Famous('Zachary')
    
    d = grl.graph.utils.from_ogb(dataset)
    g = grl.graph.utils.from_igraph(graph)
    
    assert grl.graph.utils.hexdigest(d) == grl.graph.utils.hexdigest(g)


def test_from_ogb_neighbors(ogb_dataset):
    
    dataset = ogb_dataset('zachary')
    graph = grl.graph.utils.from_ogb(dataset)
    ogb_edges = dataset.graph['edge_index'].T

    for ogb_index in range(dataset.graph["num_nodes"]):
        grl_index = ogb_index + 1  # @indexing

        fl = ogb_edges[:, 0] == ogb_index
        fr = ogb_edges[:, 1] == ogb_index
        nb = ogb_edges[np.logical_or(fl, fr)]
        nb = np.unique(nb[nb != ogb_index])

        assert grl.neighbors(grl_index, graph).shape[0] == nb.shape[0], "number of neighbors does not match"
        
        nbs = np.unique(nb.ravel())
        nbs = nbs[nbs != ogb_index]

        assert np.all(np.sort(grl.neighbors(grl_index, graph)) == np.sort(nbs) + 1), "neighbour set does not match"


def test_from_ogb_vcount(ogb_dataset):
    
    dataset = ogb_dataset('zachary')
    graph = grl.graph.utils.from_ogb(dataset)
    ogb_edges = dataset.graph['edge_index'].T
    
    assert grl.vcount(graph) == dataset.graph['num_nodes'], "node counts do not match"
