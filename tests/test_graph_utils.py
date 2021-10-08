import numpy as np

import grl

from common import igraphs, graphs


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
