import numpy as np

import grl

from common import igraphs, graphs


def test_degree(igraphs):
    for G in igraphs():
        g = grl.graph.utils.from_igraph(G)
        assert np.all(grl.degree(g) == np.array(G.degree()))


def test_density(igraphs):
    for G in igraphs():
        g = grl.graph.utils.from_igraph(G)
        assert np.allclose(grl.density(g), G.density())


def test_ecount(igraphs):
    for G in igraphs():
        g = grl.graph.utils.from_igraph(G)
        assert G.ecount() == grl.ecount(g)//2  # @symmetry


def test_neighbors(graphs):
    for g in graphs():
        G = grl.graph.utils.to_igraph(g)
        for node in grl.graph.utils.enumerate_nodes(g):
            assert np.all(grl.neighbors(node, g) == np.array(G.neighbors(node-1)) + 1)  # @indexing


def test_vcount(igraphs):
    for G in igraphs():
        g = grl.graph.utils.from_igraph(G)
        assert grl.vcount(g) == G.vcount()
