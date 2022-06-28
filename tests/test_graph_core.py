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


def test_induced_subgraph(graphs):
    for g in graphs():
        # test that induced subgraph is identical to the input graph
        # when given the full vertex set
        vs = grl.graph.utils.enumerate_nodes(g)
        sg = grl.subgraph(vs, g)
        assert np.all(sg[0] == g[0])
        assert np.all(sg[1] == g[1])
        # test that the intersection of neighbours between the input graph
        # and induced subgraph are the same when node indexing is preserved
        offset = grl.vcount(g)//2
        vs = np.arange(1, offset)
        sg = grl.subgraph(vs, g)
        for v in vs:
            nb_g = grl.neighbors(v, g)
            nb_sg = grl.neighbors(v, sg)
            assert np.all(nb_sg == np.intersect1d(nb_g, nb_sg))
        # test that reindexing works as expected by extracting induced subgraph
        # for a sequence of nodes offset by a constant, and then comparing the 
        # intersection of neighbours in the input graph and induced subgraph 
        vs = np.arange(1, offset) + offset 
	sg = grl.subgraph(vs, g)
	for v in vs:
	    nb_g = grl.neighbors(v, g)
	    nb_sg = grl.neighbors(v - offset, sg)
	    assert np.all(nb_sg == np.intersect1d(nb_g - offset, nb_sg))


def test_neighbors(graphs):
    for g in graphs():
        G = grl.graph.utils.to_igraph(g)
        for node in grl.graph.utils.enumerate_nodes(g):
            assert np.all(grl.neighbors(node, g) == np.array(G.neighbors(node-1)) + 1)  # @indexing


def test_neighbors_out_of_index_error(graphs):
    for g in graphs():
        with pytest.raises(IndexError):
            grl.neighbors(grl.vcount(g)+1, g)


def test_vcount(igraphs):
    for G in igraphs():
        g = grl.graph.utils.from_igraph(G)
        assert grl.vcount(g) == G.vcount()
