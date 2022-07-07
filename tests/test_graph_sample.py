import numpy as np

import grl

from common import igraphs, graphs


def test_get_nce_sample():
    """ Take a large noise contrast from a small graph
        and assert all possible combinations are there.
    """
    G = grl.graph.random.erdos(4, .5, 13)
    edgelist = grl.graph.sample.nce(G, 2048)[0]
    assert np.unique(edgelist[1024:], axis=0).shape[0] == 16


def test_get_neg_sample(graphs):
    for G in graphs():
        edgelist, y = grl.graph.sample.neg(G, 2048)
        edgelist = edgelist[np.argsort(y)]
        positive = edgelist[1024:]
        negative = edgelist[:1024]
        for i in range(negative.shape[0]):
            assert positive[i, 1] in grl.neighbors(positive[i, 0], G)
            assert positive[i, 0] in grl.neighbors(positive[i, 1], G)    
            assert not negative[i, 1] in grl.neighbors(negative[i, 0], G)
            assert not negative[i, 0] in grl.neighbors(negative[i, 1], G)


def test_get_random_edge_with_mask(graphs):
    for graph in graphs:
        mask = grl.graph.utils.get_edge_mask(.2, graph)
        
        samples = []
        for _ in range(grl.ecount(graph)*10):
            samples.append(grl.graph.sample.get_random_edge_with_mask(graph, mask))
        
        samples = np.unique(np.vstack(samples), axis=0)
        truth = grl.graph.utils.to_edgelist(g)[mask.astype(bool)]
        assert np.all(samples == truth)
