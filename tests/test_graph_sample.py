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
        positive = edgelist[:1024]
        negative = edgelist[1024:]
        for i in range(negative.shape[0]):
            assert positive[i, 1] in grl.neighbours(positive[i, 0], G)
            assert positive[i, 0] in grl.neighbours(positive[i, 1], G)    
            assert not negative[i, 1] in grl.neighbours(negative[i, 0], G)
            assert not negative[i, 0] in grl.neighbours(negative[i, 1], G)
