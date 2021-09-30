import random

import igraph
import numpy as np

import grl


def get_igraphs(seed=13):
    igraphs = []
    random.seed(seed)
    igraphs.append(igraph.Graph.Erdos_Renyi(128, 1e-2))
    random.seed(seed)
    igraphs.append(igraph.Graph.Forest_Fire(256, 1e-2)) 
    random.seed(seed)
    igraphs.append(igraph.Graph.Barabasi(512, 2)) 
    random.seed(seed)
    igraphs.append(igraph.Graph.GRG(256, .1))
    return igraphs


def get_graphs(seed=13):
    return [
        grl.graph.random.erdos(128, 1e-2, seed=seed),
        grl.graph.random.fire(256, 1e-2, seed=seed),
        grl.graph.random.barabasi(512, 2, seed=seed),
        grl.graph.random.geometric(256, .1, seed=seed)
    ]


# --- core

def test_degree():
    for G in get_igraphs():
        g = grl.graph.utils.from_igraph(G)
        assert np.all(grl.degree(g) == np.array(G.degree()))


def test_vcount():
    for G in get_igraphs():
        g = grl.graph.utils.from_igraph(G)
        assert grl.vcount(g) == G.vcount()


# --- sample 

def test_get_neg_sample():
    for G in get_graphs():
        edgelist = grl.graph.sample.neg(G, 2048)[0]
        positive = edgelist[:1024]
        negative = edgelist[1024:]
        for i in range(negative.shape[0]):
            assert positive[i, 1] in grl.neighbours(positive[i, 0], G)
            assert positive[i, 0] in grl.neighbours(positive[i, 1], G)    
            assert not negative[i, 1] in grl.neighbours(negative[i, 0], G)
            assert not negative[i, 0] in grl.neighbours(negative[i, 1], G)


# --- utils

def test_to_from_adjacency():
    for G in get_graphs():
        A = grl.graph.utils.to_adjacency(G)
        H = grl.graph.utils.from_adjacency(A)
        assert np.all(G[0] == H[0])
        assert np.all(G[1] == G[1])


def test_to_from_igraph():
    for G in get_igraphs():
        g = grl.graph.utils.from_igraph(G)
        H = grl.graph.utils.to_igraph(g)
        h = grl.graph.utils.from_igraph(H)
        assert np.all(g[0] == h[0])
        assert np.all(g[1] == h[1])
