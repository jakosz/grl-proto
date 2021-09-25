import random

import igraph

from ..utils import from_igraph


def barabasi(n, m, seed=None):
    random.seed(seed)
    g = igraph.Graph.Barabasi(n, m)
    return from_igraph(g)


def block(n, pref_matrix, block_sizes, seed=None):
    #random.seed(seed)
    #g = igraph.Graph.SBM(n, pref_matrix, block_sizes)
    #return from_igraph(g)
    raise NotImplementedError("pref_matrix and block_sizes need conversions to/fro numpy")


def erdos(n, p, seed=None):
    random.seed(seed)
    g = igraph.Graph.Erdos_Renyi(n, p)
    return from_igraph(g)


def fire(n, fw, bw=0.0, ambs=1, seed=None):
    random.seed(seed)
    g = igraph.Graph.Forest_Fire(n, fw_prob=fw, bw_factor=bw, ambs=ambs)
    return from_igraph(g)


def geometric(n, r, seed=None):
    random.seed(seed)
    g = igraph.Graph.GRG(n, r)
    return from_igraph(g)

