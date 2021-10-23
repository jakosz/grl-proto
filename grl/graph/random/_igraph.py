import random as _random

import igraph as _igraph

from ..utils import from_igraph as _from_igraph


def barabasi(n, m, seed=None):
    _random.seed(seed)
    g = _igraph.Graph.Barabasi(n, int(m))
    return _from_igraph(g)


def block(n, pref_matrix, block_sizes, seed=None):
    #_random.seed(seed)
    #g = _igraph.Graph.SBM(n, pref_matrix, block_sizes)
    #return _from_igraph(g)
    raise NotImplementedError("pref_matrix and block_sizes need conversions to/fro numpy")


def erdos(n, p, seed=None):
    _random.seed(seed)
    g = _igraph.Graph.Erdos_Renyi(n, p)
    return _from_igraph(g)


def fire(n, fw, bw=0.0, ambs=1, seed=None):
    _random.seed(seed)
    g = _igraph.Graph.Forest_Fire(n, fw_prob=fw, bw_factor=bw, ambs=ambs)
    return _from_igraph(g)


def geometric(n, r, seed=None):
    _random.seed(seed)
    g = _igraph.Graph.GRG(n, r)
    return _from_igraph(g)
