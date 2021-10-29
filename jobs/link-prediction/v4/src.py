import numpy as np
from scipy.stats import entropy
from sklearn.metrics import roc_auc_score

import grl


class Namespace:
    pass


class RunningAverage:
    def __init__(self, n=100):
        self.count = 0
        self.n = n
        self.p = 1/n
        self.value = 0.

    def __call__(self, x):
        """ Update with new value and return average."""
        self.count += 1
        p = 1/min(self.count, self.n)
        self.value = self.value*(1-p) + x*p
        return self.value


def properties(g):
    a = grl.graph.utils.to_adjacency(g)
    g = grl.graph.utils.to_igraph(g)
    return {
        'assortativity_degree': g.assortativity_degree(),
        'average_path_length': g.average_path_length(),
        'clique_number': g.clique_number(),
        'diameter': g.diameter(),
        'density': g.density(),
        'edge_connectivity': g.edge_connectivity(),
        'entropy': entropy(a.ravel(), base=2),
        'girth': g.girth(),
        'independence_number': g.independence_number(),
        'is_bipartite': g.is_bipartite(),
        'is_chordal': g.is_chordal(),
        'is_connected': g.is_connected(),
        'maxdegree': g.maxdegree(),
        'motifs_randesu_3': np.array(g.motifs_randesu(3)),
        'motifs_randesu_4': np.array(g.motifs_randesu(4)),
        'transitivity_undirected': g.transitivity_undirected(),
        'vertex_connectivity': g.vertex_connectivity()
    }


def barabasi():
    return grl.graph.random.barabasi(config.vcount, config.range_barabasi())


def erdos():
    return grl.graph.random.erdos(config.vcount, config.range_erdos())


def geometric():
    return grl.graph.random.geometric(config.vcount, config.range_geometric())


def eigen(graph, dim):
    a = grl.graph.utils.to_adjacency(graph)
    l, v = grl.graph.encode.eigen(graph)
    y = grl.graph.decode.eigen(l, v, dim)
    return roc_auc_score(a.ravel(), y.ravel())


def diagonal(graph, dim):
    a = grl.graph.utils.to_adjacency(graph)
    x, d = grl.graph.encode.diagonal(graph, dim, config.iter, config.lr)
    y = grl.graph.decode.diagonal(x, d)
    return roc_auc_score(a.ravel(), y.ravel())


def symmetric(graph, dim):
    a = grl.graph.utils.to_adjacency(graph)
    x = grl.graph.encode.symmetric(graph, dim, config.iter, config.lr)
    y = grl.graph.decode.symmetric(x)
    return roc_auc_score(a.ravel(), y.ravel())


def asymmetric(graph, dim):
    a = grl.graph.utils.to_adjacency(graph)
    l, r = grl.graph.encode.asymmetric(graph, dim, config.iter, config.lr)
    y = grl.graph.decode.asymmetric(l, r)
    return roc_auc_score(a.ravel(), y.ravel())


def diagonal_st(graph, dim):
    a = grl.graph.utils.to_adjacency(graph)
    x, d = grl.graph.embed.diagonal.encode_st(graph, dim, config.iter, config.lr)
    y = grl.graph.decode.diagonal(x, d)
    return roc_auc_score(a.ravel(), y.ravel())


def symmetric_st(graph, dim):
    a = grl.graph.utils.to_adjacency(graph)
    x = grl.graph.embed.symmetric.encode_st(graph, dim, config.iter, config.lr)
    y = grl.graph.decode.symmetric(x)
    return roc_auc_score(a.ravel(), y.ravel())


def asymmetric_st(graph, dim):
    a = grl.graph.utils.to_adjacency(graph)
    l, r = grl.graph.embed.asymmetric.encode_st(graph, dim, config.iter, config.lr)
    y = grl.graph.decode.asymmetric(l, r)
    return roc_auc_score(a.ravel(), y.ravel())

embs = {
    'diagonal': diagonal, 
    'symmetric': symmetric, 
    'asymmetric': asymmetric,
}
    
embs_st = {
    'diagonal': diagonal_st, 
    'symmetric': symmetric_st, 
    'asymmetric': asymmetric_st
}

rgms = {
    'barabasi': barabasi, 
    'erdos': erdos, 
    'geometric': geometric
}

config = Namespace()
config.vcount = None  # passed from the caller
config.iter = 2**21 
config.lr = .025
config.range_barabasi = lambda: np.random.choice(np.arange(2, config.vcount))
config.range_erdos = lambda: np.random.uniform()
config.range_geometric = lambda: np.random.uniform()
