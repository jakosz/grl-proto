import json

import igraph
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.collections import LineCollection
from scipy.spatial.distance import squareform
from tensorflow.keras.layers import *


class JsonNumpy(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.int32) or isinstance(obj, np.int64) or isinstance(obj, np.uint32) or isinstance(obj, np.uint64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


def gtoa(g):
    return np.array(list(g.get_adjacency()), dtype=np.uint8)

def atog(a):
    return igraph.Graph.Adjacency(a.tolist())

# atog/gtoa wrappers for single-arg functions

def xa(f):
    def wrap(x, *args, **kwargs):
        if type(x) is not np.ndarray:
            x = gtoa(x)
        return f(x, *args, **kwargs)
    return wrap
    
def xg(f):
    def wrap(x, *args, **kwargs):
        if type(x) is np.ndarray:
            x = atog(x, *args, **kwargs)
        return f(x)
    return wrap

@xg
def _edgelist_igraph(x):
    return np.array([(e.source, e.target) for e in x.es])

@xa
def _edgelist_numpy(x):
    return np.vstack(np.where(x)).T

@xa
def anti_edgelist(x):
    return np.vstack(np.where(x == 0)).T

edgelist = _edgelist_numpy

def get_nce_sample(x, bs=1024):
    bs = bs//2
    E = edgelist(x)
    y = np.hstack([np.ones(bs), np.zeros(bs)])
    x = np.vstack([
        E[np.random.choice(E.shape[0], bs)], 
        np.random.choice(vcount(x), size=(bs, 2))
    ])
    return [x[:, 0], x[:, 1]], y

def get_neg_sample(x, bs=1024):
    bs = bs//2
    E = edgelist(x)
    nE = anti_edgelist(x)
    y = np.hstack([np.ones(bs), np.zeros(bs)])
    x = np.vstack([
        E[np.random.choice(E.shape[0], bs)], 
        nE[np.random.choice(nE.shape[0], bs)]
    ])
    return [x[:, 0], x[:, 1]], y

def get_adj(n, p=.5):
    return squareform(np.random.binomial(1, p, size=np.sum(np.arange(1, n))).astype(np.uint8))

def get_graph(n, p=.5):
    return atog(get_adj(n, p))

def vcount(x):
    if type(x) is np.ndarray:
        return x.shape[0]
    else:
        return x.vcount()

def fr(x):
    if type(x) is np.ndarray:
        x = atog(x)
    return np.array(x.layout_fruchterman_reingold().coords)


def plot_graph(g, xy=None):
    if xy is None:
        xy = fr(g)
    E = edgelist(g)
    plt.scatter(xy[:, 0], xy[:, 1])
    plt.gca().add_collection(LineCollection([xy[e].tolist() for e in E]))

def get_graph_conn(n, p):
    G = get_graph(n, p)
    while not G.is_connected():
        G = get_graph(n, p)
    return G
        
def get_adj_conn(n, p):
    return gtoa(get_graph_conn(n, p))

def get_a4(i):
    return squareform([int(e) for e in np.binary_repr(i, 6)])

def get_a5(i):
    return squareform([int(e) for e in np.binary_repr(i, 10)])

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x)) 
