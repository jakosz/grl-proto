import random

import igraph
import numpy as np
import pytest

import grl


class Namespace:
    pass


@pytest.fixture(scope="module")
def igraphs():

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
    
    return get_igraphs


@pytest.fixture(scope="module")
def graphs():
    
    def get_graphs(seed=13):
        return [
            grl.graph.random.erdos(128, 1e-2, seed=seed),
            grl.graph.random.fire(256, 1e-2, seed=seed),
            grl.graph.random.barabasi(512, 2, seed=seed),
            grl.graph.random.geometric(256, .1, seed=seed)
        ]
    
    return get_graphs


@pytest.fixture(scope="module")
def ogb_dataset():
    
    def wrap(name):
        if name == 'zachary':
            ogb_zachary = np.array([(e.source, e.target) for e in igraph.Graph.Famous('Zachary').es()]).T
            zachary = Namespace()
            zachary.graph = {
                'edge_index': ogb_zachary, 
                'num_nodes': igraph.Graph.Famous('Zachary').vcount()
            }
            return zachary

    return wrap


@pytest.fixture(scope="module")
def random_binomial_2d():

    def wrap(shape=(32, 32), seed=13):
        np.random.seed(seed)
        return np.random.binomial(1, .5, size=shape)

    return wrap


@pytest.fixture(scope="module")
def random_normal_2d():

    def wrap(shape=(32, 32), seed=13):
        np.random.seed(seed)
        return np.random.randn(*shape)

    return wrap
