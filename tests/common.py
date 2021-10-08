import random

import igraph
import pytest

import grl


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
