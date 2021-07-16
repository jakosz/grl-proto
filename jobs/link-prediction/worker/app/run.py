import random

import igraph
import numpy as np

import grl.graph as sd
from grl.utils.log import get_stdout_logger

import config


log = get_stdout_logger('link-prediction-worker')

def get_rgm(obs, methods):

    rg = lambda x: getattr(igraph.Graph, x)
    method = random.choice(list(methods.keys()))
    seed = np.random.randint(2**32-1)
    obs = np.random.choice(obs)
    
    if method == 'Barabasi':
        m = random.choice(methods[method]['m'])
        random.seed(seed)
        G = rg(method)(obs, m)
        graph = sd.utils.from_igraph(G)
        sig = {'name': method, 'n': obs, 'm': m, 'seed': seed}
    elif method == 'Erdos_Renyi':
        p = np.random.uniform(*methods['Erdos_Renyi']['p'])
        random.seed(seed)
        G = rg(method)(obs, p)
        graph = sd.utils.from_igraph(G)
        sig = {'name': method, 'n': obs, 'p': p, 'seed': seed}
    elif method == 'Forest_Fire':
        fw = np.random.uniform(*methods[method]['fw'])
        ambs = np.random.choice(methods[method]['ambs'])
        random.seed(seed)
        G = rg(method)(obs, fw_prob=fw, ambs=ambs)
        graph = sd.utils.from_igraph(G)
        sig = {'name': method, 'n': obs, 'fw': fw, 'bw': 0.0, 'ambs': ambs, 'seed': seed}
    else: # GRG
        radius = np.random.uniform(*methods[method]['radius'])
        random.seed(seed)
        G = rg(method)(obs, radius)
        graph = sd.utils.from_igraph(G)
        sig = {'name': method, 'n': obs, 'radius': radius, 'seed': seed}
        
    return G, graph, sig


G, graph, sig = get_rgm(np.arange(config.VCOUNT_FROM, config.VCOUNT_TO), config.RGM_SAMPLING_SPACE)
obs = int(sig['n'])
nnb = [1, int(np.quantile(G.degree(), .9))]
dims = config.DIM_RANGE 
steps = config.TRAINING_STEPS
batch_size = config.BATCH_SIZE

log.info(sig)
