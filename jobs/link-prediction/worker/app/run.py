#!/usr/local/bin/python3

import base64
import json
import random

import igraph
import numpy as np
from sklearn.metrics import roc_auc_score

import grl.graph as sd
from grl.models import *
from grl.utils import *
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


log.info('Starting job...')
log.info('Configuration...')

G, graph, sig = get_rgm(np.arange(config.VCOUNT_FROM, config.VCOUNT_TO), config.RGM_SAMPLING_SPACE)
obs = int(sig['n'])
nnb = [1, int(np.quantile(G.degree(), .9))]
dims = config.DIM_RANGE 
steps = config.TRAINING_STEPS
batch_size = config.BATCH_SIZE

log.info(f'Generated graph from {sig}')

# eigen

a = gtoa(G)
val, vec = np.linalg.eig(a)
val = np.real(val)
vec = np.real(vec)
for dim in dims:
    yhat = vec[:, :dim].dot(np.diag(val[:dim])).dot(vec.T[:dim, :])

    res = {
        'symmetric': True,
        'diagonal': True,
        'dim': int(dim),
        'reducer': 'eig',
        'max_nb': None,
        'loss': float(tf.keras.losses.binary_crossentropy(a, yhat).numpy().mean()),
        'acc': float(tf.keras.metrics.binary_accuracy(a, yhat).numpy().mean()),
        'auc': float(roc_auc_score(a.ravel(), yhat.ravel())),
        'rgm': sig,
        'batch_size': None,
        'steps': None
    }

    log.info(res)
    log.info(f"Network request would look like this: http://server/{base64.b64encode(json.dumps(res, separators=(',',':')).encode())}")

res = []

for sym in [True, False]:
    for diag in [True, False]:
        for dim in dims:
            
            reducers = {
                'mean': lambda x: tf.reduce_mean(x, axis=-2),
                'gnn': GNNSimple(dim, n_layers=1, activation=None),
                'gnn-gelu': GNNSimple(dim, n_layers=1, activation=tf.nn.gelu),
                'gat-h1': GAT(dim, n_layers=1, n_heads=1),
                'gat-h2': GAT(dim, n_layers=1, n_heads=2)
            }
            
            for name, reducer in reducers.items():
                for max_nb in nnb:
                    model, latent = get_model(obs+1, dim, max_nb=max_nb, symmetric=sym, diagonal=diag, reducer=reducer)
                    hist = []
                    if max_nb == 1:
                        for step in range(steps):
                            x, y = get_nce_sample(G)
                            x = [e+1 for e in x]
                            hist.append(model.train_on_batch(x, y))
                    else:
                        for step in range(steps):
                            x, y = get_training_sample(G, graph, max_nb, batch_size)
                            hist.append(model.train_on_batch(x, y))
                    hist = np.array(hist)
                    
                    res = {
                        'symmetric': sym,
                        'diagonal': diag,
                        'dim': int(dim),
                        'reducer': name,
                        'max_nb': max_nb,
                        'loss': float(hist[-100:, 0].mean()),
                        'acc': float(hist[-100:, 1].mean()),
                        'auc': float(hist[-100:, 2].mean()),
                        'rgm': sig,
                        'batch_size': batch_size,
                        'steps': steps
                    }
    
                    log.info(res)
                    log.info(f"Network request would look like this: http://server/{base64.b64encode(json.dumps(res, separators=(',',':')).encode())}")
