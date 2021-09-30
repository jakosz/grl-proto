#!/usr/local/bin/python3

import argparse
import base64
import json
import random
import sys
import urllib.request

import igraph
import numba
import numpy as np
from sklearn.metrics import roc_auc_score

import grl.graph as sd
import grl.layers
from grl.models import *
from grl.utils import *
from grl.utils.log import get_stdout_logger


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


@numba.njit()
def n2(vs, graph, max_nb):
    res = np.empty((vs.shape[0], max_nb), dtype=graph[1].dtype.type)
    for i in range(vs.size):
        res[i] = sd.neighbours.neighbourhood_1d(vs[i], graph, max_nb)
    return res


def get_training_sample(G, graph, max_nb, batch_size):
    x, y = get_nce_sample(G, batch_size)
    x0 = n2(x[0]+1, graph, max_nb)
    x1 = n2(x[1]+1, graph, max_nb)
    return [x0, x1], y


def send_results(x, base_uri):
    b64 = base64.b64encode(json.dumps(x, cls=JsonNumpy, separators=(',',':')).encode()).decode()
    req = urllib.request.urlopen(f"{base_uri}/{b64}").read()
    return b64 


rg = lambda x: getattr(igraph.Graph, x)


if __name__ == '__main__':
    
    p = argparse.ArgumentParser()
    p.add_argument('-c', '--config')
    p.add_argument('-s', '--server')
    args, _ = p.parse_known_args()

    log = get_stdout_logger('link-prediction-worker')
    
    with open(args.config, 'r') as f:
        config = json.loads(f.read())

    log.info(f'Starting job with the following configuration: {config}')

    G, graph, sig = get_rgm(np.arange(config['VCOUNT_FROM'], config['VCOUNT_TO']), config['RGM_SAMPLING_SPACE'])
    obs = int(sig['n'])
    nnb = [1, int(np.quantile(G.degree(), .9))]
    dims = np.arange(*config['DIM_RANGE'])
    dims = np.random.choice(dims, config['DIM_SAMPLE'], replace=False)
    steps = config['TRAINING_STEPS']
    batch_size = config['BATCH_SIZE']

    log.info(f'Generated graph from {sig}')

    # eigen

    a = gtoa(G)
    val, vec = np.linalg.eig(a)
    val = np.real(val)
    vec = np.real(vec)

    for dim in dims:
        try:

            yhat = vec[:, :dim].dot(np.diag(val[:dim])).dot(vec.T[:dim, :])

            res = {
                'symmetric': True,
                'diagonal': True,
                'dim': int(dim),
                'reducer': 'eig',
                'max_nb': None,
                'loss': tf.keras.losses.binary_crossentropy(a, yhat).numpy().mean(),
                'acc': tf.keras.metrics.binary_accuracy(a, yhat).numpy().mean(),
                'auc': roc_auc_score(a.ravel(), yhat.ravel()),
                'rgm': sig,
                'batch_size': None,
                'steps': None
            }

            log.info(res)
            send_results(res, args.server)
            log.info(f"Sent results to {args.server}")

        except Exception as e:
            log.error(e)
            sys.exit()

    # neuro

    for sym in [True, False]:
        for diag in [True, False]:
            
            for dim in dims:
                
                reducers = {
                    'mean': lambda x: tf.reduce_mean(x, axis=-2),
                    'gnn': grl.layers.GNNSimple(dim, n_layers=1, activation=None),
                    'gnn-gelu': grl.layers.GNNSimple(dim, n_layers=1, activation=tf.nn.gelu),
                    'gat-h1': grl.layers.GAT(dim, n_layers=1, n_heads=1),
                    'gat-h2': grl.layers.GAT(dim, n_layers=1, n_heads=2)
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
                            'loss': hist[-100:, 0].mean(),
                            'acc': hist[-100:, 1].mean(),
                            'auc': hist[-100:, 2].mean(),
                            'rgm': sig,
                            'batch_size': batch_size,
                            'steps': steps
                        }
        
                        log.info(res)
                        send_results(res, args.server)
                        log.info(f"Sent results to {args.server}")
