import argparse
import json
import random
import traceback
import yaml

import numpy as np
import ray
from sklearn.metrics import accuracy_score, roc_auc_score

import grl


ray.init()


class Namespace:
    pass


def datagen_nce(G, batch_size):
    while True:
        x, y = grl.graph.sample.nce(G, batch_size)
        yield [x[:, 0], x[:, 1]], y
        
def datagen_neg(G, batch_size):
    while True:
        x, y = grl.graph.sample.neg(G, batch_size)
        yield [x[:, 0], x[:, 1]], y


def sample_rgm_params(config):
    
    # sample rgm model
    
    rgm = config['graph']['edges']
    mod = random.choice(list(rgm.keys()))
    
    # sample parameters
    
    if type(rgm[mod]['start']) is int:
        start = rgm[mod]['start']
        end = rgm[mod]['end']
        step = rgm[mod].get('step', 1)
        par = random.choice(range(start, end, step))
    elif type(rgm[mod]['start']) is float:
        start = rgm[mod]['start']
        end = rgm[mod]['end']
        par = float(np.round(np.random.uniform(start, end), 4))
    else:
        raise ValueError("RGM parameter range needs to be int or float")

    # sample vcount
    
    nod = config['graph']['nodes']
    start = nod['start']
    end = nod['end']
    step = nod.get('step', 1)
    nod = np.random.choice(np.arange(start, end, step))

    res = Namespace()
    res.nodes = nod
    res.model = mod
    res.param = par
    return res


def sample_graph(config):
    info = sample_rgm_params(config)
    info.seed = random.randint(0, 2**63-1) 
    graph = getattr(grl.graph.random, info.model)(info.nodes, info.param, seed=info.seed)
    info.name = grl.graph.utils.hexdigest(graph)[:16]
    return graph, info


@ray.remote
def embed(config, 
          output, 
          dim, 
          symmetric, 
          diagonal, 
          loss='binary_crossentropy'):
    
    graph, info = sample_graph(config)

    # unpack training info   
    t = config['embedding']['train']
    
    info.batch_size = t['batch_size']
    info.steps_per_epoch = t['steps_per_epoch']
    info.epochs = t['epochs']
    
    info.dim = dim
    info.sampling = 'neg' 
    info.symmetric = symmetric
    info.diagonal = diagonal
    info.method = 'embedding'
    
    res = info.__dict__
    
    try:
        model, [L, R, D] = grl.models.get(info.nodes, dim, symmetric=symmetric, diagonal=diagonal, loss=loss)
        hist = model.fit(datagen_neg(graph, info.batch_size), steps_per_epoch=info.steps_per_epoch, epochs=info.epochs, verbose=False).history
        res.update({
            "auc": hist["auc"], 
            "acc": hist["binary_accuracy"],
            "bce": hist["loss"]
        })
    except:
        res.update({'traceback': traceback.format_exc()})
    
    with open(output, 'a') as f:
        f.write(json.dumps(res, cls=grl.utils.JsonNumpy) + "\n")



if __name__ == "__main__":
    
    p = argparse.ArgumentParser()
    p.add_argument('--config')
    p.add_argument('--output')
    args, _ = p.parse_known_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f.read())

    embedding_configs = [
        {'symmetric': True, 'diagonal': False},
        {'symmetric': True, 'diagonal': True},
        {'symmetric': False, 'diagonal': False}
    ]

    d = config['embedding']['model']['dim']

    # embeddings get parallelised with ray:
    fu = []
    for dim in range(d['start'], d['end']):
        for sd in embedding_configs:
            fu.append(embed.remote(config, args.output, dim=dim, **sd, loss='binary_crossentropy'))

    # spectral is quicker, no need to parallelise:
    for dim in range(d['start'], d['end']):
        
        graph, info = sample_graph(config)
        A = grl.graph.utils.to_adjacency(graph)
        val, vec = grl.graph.embed.eigen.encode(graph)
        yhat = grl.graph.embed.eigen.decode(val, vec, dim)

        info.dim = dim
        info.sampling = None
        info.symmetric = True
        info.diagonal = True
        info.method = 'eigen'
        
        info.auc = [roc_auc_score(A.ravel(), yhat.ravel())]
        info.acc = [accuracy_score(A.ravel(), np.round(yhat.ravel()))]
        info.bce = [grl.binary_crossentropy(A.ravel(), yhat.ravel())]

        res = info.__dict__
        
        with open(args.output, 'a') as f:
            f.write(json.dumps(res, cls=grl.utils.JsonNumpy) + "\n")


ray.get(fu)  # block
