import argparse
import json
import random
import yaml

import numpy as np
import ray

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
def embed(graph, 
          config, 
          output, 
          dim, 
          symmetric, 
          diagonal, 
          sampling, 
          loss='binary_crossentropy'):
    
    graph, info = graph

    t = config['embedding']['train']
    info.batch_size = t['batch_size']
    info.steps_per_epoch = t['steps_per_epoch']
    info.epochs = t['epochs']
    info.dim = dim
    info.sampling = sampling
    
    if sampling == 'nce':
        dg = datagen_nce
    elif sampling == 'neg':
        dg = datagen_neg
    else:
        raise ValueError("sampling must be either nce or neg") 

    model, [L, R, D] = grl.models.get(info.nodes, dim, symmetric=symmetric, diagonal=diagonal, loss=loss)
    hist = model.fit(dg(graph, info.batch_size), steps_per_epoch=info.steps_per_epoch, epochs=info.epochs, verbose=False).history

    res = info.__dict__
    res.update({
        "auc": hist["auc"], 
        "acc": hist["binary_accuracy"]
    })

    with open(f"{output}/{info.name}.json", 'a') as f:
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

    graph = sample_graph(config)
    d = config['embedding']['model']['dim']

    fu = []
    for dim in range(d['start'], d['end']):
        for sd in embedding_configs:
            for sampling in ['nce', 'neg']:
                fu.append(embed.remote(graph, config, args.output, dim=dim, **sd, sampling=sampling, loss='binary_crossentropy'))
