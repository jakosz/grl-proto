import argparse

import json
import time
import traceback

from grl.utils import get_stdout_logger, JsonNumpy

from src import *


p = argparse.ArgumentParser()
p.add_argument('--runs', type=int)
p.add_argument('--dims', type=int, help='Upper limit of dimensions to scan')
p.add_argument('--vcount', type=int)
p.add_argument('--iter', type=int)
p.add_argument('--st', action='store_true', help='Use single-threaded implementation')
p.add_argument('--verbose', action='store_true', help='Display job status') 
p.add_argument('--output', type=str, default='link-prediction-v4.json')
args, _ = p.parse_known_args()

log = get_stdout_logger('link-prediction-v4')

config.vcount = args.vcount
if args.iter is not None:
    config.iter = args.iter

if args.st:
    embs = embs_st

times = RunningAverage()
for run in range(args.runs):
    for emb_name, emb_model in embs.items():
        for rgm_name, rgm_model in rgms.items():
            for dim in range(2, args.dims):
                
                try:
                    g = rgm_model()
                except:
                    log.error(f"Failed to sample graph from {rgm_name} model. This is unexpected.") 
                
                try:
                    t0 = time.time()
                    auc = emb_model(g, dim)
                    t1 = time.time()

                    res = {
                        'auc': auc, 
                        'dim': dim,
                        'eigen': eigen(g, dim),
                        'emb': emb_name,
                        'rgm': rgm_name,
                        'density': grl.density(g),
                        'vcount': grl.vcount(g),
                        'ecount': grl.ecount(g),
                        'iter': config.iter
                    }

                    with open(args.output, 'a') as f:
                        f.write(json.dumps(res, cls=JsonNumpy) + "\n")

                    times(t1-t0)
                    if args.verbose:
                        print(f"\t{times.count:,} models total - {times.value:.04f} sec./run    ", end="\r", flush=True)
                except:
                    log.error(f"Failed to embed {rgm_name} graph using {emb_name} model. vcount={grl.vcount(g)}, ecount={grl.ecount(g)}, dim={dim}")
