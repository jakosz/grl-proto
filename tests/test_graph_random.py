import numpy as np

import grl

from common import igraphs, graphs


def test_seed(graphs):
    for seed in [13, 14, 15]:
        sig0, sig1 = [], []
        for G in graphs(seed):
            sig0.append(grl.graph.utils.hexdigest(G))
        for G in graphs(seed):
            sig1.append(grl.graph.utils.hexdigest(G))
        assert all([e == f for e, f in zip(sig0, sig1)])
