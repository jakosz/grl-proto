import os

from ..graph import utils

HERE = os.path.dirname(__file__)


def citeseer():
    graph = utils.load(f'{HERE}/data/citeseer/citeseer.grl')
    targets = np.load(f'{HERE}/data/citeseer/citeseer-targets.npy')
    return graph, targets


def cora():
    graph = utils.load(f'{HERE}/data/cora/cora.grl')
    targets = np.load(f'{HERE}/data/cora/cora-targets.npy')
    return graph, targets
