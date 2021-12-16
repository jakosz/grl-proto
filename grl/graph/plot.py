import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from . import utils


def layout_fruchterman_reingold(graph):
    return np.array(utils.to_igraph(graph).layout_fruchterman_reingold().coords)


def plot(graph, xy=None, c='k', s=10, alpha=.8):
    if xy is None:
        xy = layout_fruchterman_reingold(graph)
    edgelist = utils.enumerate_edges(graph)-1  # @indexing
    plt.scatter(xy[:, 0], xy[:, 1], c=c, s=s)
    plt.gca().add_collection(LineCollection([xy[e].tolist() for e in edgelist], color=c, alpha=alpha))
