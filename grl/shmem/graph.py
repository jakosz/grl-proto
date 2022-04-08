""" Shared memory utilities specific to the graph data structure. 
"""
from grl.graph.utils import hexdigest

from . import _obj
from . import _ops


def discover(graph):
    """ Check if graph is registered in shared memory, and return its reference.

        Parameters
        ----------
        graph : tuple

        Returns
        -------
        str
            Graph reference (shared memory name) or empty string.
    """
    name = name(graph)
    if _ops.get(name):
        return name
    else:
        return ""


def name(graph):
    """ Create graph name from its hexdigest. 
    """
    return f"graph_{hexdigest(graph)[:16]}"


def register(graph):
    """ Copy nodes and edges to shared memory, and create references to nodes,
        edges, and the graph. 
    """
    nodes, edges = graph
    gn = name(graph) 
    nn, en = (f"{gn}_{e}" for e in ["nodes", "edges"])
    _ops.set(nodes, nn)
    _ops.set(edges, en)
    setattr(_obj, gn, (get(nn), get(en)))
    return name
