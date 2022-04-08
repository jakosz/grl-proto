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
    graph_name = name(graph)
    if _ops.get(graph_name):
        return graph_name
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
    graph_name = name(graph) 
    if _ops.get(graph_name):
        return graph_name 
    else:
        nodes, edges = graph
        nodes_name, edges_name = (f"{graph_name}_{e}" for e in ["nodes", "edges"])
        _ops.set(nodes, nodes_name)
        _ops.set(edges, edges_name)
        setattr(_obj, graph_name, (_ops.get(nodes_name), _ops.get(edges_name)))
        return graph_name
