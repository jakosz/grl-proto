import numba
import numpy as np

from . import core

""" neighbours_nd functions populate the index array initialised 
    to have a starting node at the origin (e.g. arr[0, 0, 0] in 3d case)
"""

@numba.njit(cache=True)
def _fill_node_nb_1d(arr, graph):
    nb = core.neighbours(arr[0], graph)
    if nb.size > arr.size-1:
        nb = np.random.choice(nb, arr.size-1)
    arr[1:nb.shape[0]+1] = nb

    
@numba.njit(cache=True)
def _fill_node_nb_2d(arr, graph):
    _fill_node_nb_1d(arr[:, 0], graph)
    for i in range(1, arr.shape[1]):
        _fill_node_nb_1d(arr[i, :], graph)


@numba.njit(cache=True)
def _fill_node_nb_3d(arr, graph):
    _fill_node_nb_2d(arr[:, :, 0], graph)
    for i in range(1, arr.shape[2]):
        _fill_node_nb_2d(arr[i, :, :], graph)


@numba.njit(cache=True)
def _fill_node_nb_4d(arr, graph):
    _fill_node_nb_3d(arr[:, :, :, 0], graph)
    for i in range(1, arr.shape[3]):
        _fill_node_nb_3d(arr[i, :, :, :], graph)


""" _get_node_nb_tensor_nd allocate observation tensor to be populated by neighbours_nd
"""


@numba.njit(cache=True)
def _get_node_nb_tensor_1d(vi, max_nb):
    ix = np.zeros((max_nb), dtype=np.uint32)
    ix[0] = vi
    return ix


@numba.njit(cache=True)
def _get_node_nb_tensor_2d(vi, max_nb):
    ix = np.zeros((max_nb, max_nb), dtype=np.uint32)
    ix[0, 0] = vi
    return ix
    

@numba.njit(cache=True)
def _get_node_nb_tensor_3d(vi, max_nb):
    ix = np.zeros((max_nb, max_nb, max_nb), dtype=np.uint32)
    ix[0, 0, 0] = vi
    return ix


@numba.njit(cache=True)
def _get_node_nb_tensor_4d(vi, max_nb):
    ix = np.zeros((max_nb, max_nb, max_nb, max_nb), dtype=np.uint32)
    ix[0, 0, 0, 0] = vi
    return ix


""" neighbourhood_nd returns a tensor where n-th axis encodes n-th neighbourhood order.  
    If neighbourhood size exceeds max_nb, a sample with replacement is drawn. 

  idea: max_nb could be optionally overriden by recursively increasing buffer size, 
    but then some checks on the degree sequence should be run to avoid OOM.  
"""


@numba.njit(cache=True)
def neighbourhood_1d(vi, graph, max_nb):
    res = _get_node_nb_tensor_1d(vi, max_nb)
    _fill_node_nb_1d(res, graph)
    return res
        

@numba.njit(cache=True)
def neighbourhood_2d(vi, graph, max_nb):
    res = _get_node_nb_tensor_2d(vi, max_nb)
    _fill_node_nb_2d(res, graph)
    return res
    

@numba.njit(cache=True)
def neighbourhood_3d(vi, graph, max_nb):
    res = _get_node_nb_tensor_3d(vi, max_nb)
    _fill_node_nb_3d(res, graph)
    return res


@numba.njit(cache=True)
def neighbourhood_4d(vi, graph, max_nb):
    res = _get_node_nb_tensor_4d(vi, max_nb)
    _fill_node_nb_4d(res, graph)
    return res
