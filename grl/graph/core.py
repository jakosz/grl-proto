import numba
import numpy as np


@numba.jit(nopython=True, fastmath=True)
def binary_crossentropy(p, q):
    q = clip(q, 1e-7, 1-1e-7)
    return -np.mean(p*np.log(q) + (1-p)*np.log(1-q))


@numba.jit(nopython=True, fastmath=True)
def clip(x, lower, upper):
    for i in range(x.shape[0]):
        if x[i] < lower:
            x[i] = lower
        if x[i] > upper:
            x[i] = upper
    return x


@numba.jit(nopython=True, fastmath=True)
def cos_decay(p):
    return (0.5 * (1 + np.cos(np.pi * p)))


@numba.jit(nopython=True)
def ecount(graph):
    """ Graph's edge count.
    """
    v, e = graph
    return e.shape[0] 


@numba.jit(nopython=True)
def explode_pairs(x):
    """ List all possible (symmetric) edges between unique elements of x. """
    x = np.unique(x)
    n = x.shape[0]
    res = np.empty(((n**2-n)//2, 2), dtype=np.uint32)
    c = 0
    for i in range(n):
        for j in range(i + 1, n):
            res[c, 0] = x[i]
            res[c, 1] = x[j]
            c += 1
    return res


@numba.jit(nopython=True, fastmath=True, parallel=True)
def fill(x):
    """ Fill shared array with gaussian noise, scaled down by its dimension.  
    """
    for i in numba.prange(x.shape[0]):
        x[i] = np.random.randn(x.shape[1])/x.shape[1]


@numba.jit(nopython=True, fastmath=True)
def get_nc_pairs(graph, walk_length, batch_size, nodes):
    """ Get a balanced sample of node pairs and noise contrast. 

        Parameters
        ----------

        Returns
        -------
        (X, Y) where X is 2darray of node pairs, 
               and Y flags source distribution (1 for data, 0 for noise)
    """

    bs = batch_size//2
    
    X = np.empty((batch_size, 2), dtype=np.uint32)
    Y = np.empty(batch_size, dtype=np.uint8)
    
    X[:bs] = walks_to_pairs(graph, walk_length, bs, nodes)
    X[bs:] = np.random.choice(nodes, size=(bs, 2)).astype(np.uint32)
    
    Y[:bs] = 1
    Y[bs:] = 0

    return X, Y


@numba.jit(nopython=True)
def neighbours(i, graph):
    """ Get neighbours of node i. """
    v, e = graph
    if v[i+1] - v[i] > 0:
        return e[v[i]:v[i] + (v[i+1] - v[i])]
    else:
        return np.array([0], dtype=e.dtype)
    

@numba.jit(nopython=True)
def random_walk(i, steps, graph):
    """ Perform random walk starting at node i.

        Parameters
        ----------
        i : int
            Index of the starting node.
        steps : int
            Number of edges to traverse.
        graph : (graph, node_index)
            Graph structure to traverse, as returned by from_edgelist.

        Returns
        -------
        res : 1darray[uint32]
            Array of nodes on the walker's path.

    """
    res = np.empty(steps + 1, dtype=np.uint32)
    res[0] = i
    for step in range(1, steps + 1):
        n = neighbours(i, graph)
        if not n[0]:
            return res[:step]
        else:
            i = np.random.choice(n)
            res[step] = i
    return res


@numba.jit(nopython=True, fastmath=True)
def sigmoid(x):
    return 1/(1+np.e**-x)


@numba.jit(nopython=True, fastmath=True)
def update(x, y, params, lr):
    """ Update embedding.

        Parameters
        ----------
        x : 2darray[uint32]
        y : 1darray[uint8]
        params : 2darray[float32]
        lr : float

        Returns
        -------
        res : None
            Shared array is updated in place.
    """
    for i in range(x.shape[0]):
        xL = params[x[i, 0]]
        xR = params[x[i, 1]]
        dy = sigmoid(np.dot(xL, xR)) - y[i]
        dL = clip(xR*dy*lr, -5, 5)
        dR = clip(xL*dy*lr, -5, 5)
        params[x[i, 0]] = xL-dL
        params[x[i, 1]] = xR-dR


@numba.jit(nopython=True)
def vcount(graph):
    """ Graph's vertex count.
    """
    v, e = graph
    return v.shape[0] - 2 # first is empty, last is the length of edge array


@numba.jit(nopython=True)
def walks_to_pairs(graph, walk_length, bs, nodes):
    """ Build an edgelist from nodes associated via random walks.
    
        Parameters
        ----------
        graph : (nodes, edges) 
            Tuple of node and edge arrays.  
        walk_length : int
            Walk length. 
        bs : int
            Size of the resulting edgelist. 
        
        Returns
        -------
        res : 2darray
            An edgelist. 
        
    """
    e = np.empty((bs, 2)).astype(np.uint32)
    c = 0
    while c < bs:
        i = np.random.choice(nodes)
        p = explode_pairs(random_walk(i, walk_length, graph))
        s = p.shape[0]
        if c+s > bs:
            p = p[:bs-c]
            s = p.shape[0]
        e[c:c+s] = p
        c += s
    return e


@numba.jit(nopython=True, parallel=True)
def recode(x, mapping):
    """ Recode the values of an array.

        Parameters
        ----------
        x : 1darray
            Data to recode.
        mapping : 1darray
            New values. Elements of x with value i
            will be replaced by i-th element of mapping.

        Returns
        -------
        x : 1darray
            Remapped values of x.
    """
    for i in numba.prange(x.shape[0]):
        x[i] = mapping[x[i]]


""" Force jit to try to compile with tbb backend, and fall back to  
"""

@numba.jit(nopython=True, parallel=True)
def _dummy():
    """ Dummy function to force jit to try to use a threading backend. """
    res = np.empty(2)
    for i in numba.prange(2):
        res[i] = i
    return res


try:
    numba.config.THREADING_LAYER = 'tbb'
    _dummy()
except numba.NumbaError:
    numba.config.THREADING_LAYER = 'default'
    _dummy()

