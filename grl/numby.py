import numba
import numpy as np

import grl.EPSILON as EPSILON


@numba.njit(fastmath=True)
def binary_crossentropy(p, q):
    return -np.mean(p*np.log(q+EPSILON) + (1-p)*np.log(1-q+EPSILON))


@numba.njit(fastmath=True)
def clip_1d(x, lower, upper):
    x = x.copy()
    for i in range(x.shape[0]):
        if x[i] < lower:
            x[i] = lower
        if x[i] > upper:
            x[i] = upper
    return x


@numba.njit(fastmath=True)
def clip_1d_inplace(x, lower, upper):
    for i in range(x.shape[0]):
        if x[i] < lower:
            x[i] = lower
        if x[i] > upper:
            x[i] = upper


@numba.njit(fastmath=True)
def cos_decay(p):
    return (0.5 * (1 + np.cos(np.pi * p)))


@numba.njit()
def random_choice(x, s, w):
    """ Weighted choice with replacement.
    """
    assert x.size == w.size
    expand = np.empty(w.sum(), dtype=x.dtype.type)
    _ = 0
    for i in range(w.size):
        expand[_:_+w[i]] = np.repeat(x[i], w[i])
        _ += w[i]
    return np.random.choice(expand, s)


@numba.njit(fastmath=True, parallel=True)
def random_randn_fill_inplace(x):
    """ Fill an array with gaussian noise scaled down by its dimension. 
    """
    for i in numba.prange(x.shape[0]):
        x[i] = np.random.randn(x.shape[1])/x.shape[1]


@numba.njit(fastmath=True)
def sigmoid(x):
    return 1/(1+np.exp(-x))


@numba.njit(fastmath=True)
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))
