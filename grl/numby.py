import numba
import numpy as np


@numba.njit(fastmath=True)
def binary_crossentropy(p, q):
    q = clip(q, 1e-7, 1-1e-7)
    return -np.mean(p*np.log(q) + (1-p)*np.log(1-q))


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


@numba.njit(fastmath=True)
def clip(x, lower, upper):
    for i in range(x.shape[0]):
        if x[i] < lower:
            x[i] = lower
        if x[i] > upper:
            x[i] = upper
    return x


@numba.njit(fastmath=True)
def cos_decay(p):
    return (0.5 * (1 + np.cos(np.pi * p)))


@numba.njit(fastmath=True)
def sigmoid(x):
    return 1/(1+np.e**-x)
