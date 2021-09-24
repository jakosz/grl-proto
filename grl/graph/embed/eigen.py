import numpy as np

from ..utils import to_adjacency


def encode(g):
    A = to_adjacency(g)
    val, vec = np.linalg.eig(A)
    val = np.real(val)
    vec = np.real(vec)
    srt = np.argsort(np.abs(val))[::-1]
    val = val[srt]
    vec = vec[:, srt]
    return val, vec


def decode(val, vec, dim=None):
    dim = val.shape[0] if dim is None else dim
    return vec[:, :dim].dot(np.diag(val[:dim])).dot(vec.T[:dim, :])
