import numba
import numpy as np

from grl import config 
from grl import numby


@numba.njit(cache=True)
def asymmetric(x, y, L, R, lr, activation, dropout):
    n = x.shape[0]
    j = np.random.choice(L.shape[1], int(L.shape[1]*(1-dropout)))
    for i in range(n):
        xL = L[x[i, 0]][j]
        xR = R[x[i, 1]][j]
        # compute gradients
        dy = activation(np.sum(xL*xR)) - y[i]
        dxL = clip(xR*dy)
        dxR = clip(xL*dy)
        # update parameters
        L[x[i, 0]][j] -= dxL*lr
        R[x[i, 1]][j] -= dxR*lr


@numba.njit(cache=True)
def clip(x):
    numby.clip_1d_inplace(x, -config.CLIP, config.CLIP)
    return x


@numba.njit(cache=True)
def diagonal(x, y, L, D, lr, activation):
    n = x.shape[0]
    for i in range(n):
        xL = L[x[i, 0]]
        xR = L[x[i, 1]]
        # compute gradients
        dy = activation(np.sum(xL*xR*D)) - y[i]  # output
        dxLR = D*dy  # embedding product 
        dD = clip(xL*xR*dy)  # diagonal
        dxL = clip(xR*dxLR)  # left vector
        dxR = clip(xL*dxLR)  # right vector
        # update parameters
        L[x[i, 0]] -= dxL*lr
        L[x[i, 1]] -= dxR*lr
        D[:] -= dD*lr


@numba.njit(cache=True)
def symmetric(x, y, L, lr, activation):
    n = x.shape[0]    
    for i in range(n):
        xL = L[x[i, 0]]
        xR = L[x[i, 1]]
        # compute gradients
        dy = activation(np.sum(xL*xR)) - y[i]
        dxL = clip(xR*dy)
        dxR = clip(xL*dy)
        # update parameters
        L[x[i, 0]] -= dxL*lr
        L[x[i, 1]] -= dxR*lr
