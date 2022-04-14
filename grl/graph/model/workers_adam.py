import numba
import numpy as np

from grl import config 
from grl import numby


@numba.njit(cache=True)
def asymmetric(x, y, lr, b1, b2, activation, 
               L, R, mL, mR, vL, vR, tL, tR):
    n = x.shape[0]
    for j in range(n):
        iL, iR = x[j]
        xL = L[iL]
        xR = R[iR]
        # compute gradients
        dy = activation(np.sum(xL*xR)) - y[j]
        dxL = clip(xR*dy)
        dxR = clip(xL*dy)
        # update adam parameters
        # learning rate: 
        aL = lr * np.sqrt(1.0 - b2**tL[iL]) / (1.0 - b1**tL[iL]) 
        aR = lr * np.sqrt(1.0 - b2**tR[iR]) / (1.0 - b1**tR[iR])
        # first moment: 
        mL[iL] = b1 * mL[iL] + (1.0 - b1) * dxL
        mR[iR] = b1 * mR[iR] + (1.0 - b1) * dxR
        # second moment: 
        vL[iL] = b2 * vL[iL] + (1.0 - b2) * dxL**2
        vR[iR] = b2 * vR[iR] + (1.0 - b2) * dxR**2
        # update counters:
        tL[iL] += 1
        tR[iR] += 1
        # update model parameters
        L[iL] -= aL * mL[iL] / (np.sqrt(vL[iL]) + config.EPSILON) 
        R[iR] -= aR * mR[iR] / (np.sqrt(vR[iR]) + config.EPSILON)


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
