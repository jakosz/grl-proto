import numba
import numpy as np


@numba.njit(cache=True)
def pack(x, bits):
    """ Pack bit array to unsigned integer.
        
        Parameters
        ----------
        x : uint8[:]
            Bit array to unpack.
        bits : int
            Number of bits of the result (8, 16, 32 or 64)
            
        Returns
        -------
        res : uint
    """    
    return ((2**np.arange(bits)[::-1]).astype(np.uint64) * x).sum()


@numba.njit(cache=True)
def unpack(x, bits):
    """ Unpack unsigned integer to bit array.
    
        Parameters
        ----------
        x : uint8, uint16, uint32, uint64
            Integer to unpack
        bits : int
            Number of bits (8, 16, 32 or 64)
            
        Returns
        -------
        res : uint8[:]
    """
    res = np.zeros(bits, dtype=np.uint8)
    one = np.ones(1, dtype=x.dtype.type)     # using literal 1 will make the compiler think it's a float
    two = np.array([2], dtype=x.dtype.type)  # same here ¯\_(ツ)_/¯ (both break on uint64)
    for i in range(bits):
        if x % 2:
            res[bits-1-i] = 1
            x = x-one
        x = x//two
    return res


@numba.njit(cache=True)
def pack64(x):
    assert x.shape[1] == 64
    res = np.empty(x.shape[0], dtype=np.uint64)
    for i in range(x.shape[0]):
        res[i] = pack(x[i], 64)
    return res


@numba.njit(cache=True)
def unpack64(x):
    assert x.dtype.type is np.uint64
    res = np.empty((x.shape[0], 64), dtype=np.uint8)
    for i in range(x.shape[0]):
        res[i] = unpack(x[i:i+1], 64)
    return res


@numba.njit(cache=True)
def pack32(x):
    assert x.shape[1] == 32
    res = np.empty(x.shape[0], dtype=np.uint32)
    for i in range(x.shape[0]):
        res[i] = pack(x[i], 32)
    return res


@numba.njit(cache=True)
def unpack32(x):
    assert x.dtype.type is np.uint32
    res = np.empty((x.shape[0], 32), dtype=np.uint8)
    for i in range(x.shape[0]):
        res[i] = unpack(x[i:i+1], 32)
    return res
