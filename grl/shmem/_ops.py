""" Shared memory handlers. 
"""

import ctypes as ctypes
from inspect import getmembers as getmembers
from multiprocessing import RawArray as RawArray

import numpy as np

from ..numby import random_randn_fill_inplace
from . import _obj


__all__ = [
    "empty",
    "empty_like",
    "get",
    "load",
    "ls",
    "randn",
    "rm",
    "set",
    "zeros",
    "zeros_like"
]


_ctype = lambda x: np.ctypeslib.as_ctypes_type(x)
_size = lambda x: int(np.prod(x))
_shared = lambda s, t: np.frombuffer(RawArray(_ctype(t), _size(s)), dtype=t).reshape(*s)


def empty(shape, dtype, name=None):
    a = _shared(shape, dtype)
    name = name if name is not None else f"_{id(a)}"
    setattr(_obj, name, a)
    return get(name) 


def empty_like(x, name=None):
    return empty(x.shape, x.dtype.type, name=name)


def flush_shmem():
    for e in ls():
        rm(e)


def get(name):
    return getattr(_obj, name)


def load(path, name=None):
    """ Load numpy array to shared memory and use path as a name in grl.mem._obj. 
    """
    set(np.load(path), name=path if name is None else name)
    return get(path)


def ls():
    return [e[0] for e in getmembers(_obj) if not e[0].startswith('_')]


def randn(shape, name):
    x = empty(shape, dtype=np.float32, name=name)
    random_randn_fill_inplace(x)
    return x


def rm(name):
    delattr(_obj, name)


def set(x, name):
    shared = _shared(x.shape, x.dtype.type) 
    shared[:] = x
    setattr(_obj, name, shared)


def zeros(shape, dtype, name=None):
    return empty(shape, dtype, name=name)


def zeros_like(x, name=None):
    return empty_like(x, name=name)
