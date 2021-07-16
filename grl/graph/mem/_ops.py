""" Shared memory handlers. 
"""

import ctypes as _ctypes
from inspect import getmembers as _getmembers
from multiprocessing import RawArray as _RawArray

import numpy as _np

from . import _obj


_ctype = lambda x: _np.ctypeslib.as_ctypes_type(x)
_size = lambda x: int(_np.prod(x))
_shared = lambda s, t: _np.frombuffer(_RawArray(_ctype(t), _size(s)), dtype=t).reshape(*s)


def empty(shape, dtype, name=None):
    a = _shared(shape, dtype)
    name = name if name is not None else f"_{id(a)}"
    setattr(_obj, name, a)
    return get(name) 


def empty_like(x, name=None):
    return empty(x.shape, x.dtype.dtype, name=name)


def get(name):
    return getattr(_obj, name)


def load(path, name=None):
    """ Load numpy array to shared memory and use path as a name in sd.mem._obj. """
    set(_np.load(path), name=path if name is None else name)
    return get(path)


def ls():
    return [e[0] for e in _getmembers(_obj) if not e[0].startswith('_')]


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
