import numpy as np

from grl import shmem
from grl.graph import core


def asymmetric(model):
    n = model.obs 
    d = model.dim
    if type(n) is int or len(n) == 1:
        n = (n, n)
    model._refs = [
        f"{model._id}_{n[0]}x{d}_L",
        f"{model._id}_{n[1]}x{d}_R"
    ]
    model._params = [
        shmem.randn((n[0]+1, d), model._refs[0]), 
        shmem.randn((n[1]+1, d), model._refs[1])  # @indexing - do we REALLY need this here? 
    ]


def diagonal(model):
    n = model.obs
    d = model.dim
    if type(n) is not int:
        n = n[0]
    model._refs = [
        f"{model._id}_{n}x{d}_L",
        f"{model._id}_{d}_D",
    ]
    model._params = [
        shmem.randn((n+1, d), model._refs[0]), 
        shmem.zeros((d,), np.float32, model._refs[1])
    ]


def symmetric(model):
    n = model.obs 
    d = model.dim
    if type(n) is not int:
        n = n[0]
    model._refs = [
        f"{model._id}_{n}x{d}_L"
    ]
    model._params = [
        shmem.randn((n+1, d), model._refs[0]) 
    ]
