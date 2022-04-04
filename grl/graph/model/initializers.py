import numpy as np

from grl import shmem
from grl.graph import core


def asymmetric(model):
    n = core.vcount(model.graph)
    d = model.dim
    model._refs = [
        f"{model._id}_{n}x{d}_{e}" for e in "LR"
    ]
    model._params = [
        shmem.randn((n+1, d), e) for e in model._refs        
    ]


def diagonal(model):
    n = core.vcount(model.graph)
    d = model.dim
    model._refs = [
        f"{model._id}_{n}x{d}_L",
        f"{model._id}_{d}_D",
    ]
    model._params = [
        shmem.randn((n+1, d), model._refs[0]), 
        shmem.zeros((d,), np.float32, model._refs[1])
    ]


def symmetric(model):
    n = core.vcount(model.graph)
    d = model.dim
    model._refs = [
        f"{model._id}_{n}x{d}_L"
    ]
    model._params = [
        shmem.randn((n+1, d), model._refs[0]) 
    ]
