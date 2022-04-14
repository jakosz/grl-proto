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
        f"{model._id}_{n[1]}x{d}_R",
        f"{model._id}_{n[0]}x{d}_mL",
        f"{model._id}_{n[1]}x{d}_mR",
        f"{model._id}_{n[0]}x{d}_vL",
        f"{model._id}_{n[1]}x{d}_vR",
        f"{model._id}_{n[0]}_tL",
        f"{model._id}_{n[1]}_tR",
    ]
    model._params = [
        shmem.randn((n[0]+1, d), model._refs[0]), 
        shmem.randn((n[1]+1, d), model._refs[1]),  # @indexing - do we REALLY need this here?
        shmem.zeros((n[0]+1, d), np.float32, model._refs[2]), 
        shmem.zeros((n[1]+1, d), np.float32, model._refs[3]), 
        shmem.zeros((n[0]+1, d), np.float32, model._refs[4]), 
        shmem.zeros((n[1]+1, d), np.float32, model._refs[5]), 
        shmem.zeros((n[0]+1,), np.uint64, model._refs[6]), 
        shmem.zeros((n[1]+1,), np.uint64, model._refs[7])
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
