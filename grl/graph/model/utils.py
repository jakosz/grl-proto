import numba

import grl


@numba.njit(cache=True)
def split_steps(steps, cores):
    # Calculate the number of iterations to be performed on each thread. 
    # This number needs to be even because samplers return n//2 positives 
    # and n//2 negatives to keep the batches balanced.  
    # This leads to segfaults when steps//cores happen to be an odd number, 
    # so we need to correct for that. 
    res = steps//cores
    return res - res % 2


def autoregister(graph):
    nodes, edges = graph
    name = f"graph_{grl.graph.utils.hexdigest(graph)[:16]}"
    nn, ne = (f"{name}_{e}" for e in ["nodes", "edges"])
    grl.set(nodes, nn)
    grl.set(edges, ne)
    setattr(grl.shmem._obj, name, (grl.get(nn), grl.get(ne)))
    return name
