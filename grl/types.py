""" Common data types. 
"""
import numba.types as nt


edge = nt.Array(nt.uint32, 1, "C")

edges = nt.Array(nt.uint32, 2, "C")

graph = nt.Tuple(
        (
            nt.Array(nt.uint64, 1, "C"), 
            nt.Array(nt.uint32, 1, "C"))
        )

graph_sample = nt.Tuple(
        (
            nt.Array(nt.uint32, 2, "C"), 
            nt.Array(nt.float32, 1, "C"))
        )
