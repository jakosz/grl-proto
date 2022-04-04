import numba

CLIP = 5.0
CORES = numba.config.NUMBA_NUM_THREADS
EPSILON = 1e-7
PART_SIZE = 1024  # size of the sample size used in training loops
