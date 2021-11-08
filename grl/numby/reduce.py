import numba
import numpy as np


def reducer_2d(numpy_reducer, axis):
    """ Compile numpy axis reducer for 2darrays.
    """
    axis = 0 if axis else 1  # flip to make the call look like numpy axis arg
    @numba.njit(cache=True)
    def wrap(x):
        res = np.empty(x.shape[axis])
        for i in range(x.shape[axis]):
            if axis:
                res[i] = numpy_reducer(x[:, i])
            else:
                res[i] = numpy_reducer(x[i])
        return res
    return wrap


max0 = reducer_2d(np.max, 0)
max1 = reducer_2d(np.max, 1)
min0 = reducer_2d(np.min, 0)
min1 = reducer_2d(np.min, 1)
sum0 = reducer_2d(np.sum, 0)
sum1 = reducer_2d(np.sum, 1)
std0 = reducer_2d(np.std, 0)
std1 = reducer_2d(np.std, 1)
mean0 = reducer_2d(np.mean, 0)
mean1 = reducer_2d(np.mean, 1)
