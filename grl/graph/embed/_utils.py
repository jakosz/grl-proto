import numba


@numba.njit()
def split_steps(steps, cores):
    # Calculate the number of iterations to be performed on each thread. 
    # This number needs to be even because samplers return n//2 positives 
    # and n//2 negatives to keep the batches balanced.  
    # This leads to segfaults when steps//cores happen to be an odd number, 
    # so we need to correct for that. 
    res = steps//cores
    return res - res % 2
