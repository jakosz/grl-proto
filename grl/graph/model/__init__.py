""" 
A lot of odd complexities in this module stem from how parallel 
computation without memory locks needs to be implemented in Python:  
    * shared array needs to be present in the global namespace before ProcessPoolExecutor is instantiated
    * Python function passed to ProcessPoolExecutor must not take reference to this array as an argument
    * numba function doing the computation must take it as an argument, but also must modify it in place
These considerations are implemented as follows: 
    1. `worker` implements the model fitting computation that we want to parallelise. 
    2. `worker_mp_wrapper` takes the names of the parameter arrays,  
        translates them to pointers and calls `worker`.  
        This way we can pass RawArrays around without actually passing them around. 
    3. `encode` creates arrays and phony names as members of the shmem._obj, 
        so they are accessible from the global namespace before ProcessPoolExecutor is created. 
Oh, by the way, for all this to even work you have to mount /tmp in memory before starting the Python program:
    
    $ sudo mount -o remount,size=100% /dev/shm     # increase the size of shared memory
    $ sudo mount -o size=100% -t tmpfs tmpfs /tmp  # mount /tmp in RAM

From now on it's smooth sailing xD
"""
from .shallow import Model
from .shallow_adam import ModelAdam
