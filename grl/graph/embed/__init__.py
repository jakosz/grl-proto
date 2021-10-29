""" 
A lot of odd complexities in this module stem from how parallel 
computation without memory locks needs to be implemented in Python:  
    * shared array needs to be present in the global namespace before ProcessPoolExecutor is instantiated
    * Python function passed to ProcessPoolExecutor must not take reference to this array as an argument
    * numba function doing the computation must take it as an argument, but also must modify it in place
"""

from . import asymmetric 
from . import diagonal 
from . import eigen
from . import symmetric 
