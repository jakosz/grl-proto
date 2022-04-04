""" This package serves several purposes: 
    1. Provide building blocks to enable LLVM compilation of higher-level functions wherever numba does not yet support them. 
    2. Enable interactive speeds for large datasets on highly parallel architectures. 
    3. Explore what's possible by combining the two with core grl functionality.  

At the moment it mostly boils down to replicating some of the Numpy and Tensorflow functionality.  
Convention: functions that modify their arguments in place (without a memcpy) do not return anything. 
"""

from .bits import *
from .core import *
from .reduce import *
