""" This is grl.

Some general notes: 
* I try to mark places where conversions between 1- and 0- indexing are taking place with a `@indexing` comment.
* Many tests are carried out against `igraph`, `numpy` and `tensorflow`. This is not very elegant, and shouldn't be *the only* case, but for initial development should suffice. 
"""
from . import graph
from . import metrics
from . import utils
from .graph.core import *
from .nn import *
from .numby import *
from .shmem import *

__version__ = "0.6.0"
