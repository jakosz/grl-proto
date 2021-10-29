from numba.config import NUMBA_NUM_THREADS

from . import graph
from . import layers
from . import models
from . import utils
from .graph.core import *
from .numby import *
from .shmem import *

__version__ = "0.4.20"

# constants
CLIP = 5.0
CORES = NUMBA_NUM_THREADS
EPSILON = 1e-7
