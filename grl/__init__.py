import numba

from . import graph
from . import layers
from . import metrics
from . import models
from . import utils
from .graph.core import *
from .numby import *
from .shmem import *

__version__ = "0.5.6"

# constants
CLIP = 5.0
CORES = numba.config.NUMBA_NUM_THREADS
EPSILON = 1e-7
