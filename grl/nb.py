""" Common imports for Jupyter notebook session.  
"""
import time
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numba
import numpy as np
import ray
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.keras.layers import *

import grl


inform_user = """import time
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numba
import numpy as np
import ray
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.keras.layers import *

import grl"""
print(inform_user)


def background(f):
    def wrap():
        class BackgroundTask:
            def __init__(self, f, tick=1):
                self._stop = False
                self._tick = tick 
                self.results = []
                self.f = f
            
            def run(self):
                while not self._stop:
                    self.results.append(self.f())
                    time.sleep(self._tick)
            
            def start(self, tick=1):
                self._stop = False
                self._tick = tick
                self.pool = ThreadPoolExecutor(1)
                self.future = self.pool.submit(self.run)
                
            def stop(self):
                self._stop = True
                self.pool.shutdown(wait=False)
        return BackgroundTask(f)
    return wrap
