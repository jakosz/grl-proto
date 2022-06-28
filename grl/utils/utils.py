import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import numpy as np


class DataGen:
    def __init__(self, qsize, f, *args, **kwargs):
        """ Wrap a function in a queue-populating loop
            and expose the queue as a generator.
        """
        self._step = 0
        self._stop = True
        self.args = args
        self.f = f
        self.futures = []
        self.kwargs = kwargs
        self.pool = ThreadPoolExecutor(qsize)
        self.qsize = qsize
        self.queue = Queue(qsize)
        
        self.start()
        
    def __call__(self):
        return self
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args, **kwargs):
        self.stop()
        
    def __iter__(self):
        return self.__next__()
            
    def __next__(self):
        if not self._stop:
            return self.queue.get()
        else:
            raise StopIteration
            
    def __repr__(self):
        res = f"{'stopped' if self._stop else 'running'} datagen[{self.qsize}]\n"
        res += f"  args: {self.args}\n"
        res += f"kwargs: {self.kwargs}\n"
        return res
        
    def loop(self):
        while not self._stop:
            self.queue.put(self.f(*self.args, **self.kwargs))
            self._step += 1
            
    def start(self):
        self._stop = False
        for i in range(self.qsize):
            self.futures.append(self.pool.submit(self.loop))
            
    def stop(self):
        self._stop = True
        [self.queue.get() for i in range(self.qsize)]
        self.pool.shutdown(wait=False)


class JsonNumpy(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.int32) or isinstance(obj, np.int64) or isinstance(obj, np.uint32) or isinstance(obj, np.uint64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


def background(n=1, tick=1, store_results=True):
    """ Wrap a function in a loop and delegate execution to a thread pool.
    """
    def wrap(f):
        def wrap(*args, **kwargs):
            class ThreadsAndLoops:
                def __init__(self):
                    self._step = 0
                    self._stop = False
                    self._tick = tick
                    self.results = []

                def run(self):
                    while not self._stop:
                        res = f(*args, **kwargs)
                        if store_results:
                            self.results.append(res)
                        self._step += 1
                        time.sleep(self._tick)

                def start(self, tick=tick):
                    self._stop = False
                    self._tick = tick
                    self.pool = ThreadPoolExecutor(n)
                    self.futures = [self.pool.submit(self.run) for _ in range(n)]
                    return self

                def stop(self):
                    self._stop = True
                    self.pool.shutdown(wait=False)
            return ThreadsAndLoops()
        return wrap
    return wrap


def datagen(qsize=os.cpu_count()):
    def wrap(f):
        def wrap(*args, **kwargs):
            return DataGen(qsize, f, *args, **kwargs)
        return wrap
    return wrap


def random_hex(n=16):
    return np.random.randint(0, 2**63, 8).tobytes().hex()[:n]


NumpyJson = JsonNumpy
