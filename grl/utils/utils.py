import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np


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


def random_hex(n=16):
    return np.random.randint(0, 2**63, 8).tobytes().hex()[:n]


NumpyJson = JsonNumpy
