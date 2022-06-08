from concurrent.futures import ThreadPoolExecutor
from queue import Queue


class DataGen:
    def __init__(self, qsize, f, args, kwargs):
        self._step = 0
        self._stop = None
        self.futures = []
        self.pool = ThreadPoolExecutor(qsize)
        self.qsize = qsize
        self.queue = Queue(qsize)
        self.f = f
        self.args = args
        self.kwargs = kwargs
        self.start()
    
    def __call__(self):
        return self
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args, **kwargs):
        self.stop()
    
    def __next__(self):
        return self.queue.get()
    
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
        [self.queue.get() for e in range(self.qsize)]
        self.pool.shutdown(wait=False)
    

def datagen(qsize):
    def wrap(f):
        def wrap(*args, **kwargs):
            return DataGen(qsize, f, args, kwargs)
        return wrap
    return wrap
