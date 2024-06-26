import numpy as np

from . import utils
from . import ipc 


class MirroredArray:
    def __init__(self, arr, hosts, port, sample_size=1, tick=0):
        self.arr = arr
        self.hosts = hosts
        self.port = port
        self.receiver = receiver(self.arr, port, tick=tick).start()
        self.senders = []
        for host in hosts:
            self.senders.append(sender(self.arr, sample_size, host, port, tick=tick).start())
    
    def __getattr__(self, *args, **kwargs):
        return getattr(self.arr, *args, **kwargs)
    
    def __getitem__(self, i):
        return self.arr[i]
    
    def __repr__(self):
        return f"MirroredArray[{len(self.hosts)}] \n{self.arr.__repr__()}"
    
    def __setitem__(self, i, v):
        self.arr[i] = v


def receiver(arr, port=5555, tick=0):
    ctx, sock = ipc.socket_rep(port)
    @utils.background(tick=tick)
    @ipc.pull(sock)
    def _receiver(data, arr):
        t0 = arr[data['key']].copy()
        t1 = (t0+data['value'])/2
        arr[data['key']] = t1
    return _receiver(arr)


def sender(arr, sample_size, host, port=5555, tick=0):
    ctx, sock = ipc.socket_req(host, port)
    @utils.background(tick=tick)
    @ipc.push(sock)
    def _sender(arr, sample_size):
        i = np.random.choice(arr.shape[0], sample_size)
        return {'key': i, 'value': arr[i]}
    return _sender(arr, sample_size)
