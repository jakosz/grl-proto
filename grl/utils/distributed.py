import numpy as np

from . import utils
from . import ipc 


class MirroredArray:
    def __init__(self, arr, hosts, port, sample_size=1):
        self.arr = arr
        self.hosts = hosts
        self.port = port
        self.sender = sender(self.arr, sample_size).start()
        self.receivers = []
        for host in hosts:
            self.receivers.append(receiver(self.arr, host).start())
    
    def __getattr__(self, *args, **kwargs):
        return getattr(self.arr, *args, **kwargs)
    
    def __getitem__(self, i):
        return self.arr[i]
    
    def __repr__(self):
        return f"MirroredArray[{len(self.hosts)}] \n{self.arr.__repr__()}"
    
    def __setitem__(self, i, v):
        self.arr[i] = v


def receiver(arr, host, port=5555):
    ctx, sock = ipc.socket_pull(host, port)
    @utils.background()
    @ipc.pull(sock)
    def _receiver(data, arr):
        arr[data['key']] = data['value']
    return _receiver(arr)


def sender(arr, sample_size, port=5555):
    ctx, sock = ipc.socket_push(port)
    @utils.background()
    @ipc.push(sock)
    def _sender(arr, sample_size):
        i = np.random.choice(arr.shape[0], sample_size)
        return {'key': i, 'value': arr[i]}
    return _sender(arr, sample_size)
