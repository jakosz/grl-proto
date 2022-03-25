import pickle
import zmq


def socket_pull(host="localhost", port=5555):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PULL)
    sock.connect(f"tcp://{host}:{port}")
    return ctx, sock


def socket_push(port=5555):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUSH)
    sock.bind(f"tcp://*:{port}")
    return ctx, sock


def push(sock):
    def wrap(f):
        def wrap(*args, **kwargs):
            sock.send(pickle.dumps(f(*args, **kwargs)))
        return wrap
    return wrap


def pull(sock):
    """ Pull data from socket and pass to the decorated function. 
        
        Notes
        -----
        The first argument to the wrapped function should be always specified
        and stand for data received over the socket, e.g.: 
            
            @pull(sock)
            def foo(data):
                # do something

        Other arguments are optional, but need to come after the data:

            @pull(sock)
            def foo(data, add, queue):
                data = int.from_bytes(data, 'little')
                queue.put(data+add)

    """
    def wrap(f):
        def wrap(*args, **kwargs):
            f(pickle.loads(sock.recv()), *args, **kwargs)
        return wrap
    return wrap
