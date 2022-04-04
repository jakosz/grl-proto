from concurrent.futures import ProcessPoolExecutor


class Model:
    def __init__(self, graph, dim, type='asymmetric'):
        """ Create a shallow model of a graph.

            Parameters
            ----------
            graph : tuple 
                Input graph. 
            dim : int
                Embedding dimensionality.
            type : str, optional
                Shallow model to use. Should be one of: asymmetric, diagonal, symmetric. 
                Defaults to asymmetric.
        """
        self._futures = []
        self._params = []
        self._refs = []
        self.dim = dim
        self.graph = graph
        self.type = type
        self.initialize()

    def evaluate(self, graph):
        return metrics.accuracy(graph, *self.params)

    def fit(self, graph, steps):
        checks(self, graph)
        encode(self, graph)

    @property
    def params(self):
        return self._params
    
    @property
    def refs(self):
        return self._refs


def checks():
    pass


def encode():
    pass


def worker_mp_wrapper():
    pass
