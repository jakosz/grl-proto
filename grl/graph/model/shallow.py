from concurrent.futures import ProcessPoolExecutor

from grl import config
from grl import numby
from grl import shmem
from grl.graph import sample
from . import activations
from . import initializers
from . import utils
from . import workers


class Model:
    def __init__(self, 
                 graph, 
                 dim, 
                 type='asymmetric', 
                 activation='sigmoid', 
                 sampler='nce'):
        """ Create a shallow model of a graph.

            Parameters
            ----------
            graph : tuple 
                Input graph. 
            dim : int
                Embedding dimensionality.
            type : str, optional
                Shallow model to use. Should be one of: asymmetric, diagonal, 
                symmetric. Defaults to 'asymmetric'.
            sampler : str, optional
                Name of the sampler, one of the functions implemented in the 
                graph.sample module. Defaults to 'nce' (noise contrastive). 
        """
        self._futures = []
        self._params = []
        self._refs = []
        self.activation = activations.get(activation)
        self.dim = dim
        self.graph = graph
        self.sampler = getattr(sample, sampler)
        self.type = type
        self.initialize()

    def evaluate(self, graph):
        return metrics.accuracy(graph, *self.params)

    def fit(self, graph, steps, lr=.025, cos_decay=False):
        checks(self, graph)
        encode(self, graph, steps, lr, cos_decay)

    def initialize(self):
        getattr(initializers, self.type)(self)

    @property
    def params(self):
        return self._params
    
    @property
    def refs(self):
        return self._refs


def checks(model, graph):
    pass


def encode(model,
           graph, 
           steps, 
           lr): 
    with ProcessPoolExecutor(config.CORES) as p:
        for core in range(config.CORES):
            p.submit(worker_mp_wrapper, 
                     model, 
                     graph, 
                     utils.split_steps(steps, config.CORES), 
                     lr) 


def worker_mp_wrapper(model, 
                      graph, 
                      steps, 
                      lr, 
                      cos_decay):
    parts = steps//config.PART_SIZE
    worker = getattr(workers, model.type)
    for i in range(parts):
        x, y = model.sampler(graph, config.PART_SIZE)
        clr = lr if not cos_decay else numby.cos_decay(i/parts)*lr
        worker(x, y, *(shmem.get(e) for e in model._refs), clr, model.activation)


