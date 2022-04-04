from concurrent.futures import ProcessPoolExecutor

from grl import config
from grl import metrics
from grl import numby
from grl import shmem
from grl.graph import sample
from grl.utils import log, random_hex
from . import activations
from . import initializers
from . import predictors
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
            activation : str, optional
                Name of the activation, defaults to 'sigmoid'.
            sampler : str, optional
                Name of the sampler, one of the functions implemented in the 
                graph.sample module. Defaults to 'nce' (noise contrastive). 
        """
        self._futures = []
        self._id = random_hex()
        self._params = []
        self._refs = []
        self.activation = activations.get(activation)
        self.dim = dim
        self.graph = graph
        self.sampler = getattr(sample, sampler)
        self.type = type
        self.initialize()

    def evaluate(self, graph, sample_size=8192):
        x, y = self.sampler(graph, sample_size)
        yhat = self.predict(x) 
        return metrics.accuracy(y, yhat)

    def fit(self, graph, steps, lr=.025, cos_decay=False):
        checks(self, graph)
        return encode(self, graph, steps, lr, cos_decay)

    def initialize(self):
        getattr(initializers, self.type)(self)

    @property
    def params(self):
        return self._params

    def predict(self, x):
        return getattr(predictors, self.type)(x, *self.params, self.activation)
    
    @property
    def refs(self):
        return self._refs


def checks(model, graph):
    pass


def encode(model,
           graph, 
           steps, 
           lr, 
           cos_decay): 
    p = ProcessPoolExecutor(config.CORES)
    for core in range(config.CORES):
        model._futures.append(
            p.submit(worker_mp_wrapper, 
                worker=getattr(workers, model.type),
                sampler=model.sampler,
                activation=model.activation,
                graph=graph,
                refs=model._refs,
                steps=utils.split_steps(steps, config.CORES), 
                lr=lr, 
                cos_decay=cos_decay, 
                part_size=config.PART_SIZE))


def worker_mp_wrapper(worker,
                      sampler,
                      activation,
                      graph, 
                      refs,
                      steps,
                      lr, 
                      cos_decay, 
                      part_size):
    parts = steps//part_size
    for i in range(parts):
        x, y = sampler(graph, part_size)
        clr = lr if not cos_decay else numby.cos_decay(i/parts)*lr
        worker(x, y, *[shmem.get(e) for e in refs], clr, activation)

