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
                 obs, 
                 dim, 
                 emb_type='asymmetric', 
                 activation='sigmoid', 
                 sampler='nce'):
        """ Create a shallow model of a graph.

            Parameters
            ----------
            obs : int or tuple 
                Numbers of observations: int or 1-tuple for unimodal graph, 
                2-tuple for bimodal. 
            dim : int
                Embedding dimensionality.
            emb_type : str, optional
                Shallow model to use. Should be one of: asymmetric, diagonal, 
                symmetric. Defaults to 'asymmetric'.
            activation : str, optional
                Name of the activation, defaults to 'sigmoid'.
            sampler : str, optional
                Name of the sampler, one of the functions implemented in the 
                graph.sample module. Defaults to 'nce' (noise contrastive). 
        """
        self._futures = []  # for debugging; workers return None
        self._id = random_hex()
        self._params = []
        self._refs = []  # param refs
        self.activation = activations.get(activation)
        self.bimodal = False if type(obs) is int or len(obs) == 1 else True
        self.dim = dim
        self.obs = obs
        self.sampler = getattr(sample, sampler)
        self.emb_type = emb_type
        self.nargs = (obs[1] if self.bimodal else 0,)  # :| 
        self.initialize()

    def evaluate(self, graph_or_ref, pargs=(), sample_size=8192):
        if type(graph_or_ref) is not str:
            graph = graph_or_ref
        else:
            graph = shmem.get(graph_or_ref)
        x, y = self.sampler(graph, sample_size, pargs=pargs, nargs=self.nargs)
        yhat = self.predict(x) 
        return metrics.accuracy(y, yhat)

    def fit(self, graph_or_ref, steps, pargs=(), lr=.01, dropout=.0, cos_decay=False):
        """ Perform `steps` parameter updates.

            Parameters
            ----------
            graph_or_ref : tuple or str
                A graph or a reference to a graph registered in grl's shmem.
            steps : int
                Number of updates to perform.
            lr : float, optional
                Learning rate. Defaults to 0.01.
            dropout : float, optional
                Dropout rate. Defaults to 0.0.
            cos_decay : bool, optional
                If set, cosine decay schedule will be applied to learning rate.
                Defaults to False.

            Returns
            -------
            None
                Used for side effects. 
        """
        if type(graph_or_ref) is not str:
            ref = shmem.graph.register(graph_or_ref)
        else:
            ref = graph_or_ref
        checks(self, shmem.get(ref))
        return encode(self, ref, pargs, steps, lr, cos_decay, dropout)

    def initialize(self):
        # init params
        getattr(initializers, self.emb_type)(self)

    @property
    def params(self):
        return self._params

    def predict(self, x):
        return getattr(predictors, self.emb_type)(x, *self.params, self.activation)
    
    @property
    def refs(self):
        return self._refs


def checks(model, graph):
    """ Run checks to validate that an a Model can be fitted to graph. 
    """
    pass


def encode(model,
           ref,
           pargs,
           steps, 
           lr, 
           cos_decay, 
           dropout): 
    with ProcessPoolExecutor(config.CORES) as p:
        for core in range(config.CORES):
            model._futures.append(
                p.submit(worker_mp_wrapper, 
                         worker=getattr(workers, model.emb_type),
                         sampler=model.sampler,
                         activation=model.activation,
                         ref=ref,
                         refs=model.refs,
                         pargs=pargs,
                         nargs=model.nargs,
                         steps=utils.split_steps(steps, config.CORES), 
                         lr=lr, 
                         cos_decay=cos_decay, 
                         dropout=dropout)) 


def worker_mp_wrapper(worker,
                      sampler,
                      activation,
                      ref,   # data ref  
                      refs,  # param refs
                      pargs,
                      nargs,
                      steps,
                      lr, 
                      cos_decay, 
                      dropout): 
    parts = steps//config.PART_SIZE
    for i in range(parts):
        x, y = sampler(shmem.get(ref), config.PART_SIZE, pargs=pargs, nargs=nargs)
        clr = lr if not cos_decay else numby.cos_decay(i/parts)*lr
        worker(x, y, *(shmem.get(e) for e in refs), clr, activation, dropout)

