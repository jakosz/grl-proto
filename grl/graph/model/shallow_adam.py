from concurrent.futures import ProcessPoolExecutor

from grl import config
from grl import metrics
from grl import numby
from grl import shmem
from grl.graph import sample
from grl.utils import log, random_hex
from . import activations
from . import initializers_adam
from . import predictors
from . import utils
from . import workers_adam


class ModelAdam:
    def __init__(self, 
                 obs, 
                 dim, 
                 emb_type='asymmetric', 
                 activation='sigmoid', 
                 sampler='neg'):
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
        self.vcount2 = obs[1] if self.bimodal else 0 
        self.initialize()

    def evaluate(self, graph_or_ref, sample_size=8192):
        if type(graph_or_ref) is not str:
            graph = graph_or_ref
        else:
            graph = shmem.get(graph_or_ref)
        x, y = self.sampler(graph, sample_size, self.vcount2)
        yhat = self.predict(x) 
        return metrics.accuracy(y, yhat)

    def fit(self, graph_or_ref, steps, lr=1e-4, b1=.9, b2=.999):
        """ Perform `steps` parameter updates.

            Parameters
            ----------
            graph_or_ref : tuple or str
                A graph or a reference to a graph registered in grl's shmem.
            steps : int
                Number of updates to perform.
            lr : float, optional
                Learning rate. Defaults to 0.25.
            b1 : float, optional
            b2 : float, optional

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
        return encode(self, ref, steps, lr, b1, b2)

    def initialize(self):
        # init params
        getattr(initializers_adam, self.emb_type)(self)

    @property
    def params(self):
        return self._params

    def predict(self, x):
        return getattr(predictors, self.emb_type)(x, *self.params[:2], self.activation)
    
    @property
    def refs(self):
        return self._refs


def checks(model, graph):
    """ Run checks to validate that an a Model can be fitted to graph. 
    """
    pass


def encode(model,
           ref, 
           steps, 
           lr, b1, b2): 
    with ProcessPoolExecutor(config.CORES) as p:
        for core in range(config.CORES):
            model._futures.append(
                p.submit(worker_mp_wrapper, 
                         worker=getattr(workers_adam, model.emb_type),
                         sampler=model.sampler,
                         activation=model.activation,
                         ref=ref,
                         refs=model.refs,
                         vcount2=model.vcount2,
                         steps=utils.split_steps(steps, config.CORES), 
                         lr=lr, b1=b1, b2=b2)) 


def worker_mp_wrapper(worker,
                      sampler,
                      activation,
                      ref,   # data ref  
                      refs,  # param refs
                      vcount2,
                      steps,
                      lr, b1, b2): 
    parts = steps//config.PART_SIZE
    for i in range(parts):
        x, y = sampler(shmem.get(ref), config.PART_SIZE, vcount2)
        worker(x, y, lr, b1, b2, activation, *(shmem.get(e) for e in refs))

