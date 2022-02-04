Quick start
===========

Installation
----

.. code-block:: bash
    
    $ python3 -m pip install grl

The graph
----



Conversions
----

Grl supports some conversions between formats: 

.. code-block:: python
    
    >>> import grl, igraph
    >>> iG = igraph.Graph.Famous('Zachary')
    >>> iG.density()
    0.13903743315508021
    >>> G = grl.graph.utils.from_igraph(iG)
    >>> grl.density(G)
    0.13903743315508021

.. code-block:: python
    
    >>> A = grl.graph.utils.to_adjacency(G)
    >>> A.mean()
    0.13903743315508021
    >>> grl.graph.utils.hexdigest(grl.graph.utils.from_adjacency(A)) == grl.graph.utils.hexdigest(G)
    True

There is also an OGB dataset converter:

.. code-block:: python

    >>> from ogbn.nodeproppred import NodePropPredDataset
    >>> dataset = NodePropPredDataset('ogbn-arxiv')
    >>> G = grl.graph.utils.from_ogb(dataset)
    >>> grl.vcount(G)


Random graph models
----

A few RGMs from ``igraph`` are exposed in grl's ``graph.random`` API. For example:

.. code-block:: python
    
    >>> G = grl.graph.random.geometric(n=100, r=.2, seed=13)
    >>> grl.density(G)
    0.07918367346938776

Embedding
----

Embedding in Grl refers exlucisvely to stochastic factorization of the *adjacency matrix*. 
Note that this is different from majority of models like node2vec, DeepWalk or WYS. 
``grl.graph.encode`` exposes a few algorithms akin to linear matrix factorisation formulations. 

.. code-block:: python
   
    >>> L = grl.graph.encode.symmetric(graph=G, dim=10, steps=2**24)
    >>> L.shape
    (100, 10)

You can evaluate the fitted embedding using built-in metrics: 

.. code-block:: python

	>>> grl.metrics.accuracy(G, L)
	NotImplementedError                       Traceback (most recent call last)
	<ipython-input-42-188521ae1e42> in <module>()
	----> 1 grl.metrics.accuracy(G, L)

	~/proj/grl/grl/metrics/__init__.py in accuracy()
		  9 def accuracy(g, L, R=None):
		 10     if R is None:
	---> 11         raise NotImplementedError("symmetric model not supported")
		 12     if L.shape != R.shape:
		 13         raise NotImplementedError("diagonal model not supported")

	NotImplementedError: symmetric model not supported

As you can see, you **should not** use this package **at all**. 


Plotting
----

Plotting utility uses the Fruchterman-Reingold layout by default, 
but node coordinates can be passed as an optional argument. 
It calls ``igraph`` under the hood; keep that in mind when working with larger graphs. 

.. code-block:: python

    >>> grl.graph.plot(G)

.. image:: plot.png


