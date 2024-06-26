import grl

from common import graphs_small as graphs 


def with_model(emb_type, sampler, activation):
    def wrap(f):
        def wrap(graphs):
            for G in graphs():
                obs = grl.vcount(G)
                model = grl.graph.Model(obs=grl.vcount(G), 
                                        dim=16, 
                                        emb_type=emb_type, 
                                        sampler=sampler, 
                                        activation=activation)
                at0 = model.evaluate(G)
                model.fit(G, 2**14, cos_decay=False)
                at1 = model.evaluate(G)
                assert at1 > at0
        return wrap
    return wrap


@with_model('asymmetric', 'nce', 'sigmoid')
def test_asymmetric_nce_sigmoid(graphs):
    pass


@with_model('asymmetric', 'nce', None)
def test_asymmetric_nce_linear(graphs):
    pass


@with_model('asymmetric', 'neg', 'sigmoid')
def test_asymmetric_neg_sigmoid(graphs):
    pass


@with_model('asymmetric', 'neg', None)
def test_asymmetric_neg_linear(graphs):
    pass


@with_model('diagonal', 'nce', 'sigmoid')
def test_diagonal_nce_sigmoid(graphs):
    pass


@with_model('diagonal', 'nce', None)
def test_diagonal_nce_linear(graphs):
    pass


@with_model('diagonal', 'neg', 'sigmoid')
def test_diagonal_neg_sigmoid(graphs):
    pass


@with_model('diagonal', 'neg', None)
def test_diagonal_neg_linear(graphs):
    pass


@with_model('symmetric', 'nce', 'sigmoid')
def test_symmetric_nce_sigmoid(graphs):
    pass


@with_model('symmetric', 'nce', None)
def test_symmetric_nce_linear(graphs):
    pass


@with_model('symmetric', 'neg', 'sigmoid')
def test_symmetric_neg_sigmoid(graphs):
    pass


@with_model('symmetric', 'neg', None)
def test_symmetric_neg_linear(graphs):
    pass
