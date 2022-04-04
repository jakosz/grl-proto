import grl

from common import graphs_small as graphs 


def test_asymmetric_nce_sigmoid(graphs):
    for G in graphs():
        model = grl.graph.Model(G, 16, type='asymmetric', sampler='nce', activation='sigmoid')
        at0 = model.evaluate(G)
        model.fit(G, 2**12, cos_decay=False)
        at1 = model.evaluate(G)
        assert at1 > at0


def test_asymmetric_nce_linear(graphs):
    for G in graphs():
        model = grl.graph.Model(G, 16, type='asymmetric', sampler='nce', activation=None)
        at0 = model.evaluate(G)
        model.fit(G, 2**12, cos_decay=False)
        at1 = model.evaluate(G)
        assert at1 > at0


def test_asymmetric_neg_sigmoid(graphs):
    for G in graphs():
        model = grl.graph.Model(G, 16, type='asymmetric', sampler='neg', activation='sigmoid')
        at0 = model.evaluate(G)
        model.fit(G, 2**12, cos_decay=False)
        at1 = model.evaluate(G)
        assert at1 > at0


def test_asymmetric_neg_linear(graphs):
    for G in graphs():
        model = grl.graph.Model(G, 16, type='asymmetric', sampler='neg', activation=None)
        at0 = model.evaluate(G)
        model.fit(G, 2**12, cos_decay=False)
        at1 = model.evaluate(G)
        assert at1 > at0


def test_diagonal_nce_sigmoid(graphs):
    for G in graphs():
        model = grl.graph.Model(G, 16, type='diagonal', sampler='nce', activation='sigmoid')
        at0 = model.evaluate(G)
        model.fit(G, 2**12, cos_decay=False)
        at1 = model.evaluate(G)
        assert at1 > at0


def test_diagonal_nce_linear(graphs):
    for G in graphs():
        model = grl.graph.Model(G, 16, type='diagonal', sampler='nce', activation=None)
        at0 = model.evaluate(G)
        model.fit(G, 2**12, cos_decay=False)
        at1 = model.evaluate(G)
        assert at1 > at0


def test_diagonal_neg_sigmoid(graphs):
    for G in graphs():
        model = grl.graph.Model(G, 16, type='diagonal', sampler='neg', activation='sigmoid')
        at0 = model.evaluate(G)
        model.fit(G, 2**12, cos_decay=False)
        at1 = model.evaluate(G)
        assert at1 > at0


def test_diagonal_neg_linear(graphs):
    for G in graphs():
        model = grl.graph.Model(G, 16, type='diagonal', sampler='neg', activation=None)
        at0 = model.evaluate(G)
        model.fit(G, 2**12, cos_decay=False)
        at1 = model.evaluate(G)
        assert at1 > at0


def test_symmetric_nce_sigmoid(graphs):
    for G in graphs():
        model = grl.graph.Model(G, 16, type='symmetric', sampler='nce', activation='sigmoid')
        at0 = model.evaluate(G)
        model.fit(G, 2**12, cos_decay=False)
        at1 = model.evaluate(G)
        assert at1 > at0


def test_symmetric_nce_linear(graphs):
    for G in graphs():
        model = grl.graph.Model(G, 16, type='symmetric', sampler='nce', activation=None)
        at0 = model.evaluate(G)
        model.fit(G, 2**12, cos_decay=False)
        at1 = model.evaluate(G)
        assert at1 > at0


def test_symmetric_neg_sigmoid(graphs):
    for G in graphs():
        model = grl.graph.Model(G, 16, type='symmetric', sampler='neg', activation='sigmoid')
        at0 = model.evaluate(G)
        model.fit(G, 2**12, cos_decay=False)
        at1 = model.evaluate(G)
        assert at1 > at0


def test_symmetric_neg_linear(graphs):
    for G in graphs():
        model = grl.graph.Model(G, 16, type='asymmetric', sampler='neg', activation=None)
        at0 = model.evaluate(G)
        model.fit(G, 2**12, cos_decay=False)
        at1 = model.evaluate(G)
        assert at1 > at0
