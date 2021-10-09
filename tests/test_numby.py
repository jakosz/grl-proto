import numpy as np

import grl


def test_clip():
    pass


def test_sigmoid():
    assert grl.sigmoid(0.) == .5
    assert grl.sigmoid(+1e+3) == 1
    assert np.allclose([grl.sigmoid(-1e+2)], [0])


def test_softmax():
    pass
