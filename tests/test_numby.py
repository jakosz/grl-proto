import numpy as np
import tensorflow as tf

import grl

from common import random_binomial_2d, random_normal_2d


def test_binary_crossentropy(random_normal_2d, random_binomial_2d):
    p = random_binomial_2d()
    q = grl.softmax(random_normal_2d())
    for i in range(p.shape[0]):
        res_tf = tf.keras.losses.binary_crossentropy(p[i], q[i]).numpy().mean()
        res_grl = grl.binary_crossentropy(p[i], q[i])
        assert np.allclose(res_tf, res_grl) 


def test_clip_1d(random_normal_2d):
    for e in random_normal_2d():
        x = grl.clip_1d(e, -1, 1)
        assert np.max(x) <=  1
        assert np.min(x) >= -1


def test_cumsum_2d(random_normal_2d):
    x = random_normal_2d()
    for axis in [0, 1]:
        numby = grl.cumsum_2d(x, axis=axis)
        numpy = np.cumsum(x, axis=axis)
        assert np.all(numby == numpy)


def test_sigmoid(random_normal_2d):
    x = random_normal_2d()
    res_tf = tf.nn.sigmoid(x).numpy().ravel()
    res_grl = grl.sigmoid(x).ravel()
    assert np.allclose(res_tf, res_grl)


def test_softmax(random_normal_2d):
    for e in random_normal_2d():
        assert np.allclose(tf.nn.softmax(e).numpy(), grl.softmax(e))


def test_where_1d(random_normal_2d):
    x = random_normal_2d().ravel()
    for t in np.linspace(-3, 3, 7):
	res_np = np.where(x < t) 
	res_grl = where_1d(x < t)
	assert np.all(res_np[0] == res_grl[0])
	assert np.all(res_np[1] == res_grl[1])


def test_where_2d(random_normal_2d):
    x = random_normal_2d()
    for t in np.linspace(-3, 3, 7):
	res_np = np.where(x < t) 
	res_grl = where_2d(x < t)
	assert np.all(res_np[0] == res_grl[0])
	assert np.all(res_np[1] == res_grl[1])
