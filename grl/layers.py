import tensorflow as tf
from tensorflow.keras.layers import *


class Identity(Layer):
    def __init__(self):
        super(Identity, self).__init__()
        
    def call(self, x):
        return x
    

class Vector(Layer):
    def __init__(self, dim):
        super(Vector, self).__init__()
        self.v = tf.Variable(tf.random.normal((dim,)))
        
    def call(self, x):
        return self.v * x
    
    
class GNNSimple(Layer):
    
    def __init__(self, dim, n_layers, **kwargs):
        super(GNNSimple, self).__init__()
        self.w = []
        for layer in range(n_layers):
            self.w.append(Dense(dim, **kwargs))
            
    def call(self, x):
        for i in range(x.shape.ndims - 2):
            x = tf.reduce_mean(self.w[i](x), axis=-2)
        return x


def reduce_gat(x, w, a):
    wx = w(x)
    gather = tf.gather(wx, [0], axis=-2)
    repeat = tf.repeat(gather, wx.shape[-2], axis=-2)
    concat = tf.concat([wx, repeat], axis=-1)
    matmul = a(concat)
    attend = tf.nn.softmax(tf.nn.gelu(matmul), axis=-2)
    return tf.nn.gelu(tf.reduce_sum(wx*attend, axis=-2))


class GAT(Layer):
    def __init__(self,
                 dim,
                 n_layers,
                 n_heads):
        super(GAT, self).__init__()

        self.n_layers = n_layers
        self.n_heads = n_heads

        self.a = []
        self.w = []
        for layer in range(n_layers):
            a, w = [], []
            for head in range(n_heads):
                a.append(Dense(1, activation=tf.nn.gelu, use_bias=False))
                w.append(Dense(dim, activation=None, use_bias=False))
            self.a.append(a)
            self.w.append(w)

    def call(self, x):
        for i in range(x.shape.ndims - 2):
            heads = []
            for head in range(self.n_heads):
                heads.append(tf.expand_dims(reduce_gat(x, self.w[i][head], self.a[i][head]), 0))
            x = tf.reduce_mean(tf.concat(heads, axis=0), axis=0)
        return x
