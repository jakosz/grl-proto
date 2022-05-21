import tensorflow as tf
from tensorflow.keras.layers import *


class Gelu(Layer):
    def __init__(self, dim, drop=.1):
        super(Gelu, self).__init__()
        self.w = Dense(dim, activation=None)
        self.n = LayerNormalization()
        self.d = Dropout(drop)
        
    def call(self, x):
        return self.d(tf.nn.gelu(self.w(self.n(x))))  # op order follows https://github.com/lukemelas/do-you-even-need-attention/blob/main/vision_transformer_linear.py
    
    
class Mixer(Layer):
    def __init__(self, dim, drop):
        """ Something like the MLP mixer, but with tensor dim reduction (https://arxiv.org/pdf/2105.01601.pdf).
        """
        super(Mixer, self).__init__()
        self.h0 = Gelu(dim, drop)
        self.h1 = Gelu(dim, drop)
        
    def call(self, x):
        h0 = self.h0(x)
        dims = len(h0.shape.dims)
        t0 = tf.transpose(h0, [*range(dims-2), dims-1, dims-2])
        h1 = self.h1(t0)
        dims = len(h1.shape.dims)
        return tf.transpose(h1, [*range(dims-2), dims-1, dims-2])
