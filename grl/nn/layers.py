import tensorflow as tf
from tensorflow.keras.layers import *


class Relu2(Layer):
    def __init__(self, dim, *args, **kwargs):
        super(Relu2, self).__init__(*args, **kwargs)
        self.w = Dense(dim)
        self.n = LayerNormalization()
        
    def call(self, x):
        return tf.nn.relu(self.n(self.w(x)))**2
