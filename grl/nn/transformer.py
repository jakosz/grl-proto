import tensorflow as tf
from tensorflow.keras.layers import *


class GeluNN(Layer):
    def __init__(self, dim0, dim1, dropout=0.1, **kwargs):
        super(GeluNN, self).__init__(**kwargs)
        self.w0 = Dense(dim0, activation=tf.nn.gelu)
        self.w1 = Dense(dim1, activation=tf.nn.gelu)
        self.d0 = Dropout(dropout)
        self.d1 = Dropout(dropout)

    def call(self, x):
        return self.d1(self.w1(self.d0(self.w0(x))))
    
    
class TransformerBlock(Layer):
    def __init__(self, d_model, num_heads, dff, dropout=0.1, dtype=tf.float32):
        super(TransformerBlock, self).__init__()

        self.mha = MultiHeadAttention(num_heads, d_model)
        self.ffn = GeluNN(dff, d_model, dropout)

        self.n0 = LayerNormalization(epsilon=1e-6, dtype=dtype)
        self.n1 = LayerNormalization(epsilon=1e-6, dtype=dtype)

        self.dropout0 = Dropout(dropout)
        self.dropout1 = Dropout(dropout)

    def call(self, x, training, mask=None):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout0(attn_output, training=training)
        out0 = self.n0(x + attn_output)

        ffn_output = self.ffn(out0)
        ffn_output = self.dropout1(ffn_output, training=training)
        out1 = self.n1(out0 + ffn_output)

        return out1
