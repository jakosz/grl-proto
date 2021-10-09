import tensorflow as tf
from tensorflow.keras.layers import *

from .layers import Identity, Vector


def _build_inputs(obs, dim, max_nb, symmetric, diagonal):
    inputs = [Input(shape=(max_nb,)) for _ in range(2)]
    latent = [Embedding(obs, dim, mask_zero=True) for _ in range(2-int(symmetric))]
    
    if symmetric:
        latent.append(latent[0])
    
    if diagonal:
        latent.append(Vector(dim))
    else:
        latent.append(Identity())
    
    return inputs, latent


def _build_model(inputs, latent, reducer):
    e0 = reducer(latent[0](inputs[0]))
    e1 = reducer(latent[1](inputs[1]))
    return tf.nn.sigmoid(tf.reduce_sum(latent[2](e0*e1), axis=-1))


def _compile_model(inputs, outputs, loss, metrics):
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss=loss, metrics=metrics)
    return model


def get(vcount, dim, symmetric, diagonal, 
        loss='binary_crossentropy', metrics=['binary_accuracy', 'AUC'], 
        max_nb=1, reducer=tf.squeeze):
    obs = vcount+1  # @indexing
    inputs, latent = _build_inputs(obs, dim, max_nb, symmetric, diagonal)
    outputs = _build_model(inputs, latent, reducer)
    model = _compile_model(inputs, outputs, loss, metrics)
    return model, latent
