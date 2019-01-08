import numpy as np
import tensorflow as tf


def initialise_glorot_weights(shape, name=None):
    """
    Glorot & Bengio (AISTATS 2010) init.
    :param shape: shape of the weights matrix to build
    :param name: Name for the operation (optional).
    :return: initialised matrix of weights
    """
    with tf.name_scope(name, default_name="init_glorot_weights") as scope:
        init_range = np.sqrt(6.0/(shape[0]+shape[1]))
        initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)

        if tf.executing_eagerly():
            return tf.contrib.eager.Variable(initial, name=name)
        return tf.Variable(initial, name=name)
