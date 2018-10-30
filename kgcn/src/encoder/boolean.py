import tensorflow as tf


def one_hot_boolean_encode(boolean_features_as_integers):
    """
    :param boolean_features_as_integers: A tensor of booleans represented as integers, with final dimension 1.
    Expects 0 to indicate False, 1 to indicate True, -1 to indicate neither True or False
    :return: One-hot boolean encoding tensor of same shape as `boolean_features` but with last dimension 2 (
    one-hot length of boolean)
    """
    return tf.squeeze(tf.one_hot(boolean_features_as_integers, 2, on_value=1, off_value=0), axis=-2)
