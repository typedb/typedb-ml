import tensorflow as tf


def encode(raw_arrays, encoders):
    """
    Take data from traversals and build neighbourhood_depths
    :param encoders: encoder to use for each key in the supplied dictionaries
    :param raw_arrays: expects a list of dictionaries, one for each depth, each key referring to an array of the raw
    features of the traversals
    :return:
    """

    encoded_arrays = []
    for raw_array in raw_arrays:
        encoded_features = [encoders[key](features_array) for key, features_array in raw_array.items()]
        encoded_arrays.append(tf.concat(encoded_features, -1))

    return encoded_arrays
