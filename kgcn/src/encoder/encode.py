import tensorflow as tf


def encode_all(raw_arrays, encoders, name='encode_all'):
    """
    Take data from traversals and build neighbourhood_depths
    :param encoders: encoder to use for each key in the supplied dictionaries
    :param raw_arrays: expects a list of dictionaries, one for each depth, each key referring to an array of the raw
    features of the traversals
    :return:
    """

    with tf.name_scope(name) as scope:
        encoded_arrays = []
        for raw_array in raw_arrays:
            # encoded_features = [encoders[key](features_array) for key, features_array in raw_array.items()]
            all_encoded_features = []
            for key, features_array in raw_array.items():
                encoded_features = encoders[key](features_array)
                all_encoded_features.append(encoded_features)
                tf.summary.histogram(key, encoded_features)

            concatenated_encoded_features = tf.concat(all_encoded_features, -1)
            tf.summary.histogram('concat', concatenated_encoded_features)
            encoded_arrays.append(concatenated_encoded_features)

        return encoded_arrays
