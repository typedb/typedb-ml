import tensorflow as tf

import kgcn.src.preprocessing.to_array.date_to_unixtime as date


def preprocess(raw_arrays):
    """
    :param raw_arrays: numpy arrays to be transformed into a format such that can be fed into a TensorFlow op graph
    :return: tensor-ready arrays
    """

    preprocessors = {'role_type': lambda x: tf.convert_to_tensor(x, dtype=tf.string),
                     'role_direction': lambda x: x,
                     'neighbour_type': lambda x: tf.convert_to_tensor(x, dtype=tf.string),
                     'neighbour_data_type': lambda x: x,
                     'neighbour_value_long': lambda x: x,
                     'neighbour_value_double': lambda x: x,
                     'neighbour_value_boolean': lambda x: x,
                     'neighbour_value_date': date.datetime_to_unixtime,
                     'neighbour_value_string': lambda x: x}

    preprocessed_raw_arrays = []
    for raw_array in raw_arrays:
        preprocessed_features = {}
        for key, features_array in raw_array.items():
            preprocessed_features[key] = preprocessors[key](features_array)
        preprocessed_raw_arrays.append(preprocessed_features)

    return preprocessed_raw_arrays
