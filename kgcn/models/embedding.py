import numpy as np
import tensorflow as tf

import kgcn.models.aggregation as agg


class Embedder:
    def __init__(self, feature_lengths, aggregated_length, output_length, neighbourhood_sizes,
                 normalisation=tf.nn.l2_normalize):

        self._neighbourhood_sizes = neighbourhood_sizes
        self._aggregated_length = aggregated_length
        self._feature_lengths = feature_lengths
        self._output_length = output_length
        self._combined_lengths = [feature_length + self._aggregated_length for feature_length in self._feature_lengths]
        self._normalisation = normalisation

    def __call__(self, neighbourhoods):

        # TODO pass through params for aggregators, combiners and normalisers
        aggregators = [agg.Aggregate(self._aggregated_length, name=f'aggregate_{i}') for i in
                       range(len(self._neighbourhood_sizes))]

        combiners = []
        for i in range(len(self._neighbourhood_sizes)):

            weights = initialise_glorot_weights((self._combined_lengths[i], self._output_length),
                                                name=f'weights_{i}')

            if i + 1 == len(self._neighbourhood_sizes):
                combiner = agg.Combine(weights, activation=lambda x: x, name=f'combine_{i}_linear')
            else:
                combiner = agg.Combine(weights, activation=tf.nn.relu, name=f'combine_{i}_relu')
            combiners.append(combiner)

        normalisers = [agg.normalise for _ in range(len(self._neighbourhood_sizes))]

        full_representation = agg.chain_aggregate_combine(neighbourhoods, aggregators, combiners, normalisers)
        # full_representation = tf.nn.l2_normalize(full_representation, -1)
        return full_representation


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
        return tf.Variable(initial_value=initial, name=name)
