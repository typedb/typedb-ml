#
#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.
#

import numpy as np
import tensorflow as tf

import kglib.kgcn.embed.aggregate as aggregate
import kglib.kgcn.embed.combine as combine


class Embedder:
    def __init__(self, k_hop_feature_sizes, aggregated_size, embedding_size, neighbourhood_sizes,
                 normalisation=tf.nn.l2_normalize):

        self._neighbourhood_sizes = neighbourhood_sizes
        self._aggregated_size = aggregated_size
        self._k_hop_feature_sizes = k_hop_feature_sizes
        self._embedding_size = embedding_size
        self._combined_size = [feature_size + self._aggregated_size for feature_size in self._k_hop_feature_sizes]
        self._normalisation = normalisation

    def __call__(self, neighbourhoods):

        # TODO pass through params for aggregators, combiners and normalisers
        aggregators = [aggregate.Aggregator(self._aggregated_size, name=f'aggregate_{i}') for i in
                       range(len(self._neighbourhood_sizes))]

        combiners = []
        for i in range(len(self._neighbourhood_sizes)):

            # weights = initialise_glorot_weights((self._combined_sizes[i], self._embedding_size),
            #                                     name=f'weights_{i}')

            if i + 1 == len(self._neighbourhood_sizes):
                # combiner = combine.Combiner(weights, activation=lambda x: x, name=f'combine_{i}_linear')
                combiner = combine.DenseCombiner(self._embedding_size, activation=lambda x: x, name=f'combine_{i}_linear')
            else:
                combiner = combine.DenseCombiner(self._embedding_size, activation=tf.nn.relu, name=f'combine_{i}_relu')
            combiners.append(combiner)

        normalisers = [normalise for _ in range(len(self._neighbourhood_sizes))]

        full_representation = chain_aggregate_combine(neighbourhoods, aggregators, combiners, normalisers)
        # full_representation = tf.nn.l2_normalize(full_representation, -1)
        return full_representation


# TODO Presently unused
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


def normalise(features, normalise_op=tf.nn.l2_normalize):
    normalised = normalise_op(features, axis=-1)
    tf.summary.histogram('normalised', normalised)
    return normalised


def chain_aggregate_combine(neighbourhoods, aggregators, combiners, normalisers, name=None):
    # zip(range(len(r)-1, -1, -1), reversed(r))  # To iterate in reverse with an index. Doesn't play well with zip()

    with tf.name_scope(name, default_name="chain_aggregate_combine") as scope:
        for i, (aggregator, combiner, normaliser) in enumerate(zip(aggregators, combiners, normalisers)):
            if i == 0:
                neighbour_representations = neighbourhoods[i]
            else:
                neighbour_representations = full_representations

            neighbourhood_representations = aggregator(neighbour_representations)

            targets = neighbourhoods[i + 1]
            full_representations = normaliser(combiner(targets, neighbourhood_representations))

        return full_representations
