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

import tensorflow as tf
import tensorflow.contrib.layers as layers


class Aggregate:
    def __init__(self, aggregated_length, reduction=tf.reduce_max, activation=tf.nn.relu, dropout=0.7,
                 initializer=tf.contrib.layers.xavier_initializer(), regularizer=layers.l2_regularizer(scale=0.1),
                 name=None):
        """
        :param aggregated_length: the number of elements in the representation created
        :param reduction: order-independent method of pooling the response for each neighbour
        :param activation: activation function for the included dense layer
        :param dropout: quantity of dropout regularisation on the output of the included dense layer
        :param regularizer: regularisation for the dense layer
        :param initializer: initializer for the weights of the dense layer
        :param name: Name for the operation (optional).
        """
        self._aggregated_length = aggregated_length
        self._reduction = reduction
        self._activation = activation
        self._dropout = dropout
        self._initializer = initializer
        self._regularizer = regularizer
        self._name = name

    def __call__(self, neighbour_features):
        """
        Take a tensor that describes the features (aggregated or otherwise) of a set of neighbours and aggregate
        them through a dense layer and order-independent pooling/reduction

        :param neighbour_features: the neighbours' features, shape (num_neighbours, neighbour_feat_length)
        :return: aggregated representation of neighbours, shape (1, aggregated_length)
        """

        with tf.name_scope(self._name, default_name="aggregate") as scope:

            dense_layer = tf.layers.Dense(units=self._aggregated_length, activation=self._activation, use_bias=True,
                                          kernel_initializer=self._initializer, kernel_regularizer=self._regularizer,
                                          name=f'dense_layer_{self._name}')

            dense_output = dense_layer(neighbour_features)

            # tf.summary.histogram(self._name + '/dense/weights', dense_layer.weights)
            tf.summary.histogram(self._name + '/dense/bias', dense_layer.bias)
            tf.summary.histogram(self._name + '/dense/kernel', dense_layer.kernel)

            tf.summary.histogram(self._name + '/dense_output', dense_output)

            # Use dropout on output from the dense layer to prevent overfitting
            regularised_output = tf.nn.dropout(dense_output, self._dropout)
            tf.summary.histogram(self._name + '/regularised_output', regularised_output)

            # Use max-pooling (or similar) to aggregate the results for each neighbour. This is an important operation
            # since the order of the neighbours isn't considered, which is a property we need Note that this is reducing
            # not pooling, which is equivalent to having a pool size of num_neighbours
            reduced_output = self._reduction(regularised_output, axis=1)
            tf.summary.histogram(self._name + '/reduced_output', reduced_output)

            # If reducing reduced rank to 1, then add a dimension so that we continue to deal with matrices not vectors
            rank = tf.rank(reduced_output)
            if tf.executing_eagerly():
                evaluated_rank = rank.numpy()
            else:
                evaluated_rank = rank

            if evaluated_rank == 1:
                reduced_output = tf.expand_dims(reduced_output, 0)

            # # Get the output from shape (1, neighbour_feat_length) to (neighbour_feat_length, 1)
            # final_output = tf.transpose(reduced_output)

            return reduced_output


class Combine:
    def __init__(self, weights, activation=tf.nn.relu, name=None):
        """
        :param weights: weight matrix, shape (combined_length, output_length)
        :param activation: activation function performed on the
        :param name: Name for the operation (optional).
        """
        self._weights = weights
        self._activation = activation
        self._name = name

    def __call__(self, target_features, neighbour_representations):
        """
        Combine the results of neighbour aggregation with the features of target nodes. Combine using concatenation,
        multiplication with a weight matrix (GCN approach) and process with some activation function
        :param target_features: the features of the target nodes
        :param neighbour_representations: the representations of the neighbours of the target nodes, one representation
        for each target
        :return: full representations of target nodes
        """
        with tf.name_scope(self._name, default_name="combine") as scope:
            tf.summary.histogram(self._name + '/combine_weights', self._weights)
            concatenated_features = tf.concat([target_features, neighbour_representations], axis=-1)

            weighted_output = tf.tensordot(concatenated_features, self._weights, (-1, 0), name='apply_weights')
            tf.summary.histogram(self._name + '/combine_weighted_output', weighted_output)

            activated_output = self._activation(weighted_output)
            tf.summary.histogram(self._name + '/activated_output', activated_output)

            return activated_output


class DenseCombine:
    def __init__(self, output_length, activation=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer(),
                 regularizer=layers.l2_regularizer(scale=0.1), use_bias=True, name=None):
        """
        :param activation: activation function performed on the
        :param name: Name for the operation (optional).
        """
        self._use_bias = use_bias
        self._regularizer = regularizer
        self._initializer = initializer
        self._output_length = output_length
        self._activation = activation
        self._name = name

    def __call__(self, target_features, neighbour_representations):
        """
        Combine the results of neighbour aggregation with the features of target nodes. Combine using concatenation,
        multiplication with a weight matrix (GCN approach) and process with some activation function
        :param target_features: the features of the target nodes
        :param neighbour_representations: the representations of the neighbours of the target nodes, one representation
        for each target
        :return: full representations of target nodes
        """
        with tf.name_scope(self._name, default_name="combine") as scope:
            concatenated_features = tf.concat([target_features, neighbour_representations], axis=-1)

            dense_layer = tf.layers.Dense(units=self._output_length, activation=self._activation, use_bias=False,
                                          kernel_initializer=self._initializer, kernel_regularizer=self._regularizer,
                                          name=f'dense_layer_{self._name}')

            # tf.summary.histogram(self._name + '/dense/bias', dense_layer.bias)
            # tf.summary.histogram(self._name + '/dense/kernel', dense_layer.kernel)

            dense_output = dense_layer(concatenated_features)

            tf.summary.histogram(self._name + '/dense_output', dense_output)

            return dense_output


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


