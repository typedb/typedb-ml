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


class Aggregator:
    def __init__(self, aggregated_size, reduction=tf.reduce_max, activation=tf.nn.relu, dropout=0.7,
                 initializer=tf.contrib.layers.xavier_initializer(), regularizer=layers.l2_regularizer(scale=0.1),
                 name=None):
        """
        :param aggregated_size: the number of elements in the representation created
        :param reduction: order-independent method of pooling the response for each neighbour
        :param activation: activation function for the included dense layer
        :param dropout: quantity of dropout regularisation on the output of the included dense layer
        :param regularizer: regularisation for the dense layer
        :param initializer: initializer for the weights of the dense layer
        :param name: Name for the operation (optional).
        """
        self._aggregated_size = aggregated_size
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

        :param neighbour_features: the neighbours' features, shape (num_neighbours, neighbour_feat_size)
        :return: aggregated representation of neighbours, shape (1, aggregated_size)
        """

        with tf.name_scope(self._name, default_name="aggregate") as scope:

            dense_layer = tf.layers.Dense(units=self._aggregated_size, activation=self._activation, use_bias=True,
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
