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
from tensorflow.contrib import layers as layers


# TODO Presently unused, should be removed if unnecessary
class Combiner:
    def __init__(self, weights, activation=tf.nn.relu, name=None):
        """
        :param weights: weight matrix, shape (combined_size, output_size)
        :param activation: activation function performed on the
        :param name: Name for the operation (optional).
        """
        self._weights = weights
        self._activation = activation
        self._name = name

    def __call__(self, target_features, neighbour_representations):
        """
        Combiner the results of neighbour aggregation with the features of target nodes. Combiner using concatenation,
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


class DenseCombiner:
    def __init__(self, output_size, activation=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer(),
                 regularizer=layers.l2_regularizer(scale=0.1), use_bias=True, name=None):
        """
        :param activation: activation function performed on the
        :param name: Name for the operation (optional).
        """
        self._use_bias = use_bias
        self._regularizer = regularizer
        self._initializer = initializer
        self._output_size = output_size
        self._activation = activation
        self._name = name

    def __call__(self, target_features, neighbour_representations):
        """
        Combiner the results of neighbour aggregation with the features of target nodes. Combiner using concatenation,
        multiplication with a weight matrix (GCN approach) and process with some activation function
        :param target_features: the features of the target nodes
        :param neighbour_representations: the representations of the neighbours of the target nodes, one representation
        for each target
        :return: full representations of target nodes
        """
        with tf.name_scope(self._name, default_name="combine") as scope:
            concatenated_features = tf.concat([target_features, neighbour_representations], axis=-1)

            dense_layer = tf.layers.Dense(units=self._output_size, activation=self._activation, use_bias=False,
                                          kernel_initializer=self._initializer, kernel_regularizer=self._regularizer,
                                          name=f'dense_layer_{self._name}')

            # tf.summary.histogram(self._name + '/dense/bias', dense_layer.bias)
            # tf.summary.histogram(self._name + '/dense/kernel', dense_layer.kernel)

            dense_output = dense_layer(concatenated_features)

            tf.summary.histogram(self._name + '/dense_output', dense_output)

            return dense_output
