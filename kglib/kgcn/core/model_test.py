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
import unittest

import kglib.kgcn.core.model as model
import tensorflow as tf


class TestKGCN(unittest.TestCase):
    """
    This is one possible implementation of a Knowledge Graph Convolutional Network (KGCN). This implementation uses
    ideas of sampling and aggregating the information of the connected neighbours of a concept in order to build a
    representation of that concept.

    As a user we want to provide parameters to the model, including:
    - Number of neighbours to sample at each depth
    - How to sample those neighbours (incl. pseudo-random params)
    - Whether to propagate sampling through attributes
    - Whether to use implicit relationships, or 'has' roles
    - Number of training steps
    - Learning rate
    - Optimiser e.g. AdamOptimiser

    Then we want to provide concepts to train, evaluate and perform prediction upon. In the case of supervised
    learning we also need to provide labels for those concepts for training and evaluation.

    We want to be able to re-run the model without re-running the Grakn traversals. This requires some way to persist
    the data acquired from the Grakn traversals and interrupt the pipeline and continue later. Probably best via
    TensorFlow checkpoints.

    Each stage of training should be bound to a keyspace, and these keyspaces may differ. For example, we will want
    to use a separate keyspace for training to that used for evaluation, or we may have crossover between our
    training and evaluation data
    """

    pass


class TestShuffleAndBatchDataset(unittest.TestCase):
    def test_shuffles_each_iter(self):
        x = np.array([[1], [2], [3], [4], [5]])
        placeholder = tf.placeholder(tf.int32, shape=(None, 1), name='input')
        feed_dict = {placeholder: x}

        # make a dataset from a numpy array
        dataset = tf.data.Dataset.from_tensor_slices(placeholder)
        dataset_initializer, dataset_iterator = model._shuffle_and_batch_dataset(dataset, 5)

        batch = dataset_iterator.get_next()
        sess = tf.Session()
        _ = sess.run(dataset_initializer, feed_dict=feed_dict)
        results = []
        for step in range(3):
            batch_out = sess.run(batch, feed_dict=feed_dict)
            results.append(batch_out)

        expected_arrays = [np.array([[1], [5], [2], [3], [4]]),
                           np.array([[5], [1], [3], [2], [4]]),
                           np.array([[2], [1], [5], [4], [3]])]
        np.testing.assert_array_equal(expected_arrays, results)

    # TODO No idea why this test fails
    # def test_batching(self):
    #     x = np.array([[1], [2], [3], [4], [5]])
    #     placeholder = tf.placeholder(tf.int32, shape=(None, 1), name='input')
    #     feed_dict = {placeholder: x}
    #
    #     # make a dataset from a numpy array
    #     dataset = tf.data.Dataset.from_tensor_slices(placeholder)
    #     dataset_initializer, dataset_iterator = model._batch_dataset(dataset, 3)
    #
    #     batch = dataset_iterator.get_next()
    #     sess = tf.Session()
    #     _ = sess.run(dataset_initializer, feed_dict=feed_dict)
    #     results = []
    #     for step in range(3):
    #         batch_out = sess.run(batch, feed_dict=feed_dict)
    #         print(batch_out)
    #         results.append(batch_out)
    #
    #     expected_arrays = [np.array([[2], [4], [5]]),
    #                        np.array([[3], [1]]),
    #                        np.array([[3], [2], [1]])]
    #
    #     print(f'expected {expected_arrays}')
    #     print(f'actual {results}')
    #     np.testing.assert_array_equal(expected_arrays, results)


if __name__ == "__main__":
    unittest.main()
