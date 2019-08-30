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

import unittest

import numpy as np
import tensorflow as tf
from graph_nets.graphs import GraphsTuple
from mock import Mock
from tensorflow.python.framework.ops import EagerTensor

from kglib.kgcn_experimental.encode import pass_input_through_op, TypeEncoder
from kglib.kgcn_experimental.model import TypewiseEncoder, make_mlp_model
from kglib.kgcn_experimental.test.utils import get_call_args


def test_numpy_arrays_equal(arrays_a, arrays_b):
    for a, b in zip(arrays_a, arrays_b):
        np.testing.assert_array_equal(a, b)


class TestTypewiseEncoder(unittest.TestCase):
    def setUp(self):
        tf.enable_eager_execution()

    def test_types_encoded_by_expected_functions(self):
        things = np.array([[0, 0], [1, 0], [2, 0.5673]], dtype=np.float32)

        entity_relation = Mock(return_value=np.array([[0.1, 0, 0], [0.1, 0, 0]], dtype=np.float32))

        continuous_attribute = Mock(return_value=np.array([[0.9527, 0.2367, 0.7582]], dtype=np.float32))

        encoders_for_types = {lambda: entity_relation: [0, 1], lambda: continuous_attribute: [2]}

        tm = TypewiseEncoder(encoders_for_types, 3)
        encoding = tm(things)  # The function under test

        np.testing.assert_array_equal([[np.array([[0, 0], [1, 0]], dtype=np.float32)]],
                                      get_call_args(entity_relation))

        np.testing.assert_array_equal([[np.array([[2, 0.5673]], dtype=np.float32)]], get_call_args(continuous_attribute))

        expected_encoding = np.array([[0.1, 0, 0], [0.1, 0, 0], [0.9527, 0.2367, 0.7582]], dtype=np.float32)
        np.testing.assert_array_equal(expected_encoding, encoding.numpy())

    def test_basic_encoding(self):
        things = np.array([[0], [1], [2]], dtype=np.float32)

        entity_relation = Mock(return_value=np.array([[0.1, 0, 0], [0.1, 0, 0], [0.1, 0, 0]], dtype=np.float32))

        encoders_for_types = {lambda: entity_relation: [0, 1, 2]}

        tm = TypewiseEncoder(encoders_for_types, 3)
        encoding = tm(things)  # The function under test

        expected_encoding = np.array([[0.1, 0, 0], [0.1, 0, 0], [0.1, 0, 0]], dtype=np.float32)
        np.testing.assert_array_equal(expected_encoding, encoding.numpy())

    # def test_encoders_do_not_fulfil_classes(self):
    #     things = np.array([[0], [1], [2]], dtype=np.float32)
    #
    #     entity_relation = Mock(return_value=np.array([[0.1, 0, 0], [0.1, 0, 0], [0.1, 0, 0]], dtype=np.float32))
    #
    #     encoders_for_types = {entity_relation: [0, 2]}
    #
    #     tm = TypewiseEncoder(encoders_for_types, 3)
    #     encoding = tm(things)  # The function under test
    #
    #     expected_encoding = np.array([[0.1, 0, 0], [0.1, 0, 0], [0.1, 0, 0]], dtype=np.float32)
    #     np.testing.assert_array_equal(expected_encoding, encoding.numpy())
    #
    # def test_encoding_nothing(self):
    #     things = np.array([[]], dtype=np.float32)
    #
    #     entity_relation = Mock(return_value=np.array([[]], dtype=np.float32))
    #
    #     encoders_for_types = {entity_relation: [0, 2]}
    #
    #     tm = TypewiseEncoder(encoders_for_types, 3)
    #     encoding = tm(things)  # The function under test
    #
    #     expected_encoding = np.array([[0.1, 0, 0], [0.1, 0, 0], [0.1, 0, 0]], dtype=np.float32)
    #     np.testing.assert_array_equal(expected_encoding, encoding.numpy())


class ITTypewiseEncoder(unittest.TestCase):

    def setUp(self):
        tf.enable_eager_execution()

    def test_using_tensorflow(self):
        tf.reset_default_graph()
        tf.set_random_seed(1)

        things = tf.convert_to_tensor(np.array([[0, 0], [1, 0], [2, 0.5673]], dtype=np.float32))

        entity_relation = lambda x: x
        continuous_attribute = lambda x: x

        encoders_for_types = {lambda: entity_relation: [0, 1], lambda: continuous_attribute: [2]}

        tm = TypewiseEncoder(encoders_for_types, 2)
        encoded_things = tm(things)  # The function under test

        # Check that tensorflow was actually used
        self.assertEqual(EagerTensor, type(encoded_things))

    def test_from_graphtuple(self):
        inp = GraphsTuple(nodes=np.array([[1., 0.],
                                          [1., 1.],
                                          [1., 2.]]),
                          edges=np.array([[1., 0.],
                                          [1., 1.]]),
                          receivers=np.array([1, 2], dtype=np.int32),
                          senders=np.array([0, 1], dtype=np.int32),
                          globals=np.array([[0., 0., 0., 0., 0.]], dtype=np.float32),
                          n_node=np.array([3], dtype=np.int32), n_edge=np.array([2], dtype=np.int32))

        feature_length = 15
        all_edge_types = ['employee', 'employer']
        edge_type_encoder_op = lambda: make_mlp_model(latent_size=15, num_layers=2)
        edge_encoders_for_types = {lambda: TypeEncoder(len(all_edge_types), 0, edge_type_encoder_op):
                                       [i for i, _ in enumerate(all_edge_types)]}
        edge_typewise = TypewiseEncoder(edge_encoders_for_types, feature_length, name="edge_typewise_encoder")

        def edge_model():
            return pass_input_through_op(edge_typewise)

        output = edge_model()(inp.edges)


if __name__ == '__main__':
    unittest.main()
