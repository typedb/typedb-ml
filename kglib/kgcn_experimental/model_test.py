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
from mock import Mock
from tensorflow.python import constant
from tensorflow.python.framework.ops import EagerTensor

from kglib.kgcn_experimental.model import ThingModelManager
from kglib.kgcn_experimental.test.utils import get_call_args


def test_numpy_arrays_equal(arrays_a, arrays_b):
    for a, b in zip(arrays_a, arrays_b):
        np.testing.assert_array_equal(a, b)


class TestThingModelManager(unittest.TestCase):
    def test_types_encoded_by_expected_functions(self):
        things = np.array([[0, 0], [1, 0], [2, 0.5673]], dtype=np.float64)

        # entity_relation = EntityRelationEncoder()
        entity_relation = Mock(return_value=np.array([0.1, 0, 0], dtype=np.float64))

        # continuous_attribute = ContinuousAttributeEncoder()
        continuous_attribute = Mock(return_value=np.array([0.9527, 0.2367, 0.7582], dtype=np.float64))

        encoders_by_type = [entity_relation, entity_relation, continuous_attribute]

        tm = ThingModelManager(encoders_by_type, 3)
        tm(things)  # The function under test

        np.testing.assert_array_equal([[np.array([0, 0], dtype=np.float64)], [np.array([1, 0], dtype=np.float64)]],
                                      get_call_args(entity_relation))

        np.testing.assert_array_equal([[np.array([2, 0.5673], dtype=np.float64)]], get_call_args(continuous_attribute))


class ITThingModelManager(unittest.TestCase):
    def test_using_tensorflow(self):
        tf.enable_eager_execution()

        tf.reset_default_graph()
        tf.set_random_seed(1)

        things = constant(np.array([[0, 0], [1, 0], [2, 0.5673]], dtype=np.float64))

        entity_relation = lambda x: x
        continuous_attribute = lambda x: x

        encoders_by_type = [entity_relation, entity_relation, continuous_attribute]

        tm = ThingModelManager(encoders_by_type, 2)
        encoded_things = tm(things)  # The function under test

        # Check that tensorflow was actually used
        self.assertEqual(EagerTensor, type(encoded_things))


# class TestKGIndependent(unittest.TestCase):
#     def test_type_behaviour(self):
#         nodes = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0.5673]], dtype=np.float64)
#         exp_nodes = np.array([[0, 1], [1, 0], [1, 0]], dtype=np.float64)
#         edges = np.array([[0, 1], [1, 0]], dtype=np.float64)
#         exp_edges = np.array([[1, 0], [1, 0]], dtype=np.float64)
#         node_types = np.array(['person', '@has-age', 'age'])
#         node_meta_types = np.array(['entity', 'relation', 'attribute'])
#         edge_types = np.array(['@has-age-owner', '@has-age-value'])
#
#         globals = None
#         senders = np.array([0, 1])
#         receivers = np.array([1, 2])
#         n_node = np.array([3])
#         n_edge = np.array([2])
#
#         graph = GraphsTuple(nodes=nodes,
#                             edges=edges,
#                             globals=globals,
#                             receivers=receivers,
#                             senders=senders,
#                             n_node=n_node,
#                             n_edge=n_edge,
#                             node_types=node_types,
#                             meta_types=node_meta_types,
#                             edge_types=edge_types)
#
#         expected_output = GraphsTuple(nodes=exp_nodes,
#                                       edges=exp_edges,
#                                       globals=globals,
#                                       receivers=receivers,
#                                       senders=senders,
#                                       n_node=n_node,
#                                       n_edge=n_edge,
#                                       node_types=node_types,
#                                       meta_types=node_meta_types,
#                                       edge_types=edge_types)


if __name__ == '__main__':
    unittest.main()