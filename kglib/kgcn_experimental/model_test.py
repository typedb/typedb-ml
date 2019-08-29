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
from tensorflow.python.framework.ops import EagerTensor

from kglib.kgcn_experimental.model import TypewiseEncoder, KGCN
from kglib.kgcn_experimental.test.utils import get_call_args
import networkx as nx


def test_numpy_arrays_equal(arrays_a, arrays_b):
    for a, b in zip(arrays_a, arrays_b):
        np.testing.assert_array_equal(a, b)


class TestKGCN(unittest.TestCase):
    def test_kgcn_runs(self):
        graph = nx.MultiDiGraph()
        graph.add_node(0, type='person', input=1, solution=0)
        graph.add_edge(0, 1, type='employee', input=1, solution=0)
        graph.add_node(1, type='employment', input=1, solution=0)

        kgcn = KGCN(['person', 'employment'], ['employee'])
        kgcn([graph], [graph], num_training_iterations=50, log_every_seconds=0.5)


class TestTypewiseEncoder(unittest.TestCase):
    def test_types_encoded_by_expected_functions(self):
        tf.enable_eager_execution()
        things = np.array([[0, 0], [1, 0], [2, 0.5673]], dtype=np.float32)

        entity_relation = Mock(return_value=np.array([[0.1, 0, 0], [0.1, 0, 0]], dtype=np.float32))

        continuous_attribute = Mock(return_value=np.array([[0.9527, 0.2367, 0.7582]], dtype=np.float32))

        encoders_for_types = {entity_relation: [0, 1], continuous_attribute: [2]}

        tm = TypewiseEncoder(encoders_for_types, 3)
        encoding = tm(things)  # The function under test

        np.testing.assert_array_equal([[np.array([[0, 0], [1, 0]], dtype=np.float32)]],
                                      get_call_args(entity_relation))

        np.testing.assert_array_equal([[np.array([[2, 0.5673]], dtype=np.float32)]], get_call_args(continuous_attribute))

        expected_encoding = np.array([[0.1, 0, 0], [0.1, 0, 0], [0.9527, 0.2367, 0.7582]], dtype=np.float32)
        np.testing.assert_array_equal(expected_encoding, encoding.numpy())


class ITTypewiseEncoder(unittest.TestCase):
    def test_using_tensorflow(self):
        tf.enable_eager_execution()

        tf.reset_default_graph()
        tf.set_random_seed(1)

        things = tf.Variable(initial_value=np.array([[0, 0], [1, 0], [2, 0.5673]], dtype=np.float32))

        entity_relation = lambda x: x
        continuous_attribute = lambda x: x

        encoders_for_types = {entity_relation: [0, 1], continuous_attribute: [2]}

        tm = TypewiseEncoder(encoders_for_types, 2)
        encoded_things = tm(things)  # The function under test

        # Check that tensorflow was actually used
        self.assertEqual(EagerTensor, type(encoded_things))


if __name__ == '__main__':
    unittest.main()
