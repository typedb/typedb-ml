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

import networkx as nx
import numpy as np
import tensorflow as tf
from graph_nets.graphs import GraphsTuple

from kglib.kgcn_experimental.network.model import KGCN


class TestKGCN(unittest.TestCase):
    def test_kgcn_runs(self):
        graph = nx.MultiDiGraph()
        graph.add_node(0, type='person', value=0, input=1, solution=0)
        graph.add_edge(0, 1, type='employee', value=0, input=1, solution=0)
        graph.add_node(1, type='employment', value=0, input=1, solution=0)
        graph.add_edge(1, 2, type='employer', value=0, input=1, solution=0)
        graph.add_node(2, type='company', value=0, input=1, solution=0)

        kgcn = KGCN(['person', 'employment', 'company'], ['employee', 'employer'], 5, 6,
                    attr_encoders={lambda: lambda x: tf.constant(np.zeros((3, 6), dtype=np.float32)): [0, 1, 2]})
        kgcn([graph], [graph],
             num_processing_steps_tr=2,
             num_processing_steps_ge=2,
             num_training_iterations=50,
             log_every_seconds=0.5)


class TestModel(unittest.TestCase):

    def test_model_runs(self):
        tf.enable_eager_execution()

        graph = GraphsTuple(nodes=tf.convert_to_tensor(np.array([[1, 2, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float32)),
                            edges=tf.convert_to_tensor(np.array([[1, 0, 0], [1, 0, 0]], dtype=np.float32)),
                            globals=tf.convert_to_tensor(np.array([[0, 0, 0, 0, 0]], dtype=np.float32)),
                            receivers=tf.convert_to_tensor(np.array([1, 2], dtype=np.int32)),
                            senders=tf.convert_to_tensor(np.array([0, 1], dtype=np.int32)),
                            n_node=tf.convert_to_tensor(np.array([3], dtype=np.int32)),
                            n_edge=tf.convert_to_tensor(np.array([2], dtype=np.int32)))

        kgcn = KGCN(['person', 'employment', 'company'], ['employee', 'employer'], 5, 6,
                    {lambda: lambda x: tf.constant(np.zeros((3, 6), dtype=np.float32)): [0, 1, 2]})
        model = kgcn._build()
        model(graph, 2)


if __name__ == "__main__":
    tf.enable_eager_execution()
    unittest.main()
