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

from kglib.kgcn_experimental.models.core import KGCN


class ITKGCN(unittest.TestCase):

    def test_kgcn_runs(self):
        tf.enable_eager_execution()

        graph = GraphsTuple(nodes=tf.convert_to_tensor(np.array([[1, 2, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float32)),
                            edges=tf.convert_to_tensor(np.array([[1, 0, 0], [1, 0, 0]], dtype=np.float32)),
                            globals=tf.convert_to_tensor(np.array([[0, 0, 0, 0, 0]], dtype=np.float32)),
                            receivers=tf.convert_to_tensor(np.array([1, 2], dtype=np.int32)),
                            senders=tf.convert_to_tensor(np.array([0, 1], dtype=np.int32)),
                            n_node=tf.convert_to_tensor(np.array([3], dtype=np.int32)),
                            n_edge=tf.convert_to_tensor(np.array([2], dtype=np.int32)))

        attr_embedders = {lambda: lambda x: tf.constant(np.zeros((3, 6), dtype=np.float32)): [0, 1, 2]}
        kgcn = KGCN(3, 2, 5, 6, attr_embedders, edge_output_size=3, node_output_size=3)

        kgcn(graph, 2)


if __name__ == "__main__":
    tf.enable_eager_execution()
    unittest.main()
