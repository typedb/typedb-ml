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

from kglib.kgcn_experimental.network.core import KGCN
from kglib.kgcn_experimental.learn.learn import KGCNLearner


class ITKGCNLearner(unittest.TestCase):
    def test_learner_runs(self):
        graph = nx.MultiDiGraph()
        graph.add_node(0, type='person', encoded_value=0, input=1, solution=0)
        graph.add_edge(0, 1, type='employee', encoded_value=0, input=1, solution=0)
        graph.add_node(1, type='employment', encoded_value=0, input=1, solution=0)
        graph.add_edge(1, 2, type='employer', encoded_value=0, input=1, solution=0)
        graph.add_node(2, type='company', encoded_value=0, input=1, solution=0)

        attr_embedders = {lambda: lambda x: tf.constant(np.zeros((3, 6), dtype=np.float32)): [0, 1, 2]}

        kgcn = KGCN(3, 2, 5, 6, attr_embedders, edge_output_size=3, node_output_size=3)

        learner = KGCNLearner(kgcn, ['person', 'employment', 'company'], ['employee', 'employer'])
        learner([graph], [graph],
                num_processing_steps_tr=2,
                num_processing_steps_ge=2,
                num_training_iterations=50,
                log_every_seconds=0.5)


if __name__ == "__main__":
    tf.enable_eager_execution()
    unittest.main()
