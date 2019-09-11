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

from kglib.kgcn_experimental.learn.learn import KGCNLearner
from kglib.kgcn_experimental.models.attribute import BlankAttribute
from kglib.kgcn_experimental.models.core import KGCN


class ITKGCNLearner(unittest.TestCase):
    def test_learner_runs(self):
        graph = nx.MultiDiGraph()
        # TODO Remove 'input' and 'solution' fields, only needed for plotting which should be separated
        graph.add_node(0, type='person', features=np.array([0, 1, 2], dtype=np.float32), input=1, solution=0)
        graph.add_edge(1, 0, type='employee', features=np.array([0, 1, 2], dtype=np.float32), input=1, solution=0)
        graph.add_node(1, type='employment', features=np.array([0, 1, 2], dtype=np.float32), input=1, solution=0)
        graph.add_edge(1, 2, type='employer', features=np.array([0, 1, 2], dtype=np.float32), input=1, solution=0)
        graph.add_node(2, type='company', features=np.array([0, 1, 2], dtype=np.float32), input=1, solution=0)
        graph.graph['features'] = np.zeros(5, dtype=np.float32)

        attr_embedding_dim = 6
        attr_embedders = {lambda: BlankAttribute(attr_embedding_dim): [0, 1, 2]}

        kgcn = KGCN(3, 2, 5, attr_embedding_dim, attr_embedders, edge_output_size=3, node_output_size=3)

        learner = KGCNLearner(kgcn, num_processing_steps_tr=2, num_processing_steps_ge=2)

        learner([graph], [graph], [graph], [graph], num_training_iterations=50)


if __name__ == "__main__":
    unittest.main()
