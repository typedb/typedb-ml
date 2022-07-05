#
#  Copyright (C) 2022 Vaticle
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

from kglib.utils.graph.test.case import GraphTestCase

from kglib.kgcn_data_loader.utils import apply_logits_to_graphs


class TestApplyLogitsToGraphs(GraphTestCase):
    def test_logits_applied_as_expected(self):

        graph = nx.MultiDiGraph(name=0)
        graph.add_node(0)
        graph.add_node(1)
        graph.add_edge(0, 1)

        logits_graph = nx.MultiDiGraph(name=0)
        logits_graph.add_node(0, features=np.array([0.2, 0.3, 0.01]))
        logits_graph.add_node(1, features=np.array([0.56, -0.04, 0.05]))
        logits_graph.add_edge(0, 1, features=np.array([0.5, 0.008, -0.1]))
        logits_graph.add_edge(1, 0, features=np.array([0.5, 0.008, -0.1]))

        expected_graph = nx.MultiDiGraph(name=0)
        expected_graph.add_node(0, logits=[0.2, 0.3, 0.01])
        expected_graph.add_node(1, logits=[0.56, -0.04, 0.05])
        expected_graph.add_edge(0, 1, logits=[0.5, 0.008, -0.1])

        graph_with_logits = apply_logits_to_graphs(graph, logits_graph)

        self.assertGraphsEqual(expected_graph, graph_with_logits)


if __name__ == '__main__':
    unittest.main()
