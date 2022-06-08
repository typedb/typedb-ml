#
#  Copyright (C) 2021 Vaticle
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

from kglib.utils.typedb.object.thing import Thing
from kglib.utils.graph.test.case import GraphTestCase

from kglib.kgcn_data_loader.utils import duplicate_edges_in_reverse, apply_logits_to_graphs


class TestDuplicateEdgesInReverse(GraphTestCase):

    def test_edges_are_duplicated_as_expected(self):
        graph = nx.MultiDiGraph(name=0)

        p0 = Thing('V123', 'person', 'entity')
        p1 = Thing('V456', 'person', 'entity')
        par0 = Thing('V789', 'parentship', 'relation')

        # people
        graph.add_node(p0, type='person', solution=1)
        graph.add_node(p1, type='person', solution=1)

        # parentships
        graph.add_node(par0, type='parentship', solution=1)
        graph.add_edge(par0, p0, type='parent', solution=1)
        graph.add_edge(par0, p1, type='child', solution=1)

        duplicate_edges_in_reverse(graph)

        expected_graph = nx.MultiDiGraph(name=0)

        # people
        expected_graph.add_node(p0, type='person', solution=1)
        expected_graph.add_node(p1, type='person', solution=1)

        # parentships
        expected_graph.add_node(par0, type='parentship', solution=1)
        expected_graph.add_edge(par0, p0, type='parent', solution=1)
        expected_graph.add_edge(par0, p1, type='child', solution=1)

        # Duplicates
        expected_graph.add_edge(p0, par0, type='parent', solution=1)
        expected_graph.add_edge(p1, par0, type='child', solution=1)
        self.assertGraphsEqual(expected_graph, graph)


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
