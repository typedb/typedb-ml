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

import networkx as nx
import numpy as np
from graph_nets.graphs import GraphsTuple
from graph_nets.utils_np import graphs_tuple_to_networkxs

from kglib.graph.test.case import GraphTestCase
from kglib.kgcn.core.ingest.traverse.data.context.neighbour import Thing
from kglib.kgcn_experimental.graph_utils.prepare import duplicate_edges_in_reverse, apply_logits_to_graphs


class TestDuplicateEdgesInReverse(GraphTestCase):

    def test_edges_are_duplicated_as_expected(self):
        graph = nx.MultiDiGraph(name=0)

        p0 = Thing('V123', 'person', 'entity')
        p1 = Thing('V456', 'person', 'entity')
        par0 = Thing('V789', 'parentship', 'relation')

        # people
        graph.add_node(p0, type='person', input=1, solution=1)
        graph.add_node(p1, type='person', input=1, solution=1)

        # parentships
        graph.add_node(par0, type='parentship', input=1, solution=1)
        graph.add_edge(par0, p0, type='parent', input=1, solution=1)
        graph.add_edge(par0, p1, type='child', input=1, solution=1)

        duplicate_edges_in_reverse(graph)

        expected_graph = nx.MultiDiGraph(name=0)

        # people
        expected_graph.add_node(p0, type='person', input=1, solution=1)
        expected_graph.add_node(p1, type='person', input=1, solution=1)

        # parentships
        expected_graph.add_node(par0, type='parentship', input=1, solution=1)
        expected_graph.add_edge(par0, p0, type='parent', input=1, solution=1)
        expected_graph.add_edge(par0, p1, type='child', input=1, solution=1)

        # Duplicates
        expected_graph.add_edge(p0, par0, type='parent', input=1, solution=1)
        expected_graph.add_edge(p1, par0, type='child', input=1, solution=1)
        self.assertGraphsEqual(expected_graph, graph)


class TestApplyLogitsToGraphs(GraphTestCase):
    def test_logits_applied_as_expected(self):

        graph = nx.MultiDiGraph(name=0)
        graph.add_node(0)
        graph.add_node(1)
        graph.add_edge(0, 1)

        nodes = np.array([[0.2, 0.3, 0.01], [0.56, -0.04, 0.05]], dtype=np.float32)
        edges = np.array([[0.5, 0.008, -0.1],
                          [0.5, 0.008, -0.1]], dtype=np.float32)

        globals = None
        senders = np.array([0, 1])
        receivers = np.array([1, 0])
        n_node = np.array([2])
        n_edge = np.array([2])

        # TODO Once the model doesn't require duplicate reversed edges, then the GraphsTuple will look like this, as it
        #  should:
        # globals = None
        # senders = np.array([0])
        # receivers = np.array([1])
        # n_node = np.array([2])
        # n_edge = np.array([1])

        graphstuple = GraphsTuple(nodes=nodes,
                                  edges=edges,
                                  globals=globals,
                                  receivers=receivers,
                                  senders=senders,
                                  n_node=n_node,
                                  n_edge=n_edge)

        expected_graph = nx.MultiDiGraph(name=0)
        expected_graph.add_node(0, logits=[0.2, 0.3, 0.01])
        expected_graph.add_node(1, logits=[0.56, -0.04, 0.05])
        expected_graph.add_edge(0, 1, logits=[0.5, 0.008, -0.1])

        graphs = apply_logits_to_graphs([graph], graphs_tuple_to_networkxs(graphstuple))

        self.assertGraphsEqual(expected_graph, graphs[0])
