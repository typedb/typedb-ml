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

from kglib.kgcn_experimental.data import create_graph
from kglib.kgcn_experimental.encode import encode_types_one_hot, graph_to_input_target


class TestGraphToInputTarget(unittest.TestCase):
    def test_number_of_nodes_in_outputs_is_correct(self):
        all_node_types = ['person', 'parentship', 'siblingship']
        all_edge_types = ['parent', 'child', 'sibling']
        graph = create_graph(1)
        encode_types_one_hot(graph, all_node_types, all_edge_types, attribute='one_hot_type', type_attribute='type')

        expected_n_nodes = graph.number_of_nodes()

        input_graph, target_graph = graph_to_input_target(graph)
        self.assertEqual(expected_n_nodes, input_graph.number_of_nodes())
        self.assertEqual(expected_n_nodes, target_graph.number_of_nodes())

    def test_number_of_edges_in_outputs_is_correct(self):
        all_node_types = ['person', 'parentship', 'siblingship']
        all_edge_types = ['parent', 'child', 'sibling']
        graph = create_graph(1)
        encode_types_one_hot(graph, all_node_types, all_edge_types, attribute='one_hot_type', type_attribute='type')

        expected_n_edges = graph.number_of_edges()

        input_graph, target_graph = graph_to_input_target(graph)
        self.assertEqual(expected_n_edges, input_graph.number_of_edges())
        self.assertEqual(expected_n_edges, target_graph.number_of_edges())


if __name__ == "__main__":
    unittest.main()
