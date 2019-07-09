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

from kglib.kgcn.core.ingest.traverse.data.context.neighbour import Thing, GraknEdge
from graph.label.label import label_nodes, label_direct_roles


class TestLabelNodes(unittest.TestCase):
    def test_standard_graph_nodes_labelled_as_expected(self):
        person = Thing('V123', 'person', 'entity')
        parentship = Thing('V567', 'parentship', 'relation')
        name = Thing('V987', 'name', 'attribute', data_type='string', value='Bob')
        grakn_graph = nx.MultiDiGraph()
        grakn_graph.add_edge(parentship, person, type='child')
        grakn_graph.add_edge(parentship, person, type='parent')
        grakn_graph.add_edge(person, name, type='has')

        labels_to_apply = {'gt': 1}
        elements_to_label = [person, name]
        label_nodes(grakn_graph, elements_to_label, labels_to_apply=labels_to_apply)

        self.assertDictEqual(grakn_graph.nodes[person], labels_to_apply)
        self.assertDictEqual(grakn_graph.nodes[name], labels_to_apply)

    def test_math_graph_nodes_labelled_as_expected(self):
        person = Thing('V123', 'person', 'entity')
        parentship = Thing('V567', 'parentship', 'relation')
        name = Thing('V987', 'name', 'attribute', data_type='string', value='Bob')
        parent = GraknEdge(parentship, person, 'parent')
        child = GraknEdge(parentship, person, 'child')
        grakn_graph = nx.MultiDiGraph()

        grakn_graph.add_edge(parentship, child, type='relates')
        grakn_graph.add_edge(parentship, parent, type='relates')
        grakn_graph.add_edge(person, parent, type='plays')
        grakn_graph.add_edge(person, child, type='plays')
        grakn_graph.add_edge(person, name, type='has')

        labels_to_apply = {'gt': 1}
        elements_to_label = [person, name, parent, parentship]
        label_nodes(grakn_graph, elements_to_label, labels_to_apply=labels_to_apply)

        self.assertDictEqual(grakn_graph.nodes[person], labels_to_apply)
        self.assertDictEqual(grakn_graph.nodes[name], labels_to_apply)


class TestLabelEdges(unittest.TestCase):
    def test_standard_graph_edges_labelled_as_expected(self):
        person = Thing('V123', 'person', 'entity')
        parentship = Thing('V567', 'parentship', 'relation')
        name = Thing('V987', 'name', 'attribute', data_type='string', value='Bob')
        grakn_graph = nx.MultiDiGraph()
        grakn_graph.add_edge(parentship, person, type='parent')
        grakn_graph.add_edge(parentship, person, type='child')
        grakn_graph.add_edge(person, name, type='has')
        
        parent = GraknEdge(parentship, person, 'parent')
        child = GraknEdge(parentship, person, 'child')

        labels_to_apply = {'gt': 1}
        role_edges_to_label = [
            parent,
            child
        ]
        label_direct_roles(grakn_graph, role_edges_to_label, labels_to_apply=labels_to_apply)

        self.assertDictEqual(grakn_graph.edges[parentship, person, 0], {'type': 'parent', 'gt': 1})
        self.assertDictEqual(grakn_graph.edges[parentship, person, 1], {'type': 'child', 'gt': 1})


if __name__ == "__main__":
    unittest.main()
