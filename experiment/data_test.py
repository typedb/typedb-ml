#
#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  'License'); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.
#

import unittest

import networkx as nx

import experiment.data as data


class TestDataGeneration(unittest.TestCase):
    def test_all_people_are_in_3_different_relations_with_each_other_person(self):
        num_people = 3
        G = data.generate_graph(num_people)
        expected_relation_types = ['parentship'] * 4
        expected_relation_types.extend(['siblingship'] * 2)
        for node in G.nodes:
            if G.nodes[node]['type'] == 'person':
                neighbours = nx.all_neighbors(G, node)
                relation_types = []
                for neighbour in neighbours:
                    relation_types.append(G.nodes[neighbour]['type'])
                self.assertListEqual(sorted(relation_types), sorted(expected_relation_types))

    def test__all_parentships_have_correct_roles(self):
        num_people = 3
        G = data.generate_graph(num_people)
        for node in G.nodes:
            if G.nodes[node]['type'] == 'parentship':
                role_types = [roleplayer_data['type'] for roleplayer, roleplayer_data in G[node].items()]
                self.assertListEqual(sorted(role_types), sorted(['parent', 'child']))

    def test__all_siblingships_have_correct_roles(self):
        num_people = 3
        G = data.generate_graph(num_people)
        for node in G.nodes:
            if G.nodes[node]['type'] == 'siblingship':
                role_types = [roleplayer_data['type'] for roleplayer, roleplayer_data in G[node].items()]
                self.assertListEqual(role_types, ['sibling', 'sibling'])


class TestBaseLabelling(unittest.TestCase):
    def test_node_labels_added_correctly(self):
        num_people = 3
        G = data.generate_graph(num_people)
        data.add_base_labels(G, 'input')
        expected_node_input_labels = [0] * len(list(G.nodes))
        expected_node_input_labels[:num_people] = [1] * num_people

        input_labels = [G.nodes[node]['input'] for node in G.nodes]

        self.assertListEqual(input_labels, expected_node_input_labels)

    def test_edge_labels_added_correctly(self):
        num_people = 3
        G = data.generate_graph(num_people)
        data.add_base_labels(G, 'input')
        expected_edge_input_labels = [0] * len(list(G.edges))

        input_labels = [G.edges[edge]['input'] for edge in G.edges]

        self.assertListEqual(input_labels, expected_edge_input_labels)


class TestFindRelationNode(unittest.TestCase):
    def test_parentship_found_correctly(self):
        num_people = 3
        G = data.generate_graph(num_people)
        data.add_base_labels(G, 'input')

        parentship_node = data.find_relation_node(G, 'parentship', ['parent', 'child'], [0, 1])
        self.assertEqual(parentship_node, 3)


class TestAddParentship(unittest.TestCase):
    def test_parentship_node_and_role_edges_labelled(self):
        num_people = 3
        G = data.generate_graph(num_people)
        data.add_base_labels(G, 'input')

        self.assertEqual(G.nodes[3]['input'], 0)
        self.assertEqual(G.edges[3, 0]['input'], 0)
        self.assertEqual(G.edges[3, 1]['input'], 0)

        data.add_parentship(G, 0, 1, 'input')

        self.assertEqual(G.nodes[3]['input'], 1)
        self.assertEqual(G.edges[3, 0]['input'], 1)
        self.assertEqual(G.edges[3, 1]['input'], 1)


class TestAddSiblingship(unittest.TestCase):
    def test_siblingship_node_and_role_edges_labelled(self):
        num_people = 3
        G = data.generate_graph(num_people)
        data.add_base_labels(G, 'input')

        self.assertEqual(G.nodes[9]['input'], 0)
        self.assertEqual(G.edges[9, 0]['input'], 0)
        self.assertEqual(G.edges[9, 1]['input'], 0)

        data.add_siblingship(G, 0, 1, 'input')

        self.assertEqual(G.nodes[9]['input'], 1)
        self.assertEqual(G.edges[9, 0]['input'], 1)
        self.assertEqual(G.edges[9, 1]['input'], 1)


if __name__ == "__main__":
    unittest.main()
