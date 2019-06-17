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

import graph.load as load
import kglib.kgcn.core.ingest.traverse.data.context.neighbour as neighbour
from graph.mock.concept import MockType, MockThing


class TestConceptDictsFromQuery(unittest.TestCase):
    def test_concept_dicts_are_built_as_expected(self):
        concept_map = {'x': MockThing('V123', MockType('V456', 'person', 'ENTITY'))}
        concept_dicts = load.concept_dict_from_concept_map(concept_map)

        expected_concept_dicts = {'x': neighbour.Thing('V123', 'person', 'entity')}

        self.assertEqual(expected_concept_dicts, concept_dicts)

    def test_concept_dicts_are_built_as_expected_with_2_concepts(self):
        concept_map = {
            'x': MockThing('V123', MockType('V456', 'person', 'ENTITY')),
            'y': MockThing('V789', MockType('V765', 'employment', 'RELATION')),
        }

        concept_dicts = load.concept_dict_from_concept_map(concept_map)

        expected_concept_dict = {
            'x': neighbour.Thing('V123', 'person', 'entity'),
            'y': neighbour.Thing('V789', 'employment', 'relation'),
        }

        self.assertEqual(expected_concept_dict, concept_dicts)


def match_node_things(data1, data2):
    return data1 == data2


def match_edge_types(data1, data2):
    return data1 == data2


class TestCreateThingGraph(unittest.TestCase):
    def test_single_entity_graph_is_as_expected(self):
        variable_graph = nx.MultiDiGraph()
        variable_graph.add_node('x')

        person = neighbour.Thing('V123', 'person', 'entity')
        concept_dict = {'x': person}

        thing_graph = load.create_thing_graph(concept_dict, variable_graph)
        expected_thing_graph = nx.MultiDiGraph()
        expected_thing_graph.add_node(person)

        # Requires all of these checks to ensure graphs compare as expected
        self.assertEqual(expected_thing_graph.nodes, thing_graph.nodes)
        self.assertEqual(expected_thing_graph.edges, thing_graph.edges)
        self.assertTrue(nx.is_isomorphic(expected_thing_graph, thing_graph,
                                         node_match=match_node_things,
                                         edge_match=match_edge_types))

    def test_single_entity_single_relation_graph_is_as_expected(self):
        variable_graph = nx.MultiDiGraph()
        variable_graph.add_node('x')
        variable_graph.add_node('y')
        variable_graph.add_edge('y', 'x', type='employee')

        person = neighbour.Thing('V123', 'person', 'entity')
        employment = neighbour.Thing('V123', 'employment', 'relation')
        concept_dict = {'x': person, 'y': employment}

        thing_graph = load.create_thing_graph(concept_dict, variable_graph)
        expected_thing_graph = nx.MultiDiGraph()
        expected_thing_graph.add_node(person)
        expected_thing_graph.add_node(employment)
        expected_thing_graph.add_edge(employment, person, type='employee')

        # Requires all of these checks to ensure graphs compare as expected
        self.assertEqual(expected_thing_graph.nodes, thing_graph.nodes)
        self.assertEqual(expected_thing_graph.edges, thing_graph.edges)
        self.assertTrue(nx.is_isomorphic(expected_thing_graph, thing_graph,
                                         node_match=match_node_things,
                                         edge_match=match_edge_types))

    def test_two_entity_single_relation_graph_is_as_expected(self):
        variable_graph = nx.MultiDiGraph()
        variable_graph.add_node('x')
        variable_graph.add_node('y')
        variable_graph.add_node('r')
        variable_graph.add_edge('r', 'x', type='employee')
        variable_graph.add_edge('r', 'y', type='employer')

        person = neighbour.Thing('V123', 'person', 'entity')
        company = neighbour.Thing('V1234', 'person', 'entity')
        employment = neighbour.Thing('V12345', 'employment', 'relation')
        concept_dict = {'x': person, 'y': company, 'r': employment}

        thing_graph = load.create_thing_graph(concept_dict, variable_graph)

        expected_thing_graph = nx.MultiDiGraph()
        expected_thing_graph.add_node(person)
        expected_thing_graph.add_node(company)
        expected_thing_graph.add_node(employment)
        expected_thing_graph.add_edge(employment, person, type='employee')
        expected_thing_graph.add_edge(employment, company, type='employer')

        self.assertTrue(nx.is_isomorphic(expected_thing_graph, thing_graph,
                                         node_match=match_node_things,
                                         edge_match=match_edge_types))

    def test_same_thing_occurs_in_two_different_variables(self):
        variable_graph = nx.MultiDiGraph()
        variable_graph.add_node('x')
        variable_graph.add_node('y')

        person = neighbour.Thing('V123', 'person', 'entity')
        person2 = neighbour.Thing('V123', 'person', 'entity')
        concept_dict = {'x': person,
                        'y': person2}

        thing_graph = load.create_thing_graph(concept_dict, variable_graph)
        expected_thing_graph = nx.MultiDiGraph()
        expected_thing_graph.add_node(person)

        self.assertTrue(nx.is_isomorphic(expected_thing_graph, thing_graph,
                                         node_match=match_node_things,
                                         edge_match=match_edge_types))

    def test_edge_starting_from_entity_throws_exception(self):
        variable_graph = nx.MultiDiGraph()
        variable_graph.add_node('x')
        variable_graph.add_node('y')
        variable_graph.add_edge('x', 'y', type='employee')

        person = neighbour.Thing('V123', 'person', 'entity')
        employment = neighbour.Thing('V123', 'employment', 'relation')
        concept_dict = {'x': person, 'y': employment}

        with self.assertRaises(ValueError) as context:
            _ = load.create_thing_graph(concept_dict, variable_graph)

        self.assertEqual('An edge in the variable_graph originates from a non-relation, check the variable_graph!',
                         str(context.exception))

    def test_edge_starting_from_attribute_throws_exception(self):
        variable_graph = nx.MultiDiGraph()
        variable_graph.add_node('x')
        variable_graph.add_node('y')
        variable_graph.add_edge('x', 'y', type='employee')

        name = neighbour.Thing('V123', 'name', 'attribute', data_type='string', value='Bob')
        employment = neighbour.Thing('V123', 'employment', 'relation')
        concept_dict = {'x': name, 'y': employment}

        with self.assertRaises(ValueError) as context:
            _ = load.create_thing_graph(concept_dict, variable_graph)

        self.assertEqual('An edge in the variable_graph originates from a non-relation, check the variable_graph!',
                         str(context.exception))

    def test_exception_if_sets_of_variables_differ(self):
        variable_graph = nx.MultiDiGraph()
        variable_graph.add_node('x')
        variable_graph.add_node('y')
        variable_graph.add_node('z')

        thing = neighbour.Thing('V123', 'person', 'entity')
        concept_dict = {'x': thing,
                        'y': thing,
                        'a': thing}

        with self.assertRaises(ValueError) as context:
            _ = load.create_thing_graph(concept_dict, variable_graph)

        self.assertEqual('The variables in the variable_graph must match those in the concept_dict',
                         str(context.exception))


class TestCombineGraphs(unittest.TestCase):

    def test_graph_combined_as_expected(self):

        person = neighbour.Thing('V123', 'person', 'entity')
        employment = neighbour.Thing('V567', 'employment', 'relation')
        thing_graph_a = nx.MultiDiGraph()
        thing_graph_a.add_node(person)
        thing_graph_a.add_node(employment)
        thing_graph_a.add_edge(employment, person, type='employee')

        person_b = neighbour.Thing('V123', 'person', 'entity')
        name = neighbour.Thing('V1234', 'name', 'attribute', data_type='string', value='Bob')
        thing_graph_b = nx.MultiDiGraph()
        thing_graph_b.add_node(person_b)
        thing_graph_b.add_node(name)
        thing_graph_b.add_edge(person_b, name, type='has')

        combined_graph = load.combine_graphs(thing_graph_a, thing_graph_b)

        person_ex = neighbour.Thing('V123', 'person', 'entity')
        employment_ex = neighbour.Thing('V567', 'employment', 'relation')
        name_ex = neighbour.Thing('V1234', 'name', 'attribute', data_type='string', value='Bob')
        expected_combined_graph = nx.MultiDiGraph()
        expected_combined_graph.add_node(person_ex)
        expected_combined_graph.add_node(name_ex)
        expected_combined_graph.add_node(employment_ex)
        expected_combined_graph.add_edge(employment_ex, person_ex, type='employee')
        expected_combined_graph.add_edge(person_ex, name_ex, type='has')

        self.assertTrue(nx.is_isomorphic(expected_combined_graph, combined_graph,
                                         node_match=match_node_things,
                                         edge_match=match_edge_types))


if __name__ == "__main__":
    unittest.main()
