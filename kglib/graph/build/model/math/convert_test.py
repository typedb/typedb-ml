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

import kglib.kgcn.core.ingest.traverse.data.context.neighbour as neighbour
from kglib.graph.build.from_queries_test import match_edge_types, match_node_things
from kglib.graph.build.model.math.convert import concept_dict_to_grakn_math_graph


class TestConceptDictToGraknMathGraph(unittest.TestCase):
    def test_single_entity_graph_is_as_expected(self):
        variable_graph = nx.MultiDiGraph()
        variable_graph.add_node('x')

        person = neighbour.Thing('V123', 'person', 'entity')
        concept_dict = {'x': person}

        grakn_graph = concept_dict_to_grakn_math_graph(concept_dict, variable_graph)
        expected_grakn_graph = nx.MultiDiGraph()
        expected_grakn_graph.add_node(person)

        # Requires all of these checks to ensure graphs compare as expected
        self.assertEqual(expected_grakn_graph.nodes, grakn_graph.nodes)
        self.assertEqual(expected_grakn_graph.edges, grakn_graph.edges)
        self.assertTrue(nx.is_isomorphic(expected_grakn_graph, grakn_graph,
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

        grakn_graph = concept_dict_to_grakn_math_graph(concept_dict, variable_graph)

        employee = neighbour.GraknEdge(employment, person, 'employee')

        expected_grakn_graph = nx.MultiDiGraph()
        expected_grakn_graph.add_node(person)
        expected_grakn_graph.add_node(employee)
        expected_grakn_graph.add_node(employment)
        expected_grakn_graph.add_edge(employment, employee, type='relates')
        expected_grakn_graph.add_edge(person, employee, type='plays')

        # Requires all of these checks to ensure graphs compare as expected
        self.assertTrue(nx.is_isomorphic(expected_grakn_graph, grakn_graph,
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

        grakn_graph = concept_dict_to_grakn_math_graph(concept_dict, variable_graph)

        employee = neighbour.GraknEdge(employment, person, 'employee')
        employer = neighbour.GraknEdge(employment, company, 'employer')

        expected_grakn_graph = nx.MultiDiGraph()
        expected_grakn_graph.add_node(person)
        expected_grakn_graph.add_node(company)
        expected_grakn_graph.add_node(employment)

        expected_grakn_graph.add_node(employee)
        expected_grakn_graph.add_edge(employment, employee, type='relates')
        expected_grakn_graph.add_edge(person, employee, type='plays')

        expected_grakn_graph.add_node(employer)
        expected_grakn_graph.add_edge(employment, employer, type='relates')
        expected_grakn_graph.add_edge(company, employer, type='plays')

        self.assertTrue(nx.is_isomorphic(expected_grakn_graph, grakn_graph,
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

        grakn_graph = concept_dict_to_grakn_math_graph(concept_dict, variable_graph)
        expected_grakn_graph = nx.MultiDiGraph()
        expected_grakn_graph.add_node(person)

        self.assertTrue(nx.is_isomorphic(expected_grakn_graph, grakn_graph,
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
            _ = concept_dict_to_grakn_math_graph(concept_dict, variable_graph)

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
            _ = concept_dict_to_grakn_math_graph(concept_dict, variable_graph)

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
            _ = concept_dict_to_grakn_math_graph(concept_dict, variable_graph)

        self.assertEqual('The variables in the variable_graph must match those in the concept_dict',
                         str(context.exception))


if __name__ == "__main__":
    unittest.main()
