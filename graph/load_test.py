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
import unittest.mock as mock

import grakn.client as client
import networkx as nx
import networkx.algorithms.isomorphism as iso

import graph.load as load
import kglib.kgcn.core.ingest.traverse.data.context.neighbour as neighbour
from graph.mock.concept import MockType, MockThing


class TestConceptDictsFromQuery(unittest.TestCase):
    def test_concept_dicts_are_built_as_expected(self):
        query = 'match $x id V123; get;'

        mock_tx = mock.MagicMock(client.Transaction)

        answer_map = [{'x': MockThing('V123', 'ENTITY', MockType('V456', 'ENTITY', 'person'))}]

        mock_tx.query.return_value = answer_map
        concept_dicts = load.concept_dicts_from_query(query, mock_tx)
        mock_tx.query.assert_called_with(query)

        expected_concept_dicts = [{'x': neighbour.Thing('V123', 'person', 'entity')}]

        self.assertEqual(expected_concept_dicts, concept_dicts)

    def test_concept_dicts_are_built_as_expected_with_2_concepts(self):
        query = 'match $x id V123; get;'

        mock_tx = mock.MagicMock(client.Transaction)

        answer_map = [{
            'x': MockThing('V123', 'ENTITY', MockType('V456', 'ENTITY', 'person')),
            'y': MockThing('V789', 'RELATION', MockType('V765', 'RELATION', 'employment')),
        }]

        mock_tx.query.return_value = answer_map
        concept_dicts = load.concept_dicts_from_query(query, mock_tx)
        mock_tx.query.assert_called_with(query)

        expected_concept_dicts = [{
            'x': neighbour.Thing('V123', 'person', 'entity'),
            'y': neighbour.Thing('V789', 'employment', 'relation'),
        }]

        self.assertEqual(expected_concept_dicts, concept_dicts)


def match_node_things(data1, data2):
    return data1 == data2


def match_edge_types(data1, data2):
    return data1 == data2


class TestCreateThingGraph(unittest.TestCase):
    def test_single_entity_graph_is_as_expected(self):
        variable_graph = nx.DiGraph()
        variable_graph.add_node('x')

        thing = neighbour.Thing('V123', 'person', 'entity')
        concept_dict = {'x': thing}

        thing_graph = load.create_thing_graph(concept_dict, variable_graph)
        expected_thing_graph = nx.DiGraph()
        expected_thing_graph.add_node(0, thing=thing)

        # Requires all of these checks to ensure graphs compare as expected
        self.assertEqual(expected_thing_graph.nodes, thing_graph.nodes)
        self.assertEqual(expected_thing_graph.edges, thing_graph.edges)
        self.assertTrue(nx.is_isomorphic(expected_thing_graph, thing_graph,
                                         node_match=match_node_things,
                                         edge_match=match_edge_types))

    def test_single_entity_single_relation_graph_is_as_expected(self):
        variable_graph = nx.DiGraph()
        variable_graph.add_node('x')
        variable_graph.add_node('y')
        variable_graph.add_edge('y', 'x', type='employee')

        thing1 = neighbour.Thing('V123', 'person', 'entity')
        thing2 = neighbour.Thing('V123', 'employment', 'relation')
        concept_dict = {'x': thing1, 'y': thing2}

        thing_graph = load.create_thing_graph(concept_dict, variable_graph)
        expected_thing_graph = nx.DiGraph()
        expected_thing_graph.add_node(0, thing=thing1)
        expected_thing_graph.add_node(1, thing=thing2)
        expected_thing_graph.add_edge(1, 0, type='employee')

        # Requires all of these checks to ensure graphs compare as expected
        self.assertEqual(expected_thing_graph.nodes, thing_graph.nodes)
        self.assertEqual(expected_thing_graph.edges, thing_graph.edges)
        self.assertTrue(nx.is_isomorphic(expected_thing_graph, thing_graph,
                                         node_match=match_node_things,
                                         edge_match=match_edge_types))

    def test_two_entity_single_relation_graph_is_as_expected(self):
        variable_graph = nx.DiGraph()
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

        expected_thing_graph = nx.DiGraph()
        expected_thing_graph.add_node(2, thing=person)
        expected_thing_graph.add_node(1, thing=company)
        expected_thing_graph.add_node(0, thing=employment)
        expected_thing_graph.add_edge(0, 2, type='employee')
        expected_thing_graph.add_edge(0, 1, type='employer')

        # Requires all of these checks to ensure graphs compare as expected
        self.assertTrue(nx.is_isomorphic(expected_thing_graph, thing_graph,
                                         node_match=match_node_things,
                                         edge_match=match_edge_types))

    def test_edge_starting_from_entity_throws_exception(self):
        variable_graph = nx.DiGraph()
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
        variable_graph = nx.DiGraph()
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
        variable_graph = nx.DiGraph()
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
