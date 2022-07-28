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

from typedb_ml.typedb.thing import Thing
from typedb_ml.networkx.concept_dict_to_networkx import concept_dict_to_networkx
from typedb_ml.networkx.graph_test_case import GraphTestCase


class TestConceptDictToTypeDBGraph(GraphTestCase):
    def test_single_entity_graph_is_as_expected(self):
        variable_graph = nx.MultiDiGraph()
        variable_graph.add_node('x')

        person = Thing('V123', 'person', 'entity')
        concept_dict = {'x': person}

        typedb_graph = concept_dict_to_networkx(concept_dict, variable_graph)
        expected_typedb_graph = nx.MultiDiGraph()
        expected_typedb_graph.add_node(person, type='person')

        self.assertGraphsEqual(expected_typedb_graph, typedb_graph)

    def test_single_attribute_graph_is_as_expected(self):
        variable_graph = nx.MultiDiGraph()
        variable_graph.add_node('x')

        name = Thing('V123', 'name', 'attribute', value_type='string', value='Bob')
        concept_dict = {'x': name}

        typedb_graph = concept_dict_to_networkx(concept_dict, variable_graph)
        expected_typedb_graph = nx.MultiDiGraph()
        expected_typedb_graph.add_node(name, type='name', value_type='string', value='Bob')

        self.assertGraphsEqual(expected_typedb_graph, typedb_graph)

    def test_single_entity_single_relation_graph_is_as_expected(self):
        variable_graph = nx.MultiDiGraph()
        variable_graph.add_node('x')
        variable_graph.add_node('y')
        variable_graph.add_edge('y', 'x', type='employee')

        person = Thing('V123', 'person', 'entity')
        employment = Thing('V456', 'employment', 'relation')
        concept_dict = {'x': person, 'y': employment}

        typedb_graph = concept_dict_to_networkx(concept_dict, variable_graph)
        expected_typedb_graph = nx.MultiDiGraph()
        expected_typedb_graph.add_node(person, type='person')
        expected_typedb_graph.add_node(employment, type='employment')
        expected_typedb_graph.add_edge(employment, person, type='employee')

        self.assertGraphsEqual(expected_typedb_graph, typedb_graph)

    def test_two_entity_single_relation_graph_is_as_expected(self):
        variable_graph = nx.MultiDiGraph()
        variable_graph.add_node('x')
        variable_graph.add_node('y')
        variable_graph.add_node('r')
        variable_graph.add_edge('r', 'x', type='employee')
        variable_graph.add_edge('r', 'y', type='employer')

        person = Thing('V123', 'person', 'entity')
        company = Thing('V1234', 'company', 'entity')
        employment = Thing('V12345', 'employment', 'relation')
        concept_dict = {'x': person, 'y': company, 'r': employment}

        typedb_graph = concept_dict_to_networkx(concept_dict, variable_graph)

        expected_typedb_graph = nx.MultiDiGraph()
        expected_typedb_graph.add_node(person, type='person')
        expected_typedb_graph.add_node(company, type='company')
        expected_typedb_graph.add_node(employment, type='employment')
        expected_typedb_graph.add_edge(employment, person, type='employee')
        expected_typedb_graph.add_edge(employment, company, type='employer')

        self.assertGraphsEqual(expected_typedb_graph, typedb_graph)

    def test_same_thing_occurs_in_two_different_variables(self):
        variable_graph = nx.MultiDiGraph()
        variable_graph.add_node('x')
        variable_graph.add_node('y')

        person = Thing('V123', 'person', 'entity')
        person2 = Thing('V123', 'person', 'entity')
        concept_dict = {'x': person,
                        'y': person2}

        typedb_graph = concept_dict_to_networkx(concept_dict, variable_graph)
        expected_typedb_graph = nx.MultiDiGraph()
        expected_typedb_graph.add_node(person, type='person')

        self.assertGraphsEqual(expected_typedb_graph, typedb_graph)

    def test_edge_starting_from_entity_throws_exception(self):
        variable_graph = nx.MultiDiGraph()
        variable_graph.add_node('x')
        variable_graph.add_node('y')
        variable_graph.add_edge('x', 'y', type='employee')

        person = Thing('V123', 'person', 'entity')
        employment = Thing('V456', 'employment', 'relation')
        concept_dict = {'x': person, 'y': employment}

        with self.assertRaises(ValueError) as context:
            _ = concept_dict_to_networkx(concept_dict, variable_graph)

        self.assertEqual('An edge in the variable_graph originates from a non-relation, check the variable_graph!',
                         str(context.exception))

    def test_edge_starting_from_attribute_throws_exception(self):
        variable_graph = nx.MultiDiGraph()
        variable_graph.add_node('x')
        variable_graph.add_node('y')
        variable_graph.add_edge('x', 'y', type='employee')

        name = Thing('V123', 'name', 'attribute', value_type='string', value='Bob')
        employment = Thing('V456', 'employment', 'relation')
        concept_dict = {'x': name, 'y': employment}

        with self.assertRaises(ValueError) as context:
            _ = concept_dict_to_networkx(concept_dict, variable_graph)

        self.assertEqual('An edge in the variable_graph originates from a non-relation, check the variable_graph!',
                         str(context.exception))

    def test_exception_if_sets_of_variables_differ(self):
        variable_graph = nx.MultiDiGraph()
        variable_graph.add_node('x')
        variable_graph.add_node('y')
        variable_graph.add_node('z')

        thing = Thing('V123', 'person', 'entity')
        concept_dict = {'x': thing,
                        'y': thing,
                        'a': thing}

        with self.assertRaises(ValueError) as context:
            _ = concept_dict_to_networkx(concept_dict, variable_graph)

        self.assertEqual(
            'The variables in the variable_graph must match those in the concept_dict\n'
            'In the variable graph but not in the concept dict: {\'z\'}\n'
            'In the concept dict but not in the variable graph: {\'a\'}',
            str(context.exception))

    def test_variable_graph_properties_are_transferred_to_graph(self):
        variable_graph = nx.MultiDiGraph()
        variable_graph.add_node('x', input=1)
        variable_graph.add_node('y', input=1)
        variable_graph.add_edge('y', 'x', type='employee', input=0)

        person = Thing('V123', 'person', 'entity')
        employment = Thing('V456', 'employment', 'relation')
        concept_dict = {'x': person, 'y': employment}

        typedb_graph = concept_dict_to_networkx(concept_dict, variable_graph)
        expected_typedb_graph = nx.MultiDiGraph()
        expected_typedb_graph.add_node(person, type='person', input=1)
        expected_typedb_graph.add_node(employment, type='employment', input=1)
        expected_typedb_graph.add_edge(employment, person, type='employee', input=0)

        self.assertGraphsEqual(expected_typedb_graph, typedb_graph)


if __name__ == "__main__":
    unittest.main()
