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
from typedb.api.concept.type.attribute_type import AttributeType

from typedb_ml.networkx.graph_test_case import GraphTestCase
from typedb_ml.networkx.queries_to_networkx import concept_dict_from_concept_map, \
    combine_n_graphs, build_graph_from_queries
from typedb_ml.networkx.query_graph import Query
from typedb_ml.typedb.test.mock.answer import MockConceptMap
from typedb_ml.typedb.test.mock.concept import MockType, MockAttributeType, MockThing, MockAttribute
from typedb_ml.typedb.thing import Thing


class TestBuildGraphFromQueries(GraphTestCase):
    def test_graph_is_built_as_expected(self):
        g1 = nx.MultiDiGraph()
        g1.add_node('x')

        g2 = nx.MultiDiGraph()
        g2.add_node('x')
        g2.add_node('n')
        g2.add_edge('x', 'n', type='has')

        g3 = nx.MultiDiGraph()
        g3.add_node('x')
        g3.add_node('r')
        g3.add_node('y')
        g3.add_edge('r', 'x', type='child')
        g3.add_edge('r', 'y', type='parent')

        queries = [
            Query(g1, 'match $x iid V123;'),
            Query(g2, 'match $x iid V123, has name $n;'),
            Query(g3, 'match $x iid V123; $r(child: $x, parent: $y);'),
        ]

        class MockTransaction:

            def query(self):
                return MockQueryManager()

        class MockQueryManager:
            def match(self, query):

                if query == 'match $x iid V123;':
                    return [MockConceptMap({'x': MockThing('V123', MockType('V4123', 'person', 'ENTITY'))})]
                elif query == 'match $x iid V123, has name $n;':
                    return [
                        MockConceptMap({
                            'x': MockThing('V123', MockType('V4123', 'person', 'ENTITY')),
                            'n': MockAttribute('V987', 'Bob', MockAttributeType('V555', 'name', 'ATTRIBUTE',
                                                                                AttributeType.ValueType.STRING))
                        })]
                elif query == 'match $x iid V123; $r(child: $x, parent: $y);':
                    return [
                        MockConceptMap({
                            'x': MockThing('V123', MockType('V4123', 'person', 'ENTITY')),
                            'y': MockThing('V123', MockType('V4123', 'person', 'ENTITY')),
                            'r': MockThing('V567', MockType('V9876', 'parentship', 'RELATION'))
                        })]
                else:
                    raise NotImplementedError

        mock_tx = MockTransaction()

        combined_graph = build_graph_from_queries(queries, mock_tx)

        person_exp = Thing('V123', 'person', 'entity')
        parentship_exp = Thing('V567', 'parentship', 'relation')
        name_exp = Thing('V987', 'name', 'attribute', value_type=AttributeType.ValueType.STRING, value='Bob')
        expected_combined_graph = nx.MultiDiGraph()
        expected_combined_graph.add_node(person_exp, type='person')
        expected_combined_graph.add_node(name_exp, type='name', value_type=AttributeType.ValueType.STRING, value='Bob')
        expected_combined_graph.add_node(parentship_exp, type='parentship')
        expected_combined_graph.add_edge(parentship_exp, person_exp, type='child')
        expected_combined_graph.add_edge(parentship_exp, person_exp, type='parent')
        expected_combined_graph.add_edge(person_exp, name_exp, type='has')

        self.assertGraphsEqual(expected_combined_graph, combined_graph)

    def test_warning_given_when_one_query_gives_no_results(self):
        g1 = nx.MultiDiGraph()
        g1.add_node('x')
        g2 = nx.MultiDiGraph()
        g2.add_node('y')
        queries = [Query(g1, 'match $x iid V123;'),
                   Query(g2, 'match $y iid V123;')]

        class MockTransaction:
            def query(self):
                return MockQueryManager()

        class MockQueryManager:
            def match(self, query):
                if query == 'match $x iid V123;':
                    return [MockConceptMap({'x': MockThing('V123', MockType('V4123', 'person', 'ENTITY'))})]
                elif query == 'match $y iid V123;':
                    return []

        mock_tx = MockTransaction()

        with self.assertWarns(UserWarning) as context:
            build_graph_from_queries(queries, mock_tx)

        self.assertEqual(f'There were no results for query: \n\"match $y iid V123;\"\nand so nothing will be added to the graph for this query', str(context.warning))

    def test_exception_is_raised_when_there_are_no_results_for_any_query(self):
        g1 = nx.MultiDiGraph()
        g1.add_node('x')
        queries = [Query(g1, 'match $x iid V123;')]

        class QueryManagerEmpty:
            def match(self, query):
                return []

        class MockTransactionEmpty:
            def query(self):
                return QueryManagerEmpty()

        mock_tx = MockTransactionEmpty()

        with self.assertRaises(RuntimeError) as context:
            build_graph_from_queries(queries, mock_tx)

        self.assertEqual(f'The graph from queries: {[query.string for query in queries]}\n'
                         f'could not be created, since none of these queries returned results', str(context.exception))


class TestConceptDictsFromQuery(unittest.TestCase):
    def test_concept_dicts_are_built_as_expected(self):
        concept_map = MockConceptMap({'x': MockThing('V123', MockType('V456', 'person', 'ENTITY'))})
        concept_dicts = concept_dict_from_concept_map(concept_map)

        expected_concept_dicts = {'x': Thing('V123', 'person', 'entity')}

        self.assertEqual(expected_concept_dicts, concept_dicts)

    def test_concept_dicts_are_built_as_expected_with_2_concepts(self):
        concept_map = MockConceptMap({
            'x': MockThing('V123', MockType('V456', 'person', 'ENTITY')),
            'y': MockThing('V789', MockType('V765', 'employment', 'RELATION')),
        })

        concept_dicts = concept_dict_from_concept_map(concept_map)

        expected_concept_dict = {
            'x': Thing('V123', 'person', 'entity'),
            'y': Thing('V789', 'employment', 'relation'),
        }

        self.assertEqual(expected_concept_dict, concept_dicts)


class TestCombineGraphs(GraphTestCase):

    def test_graph_combined_as_expected(self):

        person = Thing('V123', 'person', 'entity')
        employment = Thing('V567', 'employment', 'relation')
        typedb_graph_a = nx.MultiDiGraph()
        typedb_graph_a.add_node(person)
        typedb_graph_a.add_node(employment)
        typedb_graph_a.add_edge(employment, person, type='employee')

        person_b = Thing('V123', 'person', 'entity')
        name = Thing('V1234', 'name', 'attribute', value_type='string', value='Bob')
        typedb_graph_b = nx.MultiDiGraph()
        typedb_graph_b.add_node(person_b)
        typedb_graph_b.add_node(name)
        typedb_graph_b.add_edge(person_b, name, type='has')

        combined_graph = combine_n_graphs([typedb_graph_a, typedb_graph_b])

        person_ex = Thing('V123', 'person', 'entity')
        employment_ex = Thing('V567', 'employment', 'relation')
        name_ex = Thing('V1234', 'name', 'attribute', value_type='string', value='Bob')
        expected_combined_graph = nx.MultiDiGraph()
        expected_combined_graph.add_node(person_ex)
        expected_combined_graph.add_node(name_ex)
        expected_combined_graph.add_node(employment_ex)
        expected_combined_graph.add_edge(employment_ex, person_ex, type='employee')
        expected_combined_graph.add_edge(person_ex, name_ex, type='has')

        self.assertGraphsEqual(expected_combined_graph, combined_graph)

    def test_when_graph_node_properties_are_mismatched_exception_is_raised(self):
        person_a = Thing('V123', 'person', 'entity')
        name_a = Thing('V1234', 'name', 'attribute', value_type='string', value='Bob')
        typedb_graph_a = nx.MultiDiGraph(name='a')
        typedb_graph_a.add_node(person_a, input=1)
        typedb_graph_a.add_node(name_a, input=1)
        typedb_graph_a.add_edge(person_a, name_a, type='has', input=0)

        person_b = Thing('V123', 'person', 'entity')
        name_b = Thing('V1234', 'name', 'attribute', value_type='string', value='Bob')
        typedb_graph_b = nx.MultiDiGraph(name='b')
        typedb_graph_b.add_node(person_b, input=1)
        typedb_graph_b.add_node(name_b, input=0)
        typedb_graph_b.add_edge(person_b, name_b, type='has', input=0)

        with self.assertRaises(ValueError) as context:
            combine_n_graphs([typedb_graph_a, typedb_graph_b])

        self.assertEqual(('Found non-matching node properties for node <name, V1234: Bob> '
                          'between graphs a and b:\n'
                          'In graph a: {\'input\': 1}\n'
                          'In graph b: {\'input\': 0}'), str(context.exception))

    def test_when_graph_edge_properties_are_mismatched_exception_is_raised(self):
        person_a = Thing('V123', 'person', 'entity')
        name_a = Thing('V1234', 'name', 'attribute', value_type='string', value='Bob')
        typedb_graph_a = nx.MultiDiGraph(name='a')
        typedb_graph_a.add_node(person_a, input=1)
        typedb_graph_a.add_node(name_a, input=1)
        typedb_graph_a.add_edge(person_a, name_a, type='has', input=0)

        person_b = Thing('V123', 'person', 'entity')
        name_b = Thing('V1234', 'name', 'attribute', value_type='string', value='Bob')
        typedb_graph_b = nx.MultiDiGraph(name='b')
        typedb_graph_b.add_node(person_b, input=1)
        typedb_graph_b.add_node(name_b, input=1)
        typedb_graph_b.add_edge(person_b, name_b, type='has', input=1)

        with self.assertRaises(ValueError) as context:
            combine_n_graphs([typedb_graph_a, typedb_graph_b])

        self.assertEqual(('Found non-matching edge properties for edge (<person, V123>, <name, V1234: Bob>, 0) '
                          'between graphs a and b:\n'
                          'In graph a: {\'type\': \'has\', \'input\': 0}\n'
                          'In graph b: {\'type\': \'has\', \'input\': 1}'), str(context.exception))


if __name__ == "__main__":
    unittest.main()
