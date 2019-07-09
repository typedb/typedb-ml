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

import graph.build.from_queries_test
import graph.build.model.math.convert as math_convert
import kglib.kgcn.core.ingest.traverse.data.context.neighbour as neighbour
from graph.build.from_queries import build_graph_from_queries
from graph.mock.concept import MockType, MockAttributeType, MockThing, MockAttribute


def mock_sampler(input_iter):
    return input_iter


class MockTransaction:
    def query(self, query):

        if query == 'match $x id V123; get;':
            return [{'x': MockThing('V123', MockType('V4123', 'person', 'ENTITY'))}]
        elif query == 'match $x id V123, has name $n; get;':
            return [
                {
                    'x': MockThing('V123', MockType('V4123', 'person', 'ENTITY')),
                    'n': MockAttribute('V987', 'Bob', MockAttributeType('V555', 'name', 'ATTRIBUTE', 'STRING'))
                }]
        elif query == 'match $x id V123; $r(child: $x, parent: $y); get;':
            return [
                {
                    'x': MockThing('V123', MockType('V4123', 'person', 'ENTITY')),
                    'y': MockThing('V123', MockType('V4123', 'person', 'ENTITY')),
                    'r': MockThing('V567', MockType('V9876', 'parentship', 'RELATION'))
                }]
        else:
            raise NotImplementedError


class ITBuildGraphFromQueries(unittest.TestCase):
    def test_standard_graph_is_built_as_expected(self):
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

        query_sampler_variable_graph_tuples = [('match $x id V123; get;', mock_sampler, g1),
                                               ('match $x id V123, has name $n; get;', mock_sampler, g2),
                                               ('match $x id V123; $r(child: $x, parent: $y); get;', mock_sampler, g3),
                                               # TODO Add functionality for loading schema at a later date
                                               # ('match $x sub person; $x sub $type; get;', g4),
                                               # ('match $x sub $y; get;', g5),
                                               ]

        mock_tx = MockTransaction()

        combined_graph = build_graph_from_queries(query_sampler_variable_graph_tuples, mock_tx)

        person_exp = neighbour.Thing('V123', 'person', 'entity')
        employment_exp = neighbour.Thing('V567', 'parentship', 'relation')
        name_exp = neighbour.Thing('V987', 'name', 'attribute', data_type='string', value='Bob')
        expected_combined_graph = nx.MultiDiGraph()
        expected_combined_graph.add_node(person_exp)
        expected_combined_graph.add_node(name_exp)
        expected_combined_graph.add_node(employment_exp)
        expected_combined_graph.add_edge(employment_exp, person_exp, type='child')
        expected_combined_graph.add_edge(employment_exp, person_exp, type='parent')
        expected_combined_graph.add_edge(person_exp, name_exp, type='has')

        self.assertTrue(nx.is_isomorphic(expected_combined_graph, combined_graph,
                                         node_match=graph.build.from_queries_test.match_node_things,
                                         edge_match=graph.build.from_queries_test.match_edge_types))

    def test_math_graph_is_built_as_expected(self):
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

        query_sampler_variable_graph_tuples = [('match $x id V123; get;', mock_sampler, g1),
                                               ('match $x id V123, has name $n; get;', mock_sampler, g2),
                                               ('match $x id V123; $r(child: $x, parent: $y); get;', mock_sampler, g3),
                                               # TODO Add functionality for loading schema at a later date
                                               # ('match $x sub person; $x sub $type; get;', g4),
                                               # ('match $x sub $y; get;', g5),
                                               ]

        mock_tx = MockTransaction()

        combined_graph = build_graph_from_queries(query_sampler_variable_graph_tuples, mock_tx,
                                                  concept_dict_converter=math_convert.concept_dict_to_grakn_math_graph)

        person_exp = neighbour.Thing('V123', 'person', 'entity')
        parentship_exp = neighbour.Thing('V567', 'parentship', 'relation')
        name_exp = neighbour.Thing('V987', 'name', 'attribute', data_type='string', value='Bob')
        parent = neighbour.GraknEdge(parentship_exp, person_exp, 'parent')
        child = neighbour.GraknEdge(parentship_exp, person_exp, 'child')
        expected_combined_graph = nx.MultiDiGraph()
        expected_combined_graph.add_node(person_exp)
        expected_combined_graph.add_node(name_exp)
        expected_combined_graph.add_node(parentship_exp)

        expected_combined_graph.add_node(parent)
        expected_combined_graph.add_node(child)

        expected_combined_graph.add_edge(parentship_exp, child, type='relates')
        expected_combined_graph.add_edge(parentship_exp, parent, type='relates')
        expected_combined_graph.add_edge(person_exp, parent, type='plays')
        expected_combined_graph.add_edge(person_exp, child, type='plays')
        expected_combined_graph.add_edge(person_exp, name_exp, type='has')

        self.assertTrue(nx.is_isomorphic(expected_combined_graph, combined_graph,
                                         node_match=graph.build.from_queries_test.match_node_things,
                                         edge_match=graph.build.from_queries_test.match_edge_types))


if __name__ == "__main__":
    unittest.main()
