#
#  Copyright (C) 2021 Vaticle
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

import sys
import unittest

import networkx as nx
from typedb.client import *

from kglib.utils.grakn.object.thing import Thing, build_thing
from kglib.utils.grakn.test.base import GraknServer
from kglib.utils.grakn.test.mock.answer import MockConceptMap
from kglib.utils.grakn.test.mock.concept import MockType, MockAttributeType, MockThing, MockAttribute
from kglib.utils.graph.thing.queries_to_networkx_graph import build_graph_from_queries
from kglib.utils.graph.test.case import GraphTestCase

TEST_URI = "localhost:1729"


def mock_sampler(input_iter):
    return input_iter


class MockTransaction:
    def query(self, query, infer=True):

        if query == 'match $x id V123;':
            return [MockConceptMap({'x': MockThing('V123', MockType('V4123', 'person', 'ENTITY'))})]
        elif query == 'match $x id V123, has name $n;':
            return [
                MockConceptMap({
                    'x': MockThing('V123', MockType('V4123', 'person', 'ENTITY')),
                    'n': MockAttribute('V987', 'Bob', MockAttributeType('V555', 'name', 'ATTRIBUTE', 'STRING'))
                })]
        elif query == 'match $x id V123; $r(child: $x, parent: $y);':
            return [
                MockConceptMap({
                    'x': MockThing('V123', MockType('V4123', 'person', 'ENTITY')),
                    'y': MockThing('V123', MockType('V4123', 'person', 'ENTITY')),
                    'r': MockThing('V567', MockType('V9876', 'parentship', 'RELATION'))
                })]
        else:
            raise NotImplementedError


class ITBuildGraphFromQueries(GraphTestCase):
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

        query_sampler_variable_graph_tuples = [('match $x id V123;', mock_sampler, g1),
                                               ('match $x id V123, has name $n;', mock_sampler, g2),
                                               ('match $x id V123; $r(child: $x, parent: $y);', mock_sampler, g3),
                                               # TODO Add functionality for loading schema at a later date
                                               # ('match $x sub person; $x sub $type;', g4),
                                               # ('match $x sub $y;', g5),
                                               ]

        mock_tx = MockTransaction()

        combined_graph = build_graph_from_queries(query_sampler_variable_graph_tuples, mock_tx)

        person_exp = Thing('V123', 'person', 'entity')
        parentship_exp = Thing('V567', 'parentship', 'relation')
        name_exp = Thing('V987', 'name', 'attribute', value_type='string', value='Bob')
        expected_combined_graph = nx.MultiDiGraph()
        expected_combined_graph.add_node(person_exp, type='person')
        expected_combined_graph.add_node(name_exp, type='name', value_type='string', value='Bob')
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
        query_sampler_variable_graph_tuples = [('match $x id V123;', mock_sampler, g1),
                                               ('match $y id V123;', mock_sampler, g2)]

        class MockTransaction2:
            def query(self, query, infer=True):
                if query == 'match $x id V123;':
                    return [MockConceptMap({'x': MockThing('V123', MockType('V4123', 'person', 'ENTITY'))})]
                elif query == 'match $y id V123;':
                    return []

        mock_tx = MockTransaction2()

        with self.assertWarns(UserWarning) as context:
            build_graph_from_queries(query_sampler_variable_graph_tuples, mock_tx)

        self.assertEqual(f'There were no results for query: \n\"match $y id V123;\"\nand so nothing will be added to the graph for this query', str(context.warning))

    def test_exception_is_raised_when_there_are_no_results_for_any_query(self):
        g1 = nx.MultiDiGraph()
        g1.add_node('x')
        query_sampler_variable_graph_tuples = [('match $x id V123;', mock_sampler, g1)]

        class MockTransactionEmpty:
            def query(self, query, infer=True):
                return []

        mock_tx = MockTransactionEmpty()

        with self.assertRaises(RuntimeError) as context:
            build_graph_from_queries(query_sampler_variable_graph_tuples, mock_tx)

        self.assertEqual(f'The graph from queries: {[query_sampler_variable_graph_tuple[0] for query_sampler_variable_graph_tuple in query_sampler_variable_graph_tuples]}\n'
                         f'could not be created, since none of these queries returned results', str(context.exception))


class ITBuildGraphFromQueriesWithRealGrakn(GraphTestCase):

    DATABASE = "it_build_graph_from_queries"
    SCHEMA = ("define "
              "person sub entity, has name, plays parent, plays child;"
              "name sub attribute, value string;"
              "parentship sub relation, relates parent, relates child;")
    DATA = ('insert '
            '$p isa person, has name "Bob";'
            '$r(parent: $p, child: $p) isa parentship;')

    def setUp(self):
        self._database = type(self).__name__.lower()  # Use the name of this test class as the database name
        print(self._database)
        self._client = TypeDB.core_client(address="localhost:1729")

    def tearDown(self):
        self._client.databases().get(self._database).delete()
        self._client.close()

    def test_graph_is_built_from_grakn_as_expected(self):

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

        query_sampler_variable_graph_tuples = [('match $x isa person;', mock_sampler, g1),
                                               ('match $x isa person, has name $n;', mock_sampler, g2),
                                               ('match $x isa person; $r(child: $x, parent: $y);', mock_sampler, g3),
                                               # TODO Add functionality for loading schema at a later date
                                               # ('match $x sub person; $x sub $type;', g4),
                                               # ('match $x sub $y;', g5),
                                               ]

        with self._client.session(self._database, SessionType.DATA) as session:

            with session.transaction(TransactionType.WRITE) as tx:
                tx.query(ITBuildGraphFromQueriesWithRealGrakn.SCHEMA)
                tx.query(ITBuildGraphFromQueriesWithRealGrakn.DATA)
                tx.commit()

            with session.transaction(TransactionType.READ) as tx:
                combined_graph = build_graph_from_queries(query_sampler_variable_graph_tuples, tx)

                person_exp = build_thing(next(tx.query().match('match $x isa person;')).get('x'))
                name_exp = build_thing(next(tx.query().match('match $x isa name;')).get('x'))
                parentship_exp = build_thing(next(tx.query().match('match $x isa parentship;')).get('x'))

        expected_combined_graph = nx.MultiDiGraph()
        expected_combined_graph.add_node(person_exp, type='person')
        expected_combined_graph.add_node(name_exp, type='name', value_type='string', value='Bob')
        expected_combined_graph.add_node(parentship_exp, type='parentship')
        expected_combined_graph.add_edge(parentship_exp, person_exp, type='child')
        expected_combined_graph.add_edge(parentship_exp, person_exp, type='parent')
        expected_combined_graph.add_edge(person_exp, name_exp, type='has')

        self.assertGraphsEqual(expected_combined_graph, combined_graph)


if __name__ == "__main__":

    with GraknServer(sys.argv.pop()) as gs:
        unittest.main()
