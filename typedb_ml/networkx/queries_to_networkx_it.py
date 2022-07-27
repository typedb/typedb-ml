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

import sys
import unittest

import networkx as nx
from typedb.client import *

from typedb_ml.networkx.graph_test_case import GraphTestCase
from typedb_ml.networkx.queries_to_networkx import build_graph_from_queries
from typedb_ml.networkx.query_graph import Query
from typedb_ml.typedb.test.base import TypeDBServer
from typedb_ml.typedb.thing import build_thing


class ITBuildGraphFromQueriesWithRealTypeDB(GraphTestCase):

    DATABASE = "it_build_graph_from_queries"
    SCHEMA = ("define "
              "person sub entity, owns name, plays parentship:parent, plays parentship:child;"
              "name sub attribute, value string;"
              "parentship sub relation, relates parent, relates child;")
    DATA = ('insert '
            '$p isa person, has name "Bob";'
            '$r(parent: $p, child: $p) isa parentship;')

    def setUp(self):
        self._database = type(self).__name__.lower()  # Use the name of this test class as the database name
        print(self._database)
        self._client = TypeDB.core_client(address="localhost:1729")
        if not self._client.databases().contains(self._database):
            self._client.databases().create(self._database)

    def tearDown(self):
        self._client.databases().get(self._database).delete()
        self._client.close()

    def test_graph_is_built_from_typedb_as_expected(self):

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
            Query(g1, 'match $x isa person;'),
            Query(g2, 'match $x isa person, has name $n;'),
            Query(g3, 'match $x isa person; $r(child: $x, parent: $y);'),
        ]

        with self._client.session(self._database, SessionType.SCHEMA) as session:

            with session.transaction(TransactionType.WRITE) as tx:
                tx.query().define(ITBuildGraphFromQueriesWithRealTypeDB.SCHEMA)
                tx.commit()

        with self._client.session(self._database, SessionType.DATA) as session:

            with session.transaction(TransactionType.WRITE) as tx:
                tx.query().insert(ITBuildGraphFromQueriesWithRealTypeDB.DATA)
                tx.commit()
            with session.transaction(TransactionType.READ) as tx:
                combined_graph = build_graph_from_queries(queries, tx)

                person_exp = build_thing(next(tx.query().match('match $x isa person;')).get('x'))
                name_exp = build_thing(next(tx.query().match('match $x isa name;')).get('x'))
                parentship_exp = build_thing(next(tx.query().match('match $x isa parentship;')).get('x'))

        expected_combined_graph = nx.MultiDiGraph()
        expected_combined_graph.add_node(person_exp, type='person')
        expected_combined_graph.add_node(name_exp, type='name', value_type=AttributeType.ValueType.STRING, value='Bob')
        expected_combined_graph.add_node(parentship_exp, type='parentship')
        expected_combined_graph.add_edge(parentship_exp, person_exp, type='child')
        expected_combined_graph.add_edge(parentship_exp, person_exp, type='parent')
        expected_combined_graph.add_edge(person_exp, name_exp, type='has')

        self.assertGraphsEqual(expected_combined_graph, combined_graph)


def mock_sampler(input_iter):
    return input_iter


if __name__ == "__main__":

    with TypeDBServer(sys.argv.pop()) as gs:
        unittest.main()
