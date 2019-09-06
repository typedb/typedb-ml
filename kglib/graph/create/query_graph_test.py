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

from kglib.graph.create.query_graph import QueryGraph


class TestQueryGraph(unittest.TestCase):

    def test_add_vars_adds_variable_nodes_as_expected(self):
        g = QueryGraph()
        g.add_vars('a', 'b')
        nodes = {node for node in g.nodes}
        self.assertSetEqual({'a', 'b'}, nodes)

    def test_add_has_edge_adds_edge_as_expected(self):
        g = QueryGraph()
        g.add_vars('a', 'b')
        g.add_has_edge('a', 'b')
        edges = [edge for edge in g.edges]
        self.assertEqual(1, len(edges))
        self.assertEqual('has', g.edges['a', 'b', 0]['type'])

    def test_add_role_edge_adds_role_as_expected(self):
        g = QueryGraph()
        g.add_vars('a', 'b')
        g.add_role_edge('a', 'b', 'role')
        edges = [edge for edge in g.edges]
        self.assertEqual(1, len(edges))
        self.assertEqual('role', g.edges['a', 'b', 0]['type'])
