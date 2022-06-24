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


def match_node_things(data1, data2):
    return data1 == data2


def match_edge_types(data1, data2):
    return data1 == data2


class GraphTestCase(unittest.TestCase):

    def assertNodesEqual(self, graph_1, graph_2):
        try:
            self.assertCountEqual(list(graph_1.nodes()), list(graph_2.nodes()))
        except AssertionError as e:
            raise AssertionError('Node counts do not match. ' + str(e))

    def assertEdgesEqual(self, graph_1, graph_2):
        try:
            self.assertCountEqual(list(graph_1.edges()), list(graph_2.edges()))
        except AssertionError as e:
            raise AssertionError('Edge counts do not match. ' + str(e))

    def assertIsIsomorphic(self, graph_1, graph_2):
        try:
            self.assertTrue(nx.is_isomorphic(graph_1, graph_2,
                                             node_match=match_node_things,
                                             edge_match=match_edge_types))
        except AssertionError:
            raise AssertionError(
                "The two graphs are not isomorphic based on the data attached to the nodes and/or edges")

    def assertGraphsEqual(self, graph_1, graph_2):
        self.assertNodesEqual(graph_1, graph_2)
        self.assertEdgesEqual(graph_1, graph_2)
        self.assertIsIsomorphic(graph_1, graph_2)
