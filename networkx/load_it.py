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


class ITNetworkxFromQueryVariablegraphTuples(unittest.TestCase):
    def test_graph_is_built_as_expected(self):
        g1 = nx.DiGraph()
        g1.add_node('x')

        g2 = nx.DiGraph()
        g2.add_node('x')
        g2.add_node('n')
        g2.add_edge('x', 'n', type='has')

        g3 = nx.DiGraph()
        g3.add_node('x')
        g3.add_node('r')
        g3.add_node('y')
        g3.add_edge('r', 'x', type_var='role1')
        g3.add_edge('r', 'y', type='parent')

        query_variablegraph_tuples = [('match $x id V123; get;', g1),
                                   ('match $x id V123, has name $n; get;', g2),
                                   ('match $x id V123; $r($role1: $x, parent: $y); get;', g3),
                                   # TODO Add functionality for loading schema at a later date
                                   # ('match $x sub person; $x sub $type; get;', g4),
                                   # ('match $x sub $y; get;', g5),
                                   ]

        networkx_from_query_variablegraph_tuples(query_variablegraph_tuples, session)
