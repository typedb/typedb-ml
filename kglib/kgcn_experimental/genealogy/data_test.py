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
from kglib.kgcn.core.ingest.traverse.data.context.neighbour import build_thing
import networkx as nx

from kglib.kgcn_experimental.genealogy.data import create_concept_graphs
from kglib.utils.graph.test.case import GraphTestCase
from grakn.client import GraknClient


class TestCreateConceptGraphs(GraphTestCase):
    def test_graphs_are_created_as_expected(self):
        example_id = 0
        graphs = create_concept_graphs([example_id])
        graph = graphs[example_id]

        with GraknClient(uri="localhost:48555") as client:
            with client.session(keyspace="genealogy") as session:
                with session.transaction().read() as tx:

                    conceptmap = next(tx.query(
                        f'match '
                        f'$p0 isa person, has example-id {example_id}; '
                        f'$p1 isa person, has example-id {example_id}; '
                        f'$p2 isa person, has example-id {example_id}; '
                        f'$par0(parent: $p0, child: $p1) isa parentship, has example-id {example_id};'
                        f'$par1(parent: $p0, child: $p2) isa parentship, has example-id {example_id};'
                        f'$p1 != $p2;'
                        f'get;'))

                    p0 = build_thing(conceptmap.get('p0'))
                    p1 = build_thing(conceptmap.get('p1'))
                    p2 = build_thing(conceptmap.get('p2'))
                    par0 = build_thing(conceptmap.get('par0'))
                    par1 = build_thing(conceptmap.get('par1'))

                    real_sib = build_thing(tx.query(f'match $s($pa, $pb) isa siblingship; '
                                                    f'$pa id {p1.id}; $pb id {p2.id}; get $s;').collect_concepts()[0])
                    cand_sib_0 = build_thing(tx.query(f'match $s($pa, $pb) isa candidate-siblingship; '
                                                      f'$pa id {p0.id}; $pb id {p1.id}; get $s;').collect_concepts()[0])
                    cand_sib_1 = build_thing(tx.query(f'match $s($pa, $pb) isa candidate-siblingship; '
                                                      f'$pa id {p0.id}; $pb id {p2.id}; get $s;').collect_concepts()[0])

        expected_graph = nx.MultiDiGraph(name=0)

        # people
        expected_graph.add_node(p0, type='person', input=1, solution=1)
        expected_graph.add_node(p1, type='person', input=1, solution=1)
        expected_graph.add_node(p2, type='person', input=1, solution=1)

        # parentships
        expected_graph.add_node(par0, type='parentship', input=1, solution=1)
        expected_graph.add_edge(par0, p0, type='parent', input=1, solution=1)
        expected_graph.add_edge(par0, p1, type='child', input=1, solution=1)

        expected_graph.add_node(par1, type='parentship', input=1, solution=1)
        expected_graph.add_edge(par1, p0, type='parent', input=1, solution=1)
        expected_graph.add_edge(par1, p2, type='child', input=1, solution=1)

        # siblingships

        expected_graph.add_node(real_sib, type='siblingship', input=0, solution=1)
        expected_graph.add_edge(real_sib, p1, type='sibling', input=0, solution=1)
        expected_graph.add_edge(real_sib, p2, type='sibling', input=0, solution=1)

        # candidate siblingships
        expected_graph.add_node(cand_sib_0, type='siblingship', input=0, solution=0)
        expected_graph.add_edge(cand_sib_0, p0, type='sibling', input=0, solution=0)
        expected_graph.add_edge(cand_sib_0, p1, type='sibling', input=0, solution=0)

        expected_graph.add_node(cand_sib_1, type='siblingship', input=0, solution=0)
        expected_graph.add_edge(cand_sib_1, p0, type='sibling', input=0, solution=0)
        expected_graph.add_edge(cand_sib_1, p2, type='sibling', input=0, solution=0)

        self.assertIsIsomorphic(expected_graph, graph)
