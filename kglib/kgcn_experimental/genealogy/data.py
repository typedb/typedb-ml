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

import networkx as nx
from grakn.client import GraknClient

from kglib.graph.create.from_queries import build_graph_from_queries
from kglib.kgcn_experimental.utils.iterate import multidigraph_data_iterator

KEYSPACE = "genealogy"
URI = "localhost:48555"


def create_concept_graphs(example_indices):
    graphs = []
    qh = QueryHandler()

    with GraknClient(uri=URI) as client:
        with client.session(keyspace=KEYSPACE) as session:

            material_graphs = dict()
            infer = False
            for example_id in example_indices:
                print(f'Creating material graph for example {example_id}')
                material_graph_query_handles = qh.get_initial_query_handles(example_id)
                with session.transaction().read() as tx:
                    # Build a graph from the queries, samplers, and query graphs
                    graph = build_graph_from_queries(material_graph_query_handles, tx, infer=infer)
                material_graphs[example_id] = graph

            # Make any changes or additions to the graph. This could include making match-insert queries or adding
            # rules in order to add candidates for prediction, and/or positive or negative examples
            infer = True

            for example_id in example_indices:
                print(f'Creating inferred graph for example {example_id}')
                inferred_graph_query_handles = qh.get_query_handles_with_inference(example_id)
                with session.transaction().read() as tx:
                    # Build a graph from the queries, samplers, and query graphs
                    inferred_graph = build_graph_from_queries(inferred_graph_query_handles, tx, infer=infer)
                print(f'Creating combined graph for example {example_id}')
                graph = nx.compose(inferred_graph, material_graphs[example_id])

                # Remove label leakage - change type labels that indicate candidates into non-candidates

                for data in multidigraph_data_iterator(graph):
                    typ = data['type']
                    if typ == 'candidate-siblingship':
                        data.update(type='siblingship')
                    elif typ == 'candidate-sibling':
                        data.update(type='sibling')

                graph.name = example_id
                graphs.append(graph)

    return graphs


class QueryHandler:

    def __init__(self):
        # Existing elements in the graph are those that pre-exist in the graph, and should be predicted to continue to
        # exist
        self.existing = dict(input=1, solution=1)

        # Elements to infer are the graph elements whose existence we want to predict to be true, they are positive
        # examples
        self.to_infer = dict(input=0, solution=1)

        # Candidates are neither present in the input nor in the solution, they are negative examples
        self.candidate = dict(input=0, solution=0)

    def parentship_query(self, example_id):
        return (
            f'match '
            f'$p1 isa person, has example-id {example_id};'
            f'$p2 isa person, has example-id {example_id};'
            f'$r(parent: $p1, child: $p2) isa parentship;'
            f'get $p1, $p2, $r;'
        )

    def parentship_query_graph(self):
        g = nx.MultiDiGraph()
        g.add_node('p1', **self.existing)
        g.add_node('p2', **self.existing)
        g.add_node('r', **self.existing)
        g.add_edge('r', 'p1', type='parent', **self.existing)
        g.add_edge('r', 'p2', type='child', **self.existing)
        return g

    def grandparentship_query(self, example_id):
        return (
            f'match '
            f'$p1 isa person, has example-id {example_id};'
            f'$p2 isa person, has example-id {example_id};'
            f'$r(grandparent: $p1, grandchild: $p2) isa grandparentship;'
            f'get $p1, $p2, $r;'
        )

    def grandparentship_query_graph(self):
        g = nx.MultiDiGraph()
        g.add_node('p1', **self.existing)
        g.add_node('p2', **self.existing)
        g.add_node('r', **self.existing)
        g.add_edge('r', 'p1', type='grandparent', **self.existing)
        g.add_edge('r', 'p2', type='grandchild', **self.existing)
        return g

    def siblingship_query(self, example_id):
        return (
            f'match '
            f'$p1 isa person, has example-id {example_id};'
            f'$r(sibling: $p1) isa siblingship;'
            f'get $p1, $r;'
        )

    def material_siblingship_query_graph(self):
        g = nx.MultiDiGraph()
        g.add_node('p1', **self.existing)
        g.add_node('r', **self.existing)
        g.add_edge('r', 'p1', type='sibling', **self.existing)
        return g

    def siblingship_query_graph(self):
        g = nx.MultiDiGraph()
        g.add_node('p1', **self.existing)
        g.add_node('r', **self.to_infer)
        g.add_edge('r', 'p1', type='sibling', **self.to_infer)
        return g

    def candidate_siblingship_query(self, example_id):
        return (
            f'match '
            f'$p1 isa person, has example-id {example_id};'
            f'$r(candidate-sibling: $p1) isa candidate-siblingship;'
            f'get $p1, $r;'
        )

    def candidate_siblingship_query_graph(self):
        g = nx.MultiDiGraph()
        g.add_node('p1', **self.existing)
        g.add_node('r', **self.candidate)
        g.add_edge('r', 'p1', type='candidate-sibling', **self.candidate)
        return g

    def get_initial_query_handles(self, example_id):

        return [
            (self.parentship_query(example_id), lambda x: x, self.parentship_query_graph()),
            (self.siblingship_query(example_id), lambda x: x, self.material_siblingship_query_graph()),
        ]

    def get_query_handles_with_inference(self, example_id):

        return [
            (self.grandparentship_query(example_id), lambda x: x, self.grandparentship_query_graph()),
            (self.siblingship_query(example_id), lambda x: x, self.siblingship_query_graph()),
            (self.candidate_siblingship_query(example_id), lambda x: x,
             self.candidate_siblingship_query_graph())
        ]
