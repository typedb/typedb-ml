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

from  kglib.graph.label.label import label_nodes_by_property, label_edges_by_property
from kglib.graph.create.from_queries import build_graph_from_queries, concept_graph_to_indexed_graph



def create_graphs(example_indices):
    """

    :param example_indices:
    :return:
    """
    graphs = []
    examples = [get_example_graph_query_sampler_query_graph_tuples(example_id) for example_id in example_indices]

    with GraknClient(uri="localhost:48555") as client:
        with client.session(keyspace="genealogy") as session:

            for query_sampler_variable_graph_tuples in examples:

                with session.transaction().write() as tx:

                    # Build a graph from the queries, samplers, and query graphs
                    graph = build_graph_from_queries(query_sampler_variable_graph_tuples, tx)

                    # Remove label leakage - change type labels that indicate candidates into non-candidates
                    label_nodes_by_property(graph, 'type', 'candidate-siblingship', {'type', 'siblingship'})
                    label_edges_by_property(graph, 'type', 'candidate-sibling', {'type', 'sibling'})

                    # Change the graph to be indexed by integers
                    indexed_graph = concept_graph_to_indexed_graph(graph)

                    graphs.append(indexed_graph)

    return graphs


def get_example_graph_query_sampler_query_graph_tuples(example_id):

    # Existing elements in the graph are those that pre-exist in the graph, and should be predicted to continue to exist
    existing = dict(input=1, solution=1)

    # Elements to infer are the graph elements whose existence we want to predict to be true, they are positive examples
    to_infer = dict(input=0, solution=1)

    # Candidates are neither present in the input nor in the solution, they are negative examples
    candidate = dict(input=0, solution=0)

    # parentship
    parentship_query = (
        f'match '
        f'$p1 isa person, has example-id {example_id};'
        f'$p2 isa person, has example-id {example_id};'
        f'$r(parent: $p1, child: $p2) isa parentship;'
        f'get $p1, $p2, $r;'
    )
    print(parentship_query)

    g = nx.MultiDiGraph()
    g.add_node('p1', **existing)
    g.add_node('p2', **existing)
    g.add_node('r', **existing)
    g.add_edge('r', 'p1', type='parent', **existing)
    g.add_edge('r', 'p2', type='child', **existing)
    parentship_query_graph = g

    # siblingship
    siblingship_query = (
        f'match '
        f'$p1 isa person, has example-id {example_id};'
        f'$r(sibling: $p1) isa siblingship;'
        f'get $p1, $r;'
    )
    print(siblingship_query)

    g2 = nx.MultiDiGraph()
    g2.add_node('p1', **existing)
    g2.add_node('r', **to_infer)
    g2.add_edge('r', 'p1', type='sibling', **to_infer)
    siblingship_query_graph = g2

    # candidate siblingship
    candidate_siblingship_query = (
        f'match '
        f'$p1 isa person, has example-id {example_id};'
        f'$r(sibling: $p1) isa candidate-siblingship;'
        f'get $p1, $r;'
    )
    print(candidate_siblingship_query)

    g3 = nx.MultiDiGraph()
    g3.add_node('p1', **existing)
    g3.add_node('r', **candidate)
    g3.add_edge('r', 'p1', type='candidate-sibling', **candidate)

    candidate_siblingship_query_graph = g3

    query_sampler_query_graph_tuples = [
        (parentship_query, lambda x: x, parentship_query_graph),
        (siblingship_query, lambda x: x, siblingship_query_graph),
        (candidate_siblingship_query, lambda x: x, candidate_siblingship_query_graph)
    ]

    return query_sampler_query_graph_tuples


if __name__ == "__main__":
    get_example_graph_query_sampler_query_graph_tuples(0)
