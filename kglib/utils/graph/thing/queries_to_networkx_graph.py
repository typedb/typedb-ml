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

import warnings

from functools import reduce
import networkx as nx

from kglib.utils.typedb.object.thing import build_thing
from kglib.utils.graph.thing.concept_dict_to_networkx_graph import concept_dict_to_graph


def concept_dict_from_concept_map(concept_map):
    """
    Given a concept map, build a dictionary of the variables present and the concepts they refer to, locally storing any
    information required about those concepts.

    Args:
        concept_map: A dict of Concepts provided by TypeDB keyed by query variables

    Returns:
        A dictionary of concepts keyed by query variables
    """
    return {variable: build_thing(typedb_concept) for variable, typedb_concept in concept_map.map().items()}


def combine_2_graphs(graph1, graph2):
    """
    Combine two graphs into one. Do this by recognising common nodes between the two.
    Args:
        graph1: Graph to compare
        graph2: Graph to compare
    Returns:
        Combined graph
    """

    for node, data in graph1.nodes(data=True):
        if graph2.has_node(node):
            data2 = graph2.nodes[node]
            if data2 != data:
                raise ValueError((f'Found non-matching node properties for node {node} '
                                  f'between graphs {graph1} and {graph2}:\n'
                                  f'In graph {graph1}: {data}\n'
                                  f'In graph {graph2}: {data2}'))

    for sender, receiver, keys, data in graph1.edges(data=True, keys=True):
        if graph2.has_edge(sender, receiver, keys):
            data2 = graph2.edges[sender, receiver, keys]
            if data2 != data:
                raise ValueError((f'Found non-matching edge properties for edge {sender, receiver, keys} '
                                  f'between graphs {graph1} and {graph2}:\n'
                                  f'In graph {graph1}: {data}\n'
                                  f'In graph {graph2}: {data2}'))

    return nx.compose(graph1, graph2)


def combine_n_graphs(graphs_list):
    """
    Combine N graphs into one. Do this by recognising common nodes between the two.
    Args:
        graphs_list: List of graphs to combine
    Returns:
        Combined graph
    """
    return reduce(lambda x, y: combine_2_graphs(x, y), graphs_list)



def build_graph_from_queries(query_sampler_variable_graph_tuples, typedb_transaction,
                             concept_dict_converter=concept_dict_to_graph):
    """
    Builds a graph of Things, interconnected by roles (and *has*), from a set of queries and graphs representing those
    queries (variable graphs)of those queries, over a TypeDB transaction

    Args:
        infer: whether to use TypeDB's inference engine
        query_sampler_variable_graph_tuples: A list of tuples, each tuple containing a query, a sampling function,
            and a variable_graph
        typedb_transaction: A TypeDB transaction
        concept_dict_converter: The function to use to convert from concept_dicts to a TypeDB model. This could be
            a typical model or a mathematical model

    Returns:
        A networkx graph
    """

    query_concept_graphs = []

    for query, sampler, variable_graph in query_sampler_variable_graph_tuples:

        print("working on query: " + query)
        concept_maps = sampler(typedb_transaction.query().match(query))
        print("query completed")

        concept_dicts = [concept_dict_from_concept_map(concept_map) for concept_map in concept_maps]
        print("constructed concept_dicts")

        answer_concept_graphs = []
        for concept_dict in concept_dicts:
            try:
                answer_concept_graphs.append(concept_dict_converter(concept_dict, variable_graph))
            except ValueError as e:
                raise ValueError(str(e) + f'Encountered processing query:\n \"{query}\"')

        if len(answer_concept_graphs) > 1:
            query_concept_graph = combine_n_graphs(answer_concept_graphs)
            query_concept_graphs.append(query_concept_graph)
        else:
            if len(answer_concept_graphs) > 0:
                query_concept_graphs.append(answer_concept_graphs[0])
            else:
                warnings.warn(f'There were no results for query: \n\"{query}\"\nand so nothing will be added to the '
                              f'graph for this query')

    if len(query_concept_graphs) == 0:
        # Raise exception when none of the queries returned any results
        raise RuntimeError(f'The graph from queries: {[query_sampler_variable_graph_tuple[0] for query_sampler_variable_graph_tuple in query_sampler_variable_graph_tuples]}\n'
                           f'could not be created, since none of these queries returned results')

    concept_graph = combine_n_graphs(query_concept_graphs)
    return concept_graph
