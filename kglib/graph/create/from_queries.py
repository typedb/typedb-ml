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

import kglib.kgcn.core.ingest.traverse.data.context.neighbour as neighbour
from kglib.graph.create.model.standard.convert import concept_dict_to_grakn_standard_graph


def concept_dict_from_concept_map(concept_map):
    """
    Given a concept map, build a dictionary of the variables present and the concepts they refer to, locally storing any
    information required about those concepts.
    :param concept_map: A dict of Concepts provided by Grakn keyed by query variables
    :return: A dictionary of concepts keyed by query variables
    """
    return {variable: neighbour.build_thing(grakn_concept) for variable, grakn_concept in concept_map.map().items()}


def combine_2_graphs(graph1, graph2):
    """
    Combine two graphs into one. Do this by recognising common nodes between the two.
    :param graph1: Graph to compare
    :param graph2: Graph to compare
    :return: Combined graph
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
    :param graphs_list: List of graphs to combine
    :return: Combined graph
    """
    return reduce(lambda x, y: combine_2_graphs(x, y), graphs_list)


def build_graph_from_queries(query_sampler_variable_graph_tuples, grakn_transaction,
                             concept_dict_converter=concept_dict_to_grakn_standard_graph, infer=True):
    """
    Builds a graph of Things, interconnected by roles (and *has*), from a set of queries and graphs representing those
    queries (variable graphs)of those queries, over a Grakn transaction
    :param infer:
    :param query_sampler_variable_graph_tuples: A list of tuples, each tuple containing a query, a sampling function,
    and a variable_graph
    :param grakn_transaction: A Grakn transaction
    :param concept_dict_converter: The function to use to convert from concept_dicts to a Grakn model. This could be
    a typical model or a mathematical model
    :return: A networkx graph
    """
    query_concept_graphs = []

    for query, sampler, variable_graph in query_sampler_variable_graph_tuples:

        concept_maps = sampler(grakn_transaction.query(query, infer=infer))

        concept_dicts = [concept_dict_from_concept_map(concept_map) for concept_map in concept_maps]

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
