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


from functools import reduce

import networkx as nx

import kglib.kgcn.core.ingest.traverse.data.context.neighbour as neighbour
from graph.build.model.standard.convert import concept_dict_to_grakn_standard_graph


def concept_dict_from_concept_map(concept_map):
    """
    Given a concept map, build a dictionary of the variables present and the concepts they refer to, locally storing any
    information required about those concepts.
    :param concept_map: A dict of Concepts provided by Grakn keyed by query variables
    :return: A dictionary of concepts keyed by query variables
    """
    return {variable: neighbour.build_thing(grakn_concept) for variable, grakn_concept in concept_map.items()}


def combine_graphs(graph1, graph2):
    """
    Combine two graphs into one. Do this by recognising common nodes between the two.
    :param graph1: Graph to compare
    :param graph2: Graph to compare
    :return: Combined graph
    """
    return nx.compose(graph1, graph2)


def build_graph_from_queries(query_sampler_variable_graph_tuples, grakn_transaction,
                             concept_dict_converter=concept_dict_to_grakn_standard_graph):
    """
    Builds a graph of Things, interconnected by roles (and *has*), from a set of queries over a Grakn transaction
    :param query_sampler_variable_graph_tuples: A list of tuples, each tuple containing a query, a sampling function,
    and a variable_graph
    :param grakn_transaction: A Grakn transaction
    :param concept_dict_converter: The function to use to convert from concept_dicts to a Grakn model. This could be
    a typical model or a mathematical model
    :return: A networkx graph
    """
    query_concept_graphs = []

    for query, sampler, variable_graph in query_sampler_variable_graph_tuples:

        concept_maps = sampler(grakn_transaction.query(query))

        concept_dicts = [concept_dict_from_concept_map(concept_map) for concept_map in concept_maps]

        answer_concept_graphs = [concept_dict_converter(concept_dict, variable_graph) for concept_dict in concept_dicts]
        query_concept_graph = reduce(lambda x, y: combine_graphs(x, y), answer_concept_graphs)
        query_concept_graphs.append(query_concept_graph)

    concept_graph = reduce(lambda x, y: combine_graphs(x, y), query_concept_graphs)
    return concept_graph
