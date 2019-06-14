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


def conceptdicts_from_query(query, tx):
    """
    Given a query, build a dictionary of the variables present and the concepts they refer to, locally storing any 
    information required about those concepts.
    :param query: The query to make
    :param tx: A live Grakn transaction
    :return: A dictionary of concepts keyed by query variables
    """
    pass


def create_thing_graph(conceptdict, variablegraph):
    """
    Create a new graph, based on a `variablegraph` that describes the interactions of variables in a query,
    and a `conceptdict` that holds objects that satisfy the query
    :param conceptdict:
    :param variablegraph:
    :return:
    """
    pass


def combine_graphs(graph1, graph2):
    """
    Combine two graphs into one. Do this by recognising common nodes between the two.
    :param graph1: Graph to compare
    :param graph2: Graph to compare
    :return: Combined graph
    """
    pass


def networkx_from_query_variablegraph_tuples(query_variablegraph_tuples, grakn_session):
    query_conceptgraphs = []

    for query, variablegraph in query_variablegraph_tuples:

        conceptdicts = conceptdicts_from_query(query, tx)

        answer_conceptgraphs = [create_thing_graph(conceptdict, variablegraph) for conceptdict in conceptdicts]
        query_conceptgraph = reduce(lambda x, y: combine_graphs(x, y), answer_conceptgraphs)
        query_conceptgraphs.append(query_conceptgraph)

    conceptgraph = reduce(lambda x, y: combine_graphs(x, y), query_conceptgraphs)
    return conceptgraph
