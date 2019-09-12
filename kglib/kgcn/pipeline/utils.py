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


def duplicate_edges_in_reverse(graph):
    """
    Takes in a directed multi graph, and creates duplicates of all edges, the duplicates having reversed direction to
    the originals. This is useful since directed edges constrain the direction of messages passed. We want to permit
    omni-directional message passing.
    :param graph: The graph
    :return: The graph with duplicated edges, reversed, with all original edge properties attached to the duplicates
    """
    for sender, receiver, keys, data in graph.edges(data=True, keys=True):
        graph.add_edge(receiver, sender, keys, **data)
    return graph


def apply_logits_to_graphs(graph, logits_graph):
    """
    Take in a graph that describes the logits of the graph of interest, and store those logits on the graph as the
    property 'logits'. The graphs must correspond with one another

    Args:
        graph: Graph to apply logits to
        logits_graph: Graph containing logits

    Returns:
        graph with logits added as property 'logits'
    """

    for node, data in logits_graph.nodes(data=True):
        graph.nodes[node]['logits'] = list(data['features'])

    # TODO This is the desired implementation, but the graphs are altered by the model to have duplicated reversed
    #  edges, so this won't work for now
    # for sender, receiver, keys, data in logit_graph.edges(keys=True, data=True):
    #     graph.edges[sender, receiver, keys]['logits'] = list(data['features'])

    for sender, receiver, keys, data in graph.edges(keys=True, data=True):
        data['logits'] = list(logits_graph.edges[sender, receiver, keys]['features'])

    return graph
