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
from graph_nets import utils_tf

from kglib.kgcn_experimental.encode import encode_types_one_hot, graph_to_input_target


def make_all_runnable_in_session(*args):
    """Lets an iterable of TF graphs be output from a session as NP graphs."""
    return [utils_tf.make_runnable_in_session(a) for a in args]


def create_input_target_graphs(graphs, all_node_types, all_edge_types):
    """
    Builds graphs ready to be used for training
    :param graph: The list of graphs to use
    :param all_node_types: All of the types that can occur at nodes, for encoding purposes
    :param all_edge_types: All of the types that can occur at edges, for encoding purposes
    :return: the input graphs, the target (desired output) graphs, and the original_graphs
    """

    input_graphs = []
    target_graphs = []
    for graph in graphs:
        encode_types_one_hot(graph, all_node_types, all_edge_types, attribute='one_hot_type', type_attribute='type')

        input_graph, target_graph = graph_to_input_target(graph)
        input_graphs.append(input_graph)
        target_graphs.append(target_graph)

    return input_graphs, target_graphs


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
