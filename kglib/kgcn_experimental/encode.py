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

import numpy as np


def graph_to_input_target(graph):
    """Returns 2 graphs with input and target feature vectors for training.

    Args:
    graph: An `nx.MultiDiGraph` instance.

    Returns:
    The input `nx.MultiDiGraph` instance.
    The target `nx.MultiDiGraph` instance.

    Raises:
    ValueError: unknown node type
    """

    def create_feature(attr, fields):
        return np.hstack([np.array(attr[field], dtype=float) for field in fields])

    input_node_fields = ("input", "one_hot_type")
    input_edge_fields = ("input", "one_hot_type")
    target_node_fields = ("solution",)
    target_edge_fields = ("solution",)

    input_graph = graph.copy()
    target_graph = graph.copy()

    for node_index, node_feature in graph.nodes(data=True):
        input_graph.nodes[node_index]['features'] = create_feature(node_feature, input_node_fields)
        target_node = to_one_hot(create_feature(node_feature, target_node_fields).astype(int), 2)[0]
        target_graph.nodes[node_index]['features'] = target_node

    for receiver, sender, edge_features in graph.edges(data=True):
        input_graph.edges[receiver, sender, 0]['features'] = create_feature(edge_features, input_edge_fields)

        target_edge = to_one_hot(create_feature(edge_features, target_edge_fields).astype(int), 2)[0]
        target_graph.edges[receiver, sender, 0]['features'] = target_edge

    input_graph.graph["features"] = np.array([0.0]*5)
    target_graph.graph["features"] = np.array([0.0]*5)

    return input_graph, target_graph


def to_one_hot(indices, max_value, axis=-1):
    one_hot = np.eye(max_value)[indices]
    if axis not in (-1, one_hot.ndim):
        one_hot = np.moveaxis(one_hot, -1, axis)
    return one_hot


def encode_types_one_hot(G, all_node_types, all_edge_types, attribute='one_hot_type', type_attribute='type'):
    """
    Creates a one-hot encoding for every element in the graph, based on the "type" attribute of each element.
    Adds this one-hot vector to each element as `attribute`
    :param G: The graph to encode
    :param all_node_types: The list of node types to encode from
    :param all_edge_types: The list of edge types to encode from
    :param attribute: The attribute to store the encodings on
    :param type_attribute: The pre-existing attribute that indicates the type of the element
    """

    # TODO Catch the case where all types haven't been given correctly
    for node_index, node_feature in G.nodes(data=True):
        one_hot = np.zeros(len(all_node_types), dtype=np.int)
        index_to_one_hot = all_node_types.index(node_feature[type_attribute])
        one_hot[index_to_one_hot] = 1
        G.nodes[node_index][attribute] = one_hot

    for sender, receiver, keys, edge_feature in G.edges(data=True, keys=True):
        one_hot = np.zeros(len(all_edge_types), dtype=np.int)
        index_to_one_hot = all_edge_types.index(edge_feature[type_attribute])
        one_hot[index_to_one_hot] = 1
        G.edges[sender, receiver, keys][attribute] = one_hot