#
#  Copyright (C) 2021 Vaticle
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

from kglib.utils.graph.iterate import multidigraph_data_iterator, multidigraph_node_data_iterator, \
    multidigraph_edge_data_iterator


def encode_values(graph, categorical_attributes, continuous_attributes):
    for node_data in multidigraph_node_data_iterator(graph):
        typ = node_data['type']

        if categorical_attributes is not None and typ in categorical_attributes.keys():
            # Add the integer value of the category for each categorical attribute instance
            category_values = categorical_attributes[typ]
            node_data['encoded_value'] = category_values.index(node_data['value'])

        elif continuous_attributes is not None and typ in continuous_attributes.keys():
            min_val, max_val = continuous_attributes[typ]
            node_data['encoded_value'] = (node_data['value'] - min_val) / (max_val - min_val)

        else:
            node_data['encoded_value'] = 0
    for edge_data in multidigraph_edge_data_iterator(graph):
        edge_data['encoded_value'] = 0

    return graph


def encode_types(graph, iterator_func, types):
    """
    Encodes the type found in graph data as an integer according to the index it is found in `all_types`
    Args:
        graph: The graph to encode
        iterator_func: An function to create an iterator of data in the graph (node data, edge data or combined node and edge data)
        types: The full list of types to be encoded in this order
        
    Returns:
        The graph, which is also is updated in-place

    """
    iterator = iterator_func(graph)

    for data in iterator:
        data['categorical_type'] = types.index(data['type'])

    return graph


def create_input_graph(graph):
    input_graph = graph.copy()

    for data in multidigraph_data_iterator(input_graph):
        if data["solution"] == 0:
            preexists = 1
        else:
            preexists = 0

        features = stack_features([preexists, data["categorical_type"], data["encoded_value"]])
        data.clear()
        data["features"] = features

    input_graph.graph["features"] = np.array([0.0] * 5, dtype=np.float32)
    return input_graph


def create_target_graph(graph):
    target_graph = graph.copy()
    solution_one_hot_encoding = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=np.float32)

    for data in multidigraph_data_iterator(target_graph):
        features = solution_one_hot_encoding[data["solution"]]
        data.clear()
        data["features"] = features

    target_graph.graph["features"] = np.array([0.0] * 5, dtype=np.float32)
    return target_graph


def stack_features(features):
    """
    Stacks features together into a single vector

    Args:
        features: iterable of features, features can be a single value or iterable

    Returns:
        Numpy array (vector) of stacked features

    """

    return np.hstack([np.array(feature, dtype=np.float32) for feature in features])
