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

import numpy as np

from kglib.utils.graph.iterate import multidigraph_node_data_iterator, multidigraph_edge_data_iterator


def encode_values(graph, attribute_encoders, attribute_encoding_size):
    for node_data in multidigraph_node_data_iterator(graph):
        typ = node_data['type']
        if typ in attribute_encoders.keys():
            # Add the integer value of the category for each categorical attribute instance
            node_data['encoded_value'] = attribute_encoders[typ].encode(node_data['value'])
        else:
            node_data['encoded_value'] = [0] * attribute_encoding_size

    for edge_data in multidigraph_edge_data_iterator(graph):
        edge_data['encoded_value'] = [0] * attribute_encoding_size


def encode_types(graph, iterator_func, all_types):
    """
    Encodes the type found in graph data as an integer according to the index it is found in `all_types`
    Args:
        graph: The graph to encode
        iterator_func: An function to create an iterator of data in the graph (node data, edge data or combined node
        and edge data)
        all_types: The full list of types to be encoded in this order

    Returns:
        Nothing, the graph is updated in-place

    """
    iterator = iterator_func(graph)

    for data in iterator:
        data['categorical_type'] = all_types.index(data['type'])


def stack_features(features):
    """
    Stacks features together into a single vector

    Args:
        features: iterable of features, features can be a single value or iterable

    Returns:
        Numpy array (vector) of stacked features

    """

    return np.hstack([np.array(feature, dtype=np.float32) for feature in features])
