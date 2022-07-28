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
from typing import List

import torch
from torch.nn import Embedding

from typedb_ml.networkx.iterate import (
    multidigraph_node_data_iterator,
    multidigraph_edge_data_iterator,
)


class FeatureEncoder:
    """
    Feature encoder for NetworkX representations of TypeDB data. Type data is assumed always present for each node
    and edge, consistent with the TypeDB knowledge model. Therefore, this encoder provides a de-factor method to
    embed that type information. Supply attribute-specific encoders to handle encoding values of attributes,
    since the meaning of these values is domain-dependent.
    """

    def __init__(self, node_types, edge_types, type_encoding_size, attribute_encoders, attribute_encoding_size):
        self.node_types = node_types
        self.edge_types = edge_types
        self.attribute_encoders = attribute_encoders
        self.attribute_encoding_size = attribute_encoding_size
        self.node_type_embedding = Embedding(len(self.node_types), type_encoding_size)
        self.edge_type_embedding = Embedding(len(self.edge_types), type_encoding_size)

    def __call__(self, graph):
        self.encode_node_features(graph)
        self.encode_edge_features(graph)
        return graph

    def encode_node_features(self, graph):
        for node_data in multidigraph_node_data_iterator(graph):
            typ = node_data['type']
            if typ in self.attribute_encoders.keys():
                # Add the integer value of the category for each categorical attribute instance
                encoded_value = self.attribute_encoders[typ](node_data['value'])
            else:
                encoded_value = [0] * self.attribute_encoding_size

            type_embedding = self.node_type_embedding(torch.as_tensor(self.node_types.index(node_data['type'])))
            node_data['x'] = torch.hstack([type_embedding, torch.as_tensor(encoded_value)])\
                .cpu().detach().numpy()  # Conversion to numpy array, otherwise the graph representation breaks

    def encode_edge_features(self, graph):
        for edge_data in multidigraph_edge_data_iterator(graph):
            type_embedding = self.node_type_embedding(torch.as_tensor(self.edge_types.index(edge_data['type'])))
            edge_data['edge_attr'] = torch.hstack([type_embedding, torch.as_tensor([0] * self.attribute_encoding_size)])\
                .cpu().detach().numpy()  # Conversion to numpy array, otherwise the graph representation breaks


class CategoricalEncoder:

    def __init__(self, categories: List, attribute_encoding_size):
        self.categories = categories
        self.embedding = Embedding(len(self.categories), attribute_encoding_size)

    def __call__(self, value):
        return self.embedding(torch.as_tensor(self.categories.index(value)))


class ContinuousEncoder:
    def __init__(self, min_val, max_val, attribute_encoding_size):
        self.attribute_encoding_size = attribute_encoding_size
        self.max_val = max_val
        self.min_val = min_val

    def __call__(self, value):
        return [(value - self.min_val) / (self.max_val - self.min_val)] * self.attribute_encoding_size
