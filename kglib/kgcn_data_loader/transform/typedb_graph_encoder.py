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
from typing import List

import numpy as np

from kglib.utils.graph.iterate import (
    multidigraph_node_data_iterator,
    multidigraph_edge_data_iterator,
)


class GraphFeatureEncoder:
    """Transform of the networkx graph as it comes out of TypeDB
    to a networkx graph that pytorch geometric likes to ingest.
    Now this is very much geared to pytorch geometric especially
    because I set the attribute names to things like "x" and
    "edge_attr" which are the standard names in pytorch geometric.

    One thing I encountered when trying to load the graph form the
    original kglib example directly in pytorch geometric is that
    since in the original example the feature vector on the nodes
    and the edges were both called "features", the stock function
    from pytorch geometric: torch_geometric.utils.from_networkx
    does not deal with this well (it ends up overwriting node
    features with edge features).

    :arg graph: networkx graph object
    :returns: networkx graph object

    """

    def __init__(self, node_types, edge_types, attribute_encoders, attribute_encoding_size):
        self.node_types = node_types
        self.edge_types = edge_types
        self.attribute_encoders = attribute_encoders
        self.attribute_encoding_size = attribute_encoding_size

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
                # encoded_value = [0] * self.attribute_encoding_size
                encoded_value = [0]

            # TODO: We can skip encoding type information as we are using the hetero models from PyTorch to take care
            #  of that for us
            one_hot_encoded_type = self.node_types.index(node_data['type'])
            node_data['x'] = np.hstack(
                [np.array(one_hot_encoded_type, dtype=np.float32), np.array(encoded_value, dtype=np.float32)]
            )

    def encode_edge_features(self, graph):
        for edge_data in multidigraph_edge_data_iterator(graph):
            one_hot_encoded_type = self.edge_types.index(edge_data['type'])
            # TODO: Unnecessary empty array added to have common features size between nodes and edges
            edge_data['edge_attr'] = np.hstack([np.array(one_hot_encoded_type, dtype=np.float32), np.array([0], dtype=np.float32)])


class CategoricalEncoder:

    def __init__(self, categories: List):
        self.categories = categories

    def __call__(self, value):
        return [self.categories.index(value)]


class ContinuousEncoder:
    def __init__(self, min_val, max_val):
        self.max_val = max_val
        self.min_val = min_val

    def __call__(self, value):
        return [(value - self.min_val) / (self.max_val - self.min_val)]


# class SequenceEncoder(object):
#     # The 'SequenceEncoder' encodes raw column strings into embeddings.
#     def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
#         self.device = device
#         self.model = SentenceTransformer(model_name, device=device)
#
#     @torch.no_grad()
#     def __call__(self, df):
#         x = self.model.encode(df.values, show_progress_bar=True,
#                               convert_to_tensor=True, device=self.device)
#         return x.cpu()
#
#
# class IdentityEncoder(object):
#     # The 'IdentityEncoder' takes the raw column values and converts them to
#     # PyTorch tensors.
#     def __init__(self, dtype=None):
#         self.dtype = dtype
#
#     def __call__(self, df):
#         return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)
