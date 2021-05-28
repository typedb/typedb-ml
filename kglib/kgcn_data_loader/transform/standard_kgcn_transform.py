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

import networkx as nx
from kglib.utils.graph.iterate import (
    multidigraph_node_data_iterator,
    multidigraph_data_iterator,
    multidigraph_edge_data_iterator,
)

from kglib.kgcn_data_loader.encoding.standard_encode import encode_types, encode_values, stack_features
from kglib.kgcn_data_loader.utils import duplicate_edges_in_reverse


class StandardKGCNNetworkxTransform:
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

    def __init__(
        self,
        node_types,
        edge_types,
        target_name,
        obfuscate=None,
        categorical=None,
        continuous=None,
        duplicate_in_reverse=True,
        label_attribute="concept",
    ):
        self.node_types = node_types
        self.edge_types = edge_types
        self.target_name = target_name
        self.obfuscate = obfuscate or {}
        self.categorical = categorical or {}
        self.continuous = continuous or {}
        self.duplicate = duplicate_in_reverse
        self.label_attribute = label_attribute

    def __call__(self, graph):
        if self.obfuscate:
            obfuscate_labels(graph, self.obfuscate)
        # Encode attribute values as number
        graph = encode_values(graph, self.categorical, self.continuous)
        graph = nx.convert_node_labels_to_integers(
            graph, label_attribute=self.label_attribute
        )
        if self.duplicate:
            graph = duplicate_edges_in_reverse(graph)
        # Node or Edge Type as int
        graph = encode_types(graph, multidigraph_node_data_iterator, self.node_types)
        graph = encode_types(graph, multidigraph_edge_data_iterator, self.edge_types)

        for data in multidigraph_node_data_iterator(graph):
            features = create_feature_vector(data)
            target = data[self.target_name]
            data.clear()
            data["x"] = features
            data["y"] = target

        for data in multidigraph_edge_data_iterator(graph):
            features = create_feature_vector(data)
            target = data[self.target_name]
            data.clear()
            data["edge_attr"] = features
            data["y_edge"] = target

        return graph


def create_feature_vector(node_or_edge_data_dict):
    """Make a feature 3-dimensional feature vector,

    Factored out of kglib.kgcn_tensorflow.pipeline.encode.create_input_graph.

    Args:
        node_or_edge_dict: the dict coming describing a node or edge
        obtained from an element of graph.nodes(data=True) or graph.edges(data=True)
        of a networkx graph.

    Returns:
        Numpy array (vector) of stacked features

    """
    if node_or_edge_data_dict["solution"] == -1:
        preexists = 1
    else:
        preexists = 0
    features = stack_features(
        [
            preexists,
            node_or_edge_data_dict["categorical_type"],
            node_or_edge_data_dict["encoded_value"],
        ]
    )
    return features


def obfuscate_labels(graph, types_and_roles_to_obfuscate):
    """Taken directly from diagnosis.py from the kglib example"""
    # Remove label leakage - change type labels that indicate candidates into non-candidates
    for data in multidigraph_data_iterator(graph):
        for label_to_obfuscate, with_label in types_and_roles_to_obfuscate.items():
            if data["type"] == label_to_obfuscate:
                data.update(type=with_label)
                break
