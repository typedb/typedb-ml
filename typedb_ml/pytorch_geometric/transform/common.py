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
from typedb_ml.networkx.iterate import multidigraph_node_data_iterator, multidigraph_edge_data_iterator


def clear_unneeded_fields(graph):
    for node_data in multidigraph_node_data_iterator(graph):
        x = node_data["x"]
        t = node_data["type"]
        node_data.clear()
        node_data["x"] = x
        node_data["type"] = t

    for edge_data in multidigraph_edge_data_iterator(graph):
        x = edge_data["edge_attr"]
        y = edge_data["y_edge"]
        t = edge_data["type"]
        edge_data.clear()
        edge_data["edge_attr"] = x
        edge_data["y_edge"] = y
        edge_data["type"] = t
    return graph


def store_concepts_by_type(graph):
    """
    Organises concepts by type the same way `data.to_heterogeneous()` organises nodes by type. This is necessary to
    be able to map back from a `HeteroData` object to the TypeDB concepts that the nodes refer to.
    Args:
        graph: NetworkX graph to operate on
    Returns:
        The same graph, with a field `concepts_by_type` holding concepts organised by type
    """
    concepts_by_type = {}
    for node_data in multidigraph_node_data_iterator(graph):
        typ = node_data['type']
        if typ in concepts_by_type:
            concepts_by_type[typ].append(node_data['concept'])
        else:
            concepts_by_type[typ] = [node_data['concept']]
    graph.concepts_by_type = concepts_by_type
    return graph
