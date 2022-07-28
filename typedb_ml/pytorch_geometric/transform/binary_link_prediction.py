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
import networkx as nx

from typedb_ml.networkx.iterate import multidigraph_edge_data_iterator
from typedb_ml.typedb.type import get_edge_type_triplets, reverse_edge_type_triplets


class LinkPredictionLabeller:

    def __init__(self, edge_type_to_predict):
        self.edge_type_to_predict = edge_type_to_predict

    def __call__(self, graph):
        self.label_edges(graph)
        return graph

    def label_edges(self, graph):
        for data in multidigraph_edge_data_iterator(graph):
            if data["type"] == self.edge_type_to_predict:
                data["y_edge"] = 1
            else:
                data["y_edge"] = 0


def binary_relations_to_edges(graph: nx.MultiDiGraph, binary_relation_type):
    """
    A function to convert a TypeDB relation (a hyperedge) to a directed binary edge.

    The replaced relations may not be:
    - a roleplayer in any other relations
    - own any attributes
    - have more than two roles
    and must:
    - have exactly two outgoing role edges (which can be of the same role)

    This is useful because edges are often stored as an adjacency matrix. In PyTorch Geometric, for example,
    this adjacency matrix expects binary edges. To be compatible we convert to binary edges so that when creating
    negative samples we can simply change the values of the adjacency matrix. In the case of a hyperedge we cannot
    simply do this to add negative edges.

    Args:
        graph: The graph to modify in-place
        binary_relation_type: A triple of the `(role1, relation, role2)` types to convert to a single edge labelled with `relation`

    Returns:
        Dict of edges to the node each replaces
    """
    replacement_edges = {}
    for node, node_data in graph.nodes(data=True):
        if node_data["type"] == binary_relation_type[1]:
            if len(list(graph.in_edges(node))) > 0:
                raise ValueError(
                    "The given binary relation can't be transformed into an edge because it plays a role in another "
                    "relation."
                )
            roles = list(graph.edges(node, data=True))  # equivalent to out_edges()
            assert len(roles) == 2
            new_edge_start = None
            new_edge_end = None
            for from_roleplayer, to_roleplayer, data in roles:
                if data["type"] == binary_relation_type[0]:
                    if new_edge_start is None:
                        new_edge_start = to_roleplayer
                    else:
                        # In this case the given relation type is symmetric. Therefore direction is arbitrary (but
                        # expect to add a reverse edge elsewhere).
                        assert binary_relation_type[0] == binary_relation_type[2]
                        new_edge_end = to_roleplayer
                elif data["type"] == binary_relation_type[2]:
                    new_edge_end = to_roleplayer
                else:
                    raise ValueError(
                        f"Unexpected role in relation {binary_relation_type[1]}. Expected \""
                        f"{binary_relation_type[0]}\" or \"{binary_relation_type[2]}\" but got \"{data['type']}\"."
                    )
                graph.remove_edge(from_roleplayer, to_roleplayer)
            graph.add_edge(new_edge_start, new_edge_end, type=binary_relation_type[1])
            replacement_edges[(new_edge_start, new_edge_end)] = node
    for node in replacement_edges.values():
        graph.remove_node(node)
    return graph


def binary_link_prediction_edge_triplets(session, relation_type_to_predict, types_to_ignore):
    edge_type_triplets = get_edge_type_triplets(session)
    edge_type_triplets = [e for e in edge_type_triplets if not (set(e).intersection(types_to_ignore))]
    replace_relation_with_binary_edge(edge_type_triplets, relation_type_to_predict)
    edge_type_triplets_reversed = reverse_edge_type_triplets(edge_type_triplets)
    return edge_type_triplets, edge_type_triplets_reversed


def replace_relation_with_binary_edge(edge_type_triplets, relation_type_to_predict):
    # Remove the two roles of the relation_to_predict and replace them with a binary edge that uses the name of the
    # binary relation it represents
    edge_type_triplets.remove(relation_type_to_predict[2:])
    if relation_type_to_predict[1] != relation_type_to_predict[3]:
        edge_type_triplets.remove(tuple(reversed(relation_type_to_predict[:3])))
    edge_type_triplets.append((relation_type_to_predict[0], relation_type_to_predict[2], relation_type_to_predict[4]))
