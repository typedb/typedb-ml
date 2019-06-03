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
import itertools

import networkx as nx

import experiment.plotting


def generate_graph(num_people):
    """
    Create a graph of people, where all people are connected to all others as:
    - the parent in a parentship
    - the child in a parentship
    - a sibling in a siblingship
    The nodes are consecutive integers, starting from zero, with the `num_people` as the first `num_people` nodes.
    The type of each node and edge is stored in the data field `type`.
    The model used imitates the classic data model seen by Grakn users (rather than the meta-model)

    :param num_people: the number of people in the graph
    :return: the graph
    """

    G = nx.DiGraph()

    G.add_nodes_from(range(num_people), type="person")

    last_id = len(G.nodes)
    for i, (node_1, node_2) in enumerate(itertools.permutations(range(num_people), 2)):
        parentship_node = i + last_id
        G.add_node(parentship_node, type="parentship")
        G.add_edge(parentship_node, node_1, type="parent")
        G.add_edge(parentship_node, node_2, type="child")

    last_id = len(G.nodes)

    for i, (node_1, node_2) in enumerate(itertools.combinations(range(num_people), 2)):
        siblingship_node = i + last_id
        G.add_node(siblingship_node, type="siblingship")
        G.add_edge(siblingship_node, node_1, type="sibling")
        G.add_edge(siblingship_node, node_2, type="sibling")

    return G


if __name__ == "__main__":
    G = generate_graph(4)
    experiment.plotting.plot_with_matplotlib(G)
