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

import numpy as np


def generate_graph(num_people: int) -> nx.DiGraph:
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

    last_id = G.number_of_nodes()
    for i, (node_1, node_2) in enumerate(itertools.permutations(range(num_people), 2)):
        parentship_node = i + last_id
        G.add_node(parentship_node, type="parentship")
        G.add_edge(parentship_node, node_1, type="parent")
        G.add_edge(parentship_node, node_2, type="child")

    last_id = G.number_of_nodes()

    for i, (node_1, node_2) in enumerate(itertools.combinations(range(num_people), 2)):
        siblingship_node = i + last_id
        G.add_node(siblingship_node, type="siblingship")
        G.add_edge(siblingship_node, node_1, type="sibling")
        G.add_edge(siblingship_node, node_2, type="sibling")

    return G


def add_base_labels(G, *attributes):
    """
    Add basic input labels to the Graph to indicate whether nodes and edges are known to exist in the input. These are
    indicated with `attribute` set to 1. Elements that are not known to exist are indicated with `attribute` set to 0.
    :param attribute: where to store the labelling
    :param G: the graph to label
    """
    for node in G.nodes:
        if G.nodes[node]['type'] == 'person':
            value = 1
        else:
            value = 0

        for attribute in attributes:
            G.nodes[node][attribute] = value

    for edge in G.edges:
        for attribute in attributes:
            G.edges[edge][attribute] = 0


def find_relation_node(G, relation_type, roles, roleplayers):
    for node in set(G.predecessors(roleplayers[0])).intersection(G.predecessors(roleplayers[1])):

        if G.nodes[node]['type'] == relation_type \
                and G.edges[node, roleplayers[0]]['type'] == roles[0] \
                and G.edges[node, roleplayers[1]]['type'] == roles[1]:
            return node


def add_relation_label(G, relation_type, roles, roleplayers, *attributes):
    relation = find_relation_node(G, relation_type, roles, roleplayers)
    for attribute in attributes:
        G.nodes[relation][attribute] = 1
        G.edges[relation, roleplayers[0]][attribute] = 1
        G.edges[relation, roleplayers[1]][attribute] = 1


def add_parentship(G, parent_node, child_node, *attributes):
    add_relation_label(G, 'parentship', ['parent', 'child'], [parent_node, child_node], *attributes)


def add_siblingship(G, sibling_node_1, sibling_node_2, *attributes):
    add_relation_label(G, 'siblingship', ['sibling', 'sibling'], (sibling_node_1, sibling_node_2), *attributes)


def create_graph(i):

    G = None

    existing_elements = ('input', 'solution')
    to_induce = ('solution',)

    def base_graph(num_people):
        G = generate_graph(num_people)
        add_base_labels(G, *existing_elements)
        return G

    # ---- 3-person graphs ----

    if i == 0:
        G = base_graph(3)
        add_parentship(G, 0, 1, *existing_elements)
        add_parentship(G, 0, 2, *existing_elements)
        add_siblingship(G, 1, 2, *to_induce)

    elif i == 1:
        # All mutual siblings, no parents
        G = base_graph(3)
        add_siblingship(G, 0, 1, *existing_elements)
        add_siblingship(G, 1, 2, *existing_elements)
        add_siblingship(G, 0, 2, *to_induce)

    elif i == 2:
        G = base_graph(3)
        add_parentship(G, 0, 1, *existing_elements)
        add_parentship(G, 1, 2, *existing_elements)

    elif i == 3:
        G = base_graph(3)
        add_parentship(G, 0, 1, *existing_elements)
        add_siblingship(G, 1, 2, *existing_elements)

    # elif i == 4:
        # # Infers a parentship
        # G = base_graph(num_people)
        # add_parentship(G, 0, 1, *existing_elements)
        # add_parentship(G, 0, 2, *to_induce)
        # add_siblingship(G, 1, 2, *existing_elements)

    # ---- 4-person graphs ----
    elif i == 5:
        G = base_graph(4)
        add_parentship(G, 0, 1, *existing_elements)
        add_parentship(G, 0, 2, *existing_elements)
        add_parentship(G, 2, 3, *existing_elements)
        add_siblingship(G, 1, 2, *to_induce)

    elif i == 6:
        G = base_graph(4)
        add_parentship(G, 0, 1, *existing_elements)
        add_parentship(G, 0, 2, *existing_elements)
        add_parentship(G, 2, 3, *existing_elements)
        add_siblingship(G, 1, 2, *to_induce)

    elif i == 7:
        G = base_graph(4)
        add_parentship(G, 0, 1, *existing_elements)
        add_parentship(G, 0, 2, *existing_elements)
        add_parentship(G, 3, 0, *existing_elements)
        add_siblingship(G, 1, 2, *to_induce)

    elif i == 8:
        G = base_graph(4)
        add_parentship(G, 0, 1, *existing_elements)
        add_parentship(G, 1, 2, *existing_elements)
        add_parentship(G, 2, 3, *existing_elements)

    elif i == 9:
        # 1 parent to 3 siblings
        G = base_graph(4)
        add_parentship(G, 0, 1, *existing_elements)
        add_parentship(G, 0, 2, *existing_elements)
        add_parentship(G, 0, 3, *existing_elements)
        add_siblingship(G, 1, 2, *to_induce)
        add_siblingship(G, 2, 3, *to_induce)
        add_siblingship(G, 1, 3, *to_induce)

    elif i == 10:
        # All mutual siblings, no parents
        G = base_graph(4)
        add_siblingship(G, 0, 1, *existing_elements)
        add_siblingship(G, 1, 2, *existing_elements)
        add_siblingship(G, 2, 3, *existing_elements)
        add_siblingship(G, 0, 2, *to_induce)
        add_siblingship(G, 0, 3, *to_induce)
        add_siblingship(G, 1, 3, *to_induce)

    elif i == 11:
        G = base_graph(4)
        add_siblingship(G, 0, 1, *existing_elements)
        add_siblingship(G, 1, 2, *existing_elements)
        add_siblingship(G, 2, 3, *existing_elements)
        add_siblingship(G, 0, 2, *to_induce)
        add_siblingship(G, 0, 3, *to_induce)
        add_siblingship(G, 1, 3, *to_induce)

    elif i == 12:
        G = base_graph(4)
        add_parentship(G, 0, 1, *existing_elements)
        add_parentship(G, 0, 2, *existing_elements)
        add_parentship(G, 3, 1, *existing_elements)
        add_siblingship(G, 1, 2, *to_induce)

    return G


def create_graphs(graph_ids=(0, 1, 2, 3, 5, 6, 7, 8, 9, 10)):
    return tuple(create_graph(i) for i in graph_ids)


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

    for sender, receiver, edge_feature in G.edges(data=True):
        one_hot = np.zeros(len(all_edge_types), dtype=np.int)
        index_to_one_hot = all_edge_types.index(edge_feature[type_attribute])
        one_hot[index_to_one_hot] = 1
        G.edges[sender, receiver][attribute] = one_hot


if __name__ == "__main__":
    g = create_graph(12)
    experiment.plotting.plot_with_matplotlib(g)
