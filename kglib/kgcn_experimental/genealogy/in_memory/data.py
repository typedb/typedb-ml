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

from kglib.kgcn_experimental.plotting import plot_with_matplotlib


def generate_graph(num_people: int) -> nx.MultiDiGraph:
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

    G = nx.MultiDiGraph()

    G.add_nodes_from(range(num_people), type="person")

    last_id = G.number_of_nodes()
    for i, (node_1, node_2) in enumerate(itertools.permutations(range(num_people), 2)):
        parentship_node = i + last_id
        G.add_node(parentship_node, type="parentship")
        G.add_edge(parentship_node, node_1, type="parent")
        G.add_edge(parentship_node, node_2, type="child")

        # Also add directed edges going in the opposite direction
        G.add_edge(node_1, parentship_node, type="parent")
        G.add_edge(node_2, parentship_node, type="child")

    last_id = G.number_of_nodes()

    for i, (node_1, node_2) in enumerate(itertools.combinations(range(num_people), 2)):
        siblingship_node = i + last_id
        G.add_node(siblingship_node, type="siblingship")
        G.add_edge(siblingship_node, node_1, type="sibling")
        G.add_edge(siblingship_node, node_2, type="sibling")

        # Also add directed edges going in the opposite direction
        G.add_edge(node_1, siblingship_node, type="sibling")
        G.add_edge(node_2, siblingship_node, type="sibling")

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
                and G.edges[node, roleplayers[0], 0]['type'] == roles[0] \
                and G.edges[node, roleplayers[1], 0]['type'] == roles[1]:
            return node


def add_relation_label(G, relation_type, roles, roleplayers, *attributes):
    relation = find_relation_node(G, relation_type, roles, roleplayers)
    for attribute in attributes:
        G.nodes[relation][attribute] = 1
        G.edges[relation, roleplayers[0], 0][attribute] = 1
        G.edges[relation, roleplayers[1], 0][attribute] = 1

        # Also label directed edges going in the opposite direction
        G.edges[roleplayers[0], relation, 0][attribute] = 1
        G.edges[roleplayers[1], relation, 0][attribute] = 1


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

    # elif i == ?:
        # # Infers a parentship
        # G = base_graph(num_people)
        # add_parentship(G, 0, 1, *existing_elements)
        # add_parentship(G, 0, 2, *to_induce)
        # add_siblingship(G, 1, 2, *existing_elements)

    # ---- 4-person graphs ----
    elif i == 4:
        G = base_graph(4)
        add_parentship(G, 0, 1, *existing_elements)
        add_parentship(G, 0, 2, *existing_elements)
        add_parentship(G, 2, 3, *existing_elements)
        add_siblingship(G, 1, 2, *to_induce)

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
        add_parentship(G, 3, 0, *existing_elements)
        add_siblingship(G, 1, 2, *to_induce)

    elif i == 7:
        G = base_graph(4)
        add_parentship(G, 0, 1, *existing_elements)
        add_parentship(G, 1, 2, *existing_elements)
        add_parentship(G, 2, 3, *existing_elements)

    elif i == 8:
        # 1 parent to 3 siblings
        G = base_graph(4)
        add_parentship(G, 0, 1, *existing_elements)
        add_parentship(G, 0, 2, *existing_elements)
        add_parentship(G, 0, 3, *existing_elements)
        add_siblingship(G, 1, 2, *to_induce)
        add_siblingship(G, 2, 3, *to_induce)
        add_siblingship(G, 1, 3, *to_induce)

    elif i == 9:
        # All mutual siblings, no parents
        G = base_graph(4)
        add_siblingship(G, 0, 1, *existing_elements)
        add_siblingship(G, 1, 2, *existing_elements)
        add_siblingship(G, 2, 3, *existing_elements)
        add_siblingship(G, 0, 2, *to_induce)
        add_siblingship(G, 0, 3, *to_induce)
        add_siblingship(G, 1, 3, *to_induce)

    elif i == 10:
        G = base_graph(4)
        add_siblingship(G, 0, 1, *existing_elements)
        add_siblingship(G, 1, 2, *existing_elements)
        add_siblingship(G, 2, 3, *existing_elements)
        add_siblingship(G, 0, 2, *to_induce)
        add_siblingship(G, 0, 3, *to_induce)
        add_siblingship(G, 1, 3, *to_induce)

    elif i == 11:
        G = base_graph(4)
        add_parentship(G, 0, 1, *existing_elements)
        add_parentship(G, 0, 2, *existing_elements)
        add_parentship(G, 3, 1, *existing_elements)
        add_siblingship(G, 1, 2, *to_induce)

    return G


if __name__ == "__main__":
    g = create_graph(12)
    plot_with_matplotlib(g)
