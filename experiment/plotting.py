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

import matplotlib.pyplot as plt
import networkx as nx


def plot_with_matplotlib(G):

    attribute = 'solution'
    edges = []
    for edge in G.edges:
        if G.edges[edge][attribute] == 1:
            edges.append(edge)

    nodes = []
    for node in G.nodes:
        if G.nodes[node][attribute] == 1:
            nodes.append(node)

    labels_dict = {node_id: G.nodes[node_id]['type'] for node_id in nodes}
    edge_labels_dict = {(edge_id[0], edge_id[1]): G.edges[edge_id]['type'] for edge_id in edges}
    # pos = nx.spring_layout(G, k=6 / math.sqrt(G.number_of_nodes()))
    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos,
                           nodelist=nodes,
                           cmap=plt.get_cmap('jet'),
                           # node_color=values,
                           node_size=500
                           )
    nx.draw_networkx_labels(G, pos, labels=labels_dict)
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='r', arrows=True, arrowsize=30)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels_dict)
    plt.show()
