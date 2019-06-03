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
    labels_dict = {node_id: G.nodes[node_id]['type'] for node_id in G.nodes}
    edge_labels_dict = {(edge_id[0], edge_id[1]): G.edges[edge_id]['type'] for edge_id in G.edges}
    # pos = nx.spring_layout(G, k=6 / math.sqrt(len(G.nodes)))
    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'),
                           # node_color=values,
                           node_size=500
                           )
    nx.draw_networkx_labels(G, pos, labels=labels_dict)
    nx.draw_networkx_edges(G, pos, edgelist=None, edge_color='r', arrows=True, arrowsize=30)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels_dict)
    plt.show()
