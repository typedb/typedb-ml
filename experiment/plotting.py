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
import collections

import graph_nets.utils_np as utils_np
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


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


def get_node_dict(graph, attr):
    """Return a `dict` of node:attribute pairs from a graph."""
    return {k: v[attr] for k, v in graph.node.items()}


def softmax_prob_last_dim(x):  # pylint: disable=redefined-outer-name
    e = np.exp(x)
    return e[:, -1] / np.sum(e, axis=-1)
    # return x[:, 0]


def above_base(val, base=0.0):
    return val * (1.0 - base) + base


def colors_array(probability):
    """
    Determine the color values to use for a node/edge
    :param probability: the score for the node
    :return: array of rgba color values to use
    """
    return np.array([above_base(1.0 - probability), 0.0, above_base(probability), above_base(probability, base=0.1)])


def draw_subplot(graph, fig, pos, node_size, h, w, iax, node_prob, edge_prob):
    ax = fig.add_subplot(h, w, iax)
    node_color = {}
    edge_color = {}

    # Draw the nodes
    for i, n in enumerate(graph.nodes):
        node_color[n] = colors_array(node_prob[n])

    # Draw the edges
    for n, (sender, receiver) in enumerate(graph.edges):
            edge_color[(sender, receiver)] = colors_array(edge_prob[n])
    draw_graph(graph, pos, ax, node_size=node_size, node_color=node_color, edge_color=edge_color)
    return ax


def plot_input_vs_output(raw_graphs,
                         test_values,
                         num_processing_steps_ge):

    # # Plot graphs and results after each processing step.
    # The white node is the start, and the black is the end. Other nodes are colored
    # from red to purple to blue, where red means the model is confident the node is
    # off the shortest path, blue means the model is confident the node is on the
    # shortest path, and purplish colors mean the model isn't sure.

    max_graphs_to_plot = 6
    num_steps_to_plot = 4
    node_size = 120

    num_graphs = len(raw_graphs)
    targets = utils_np.graphs_tuple_to_data_dicts(test_values["target"])
    step_indices = np.floor(
        np.linspace(0, num_processing_steps_ge - 1,
                    num_steps_to_plot)).astype(int).tolist()
    outputs = list(
        zip(*(utils_np.graphs_tuple_to_data_dicts(test_values["outputs"][i])
              for i in step_indices)))
    h = min(num_graphs, max_graphs_to_plot)
    w = num_steps_to_plot + 1
    fig = plt.figure(101, figsize=(18, h * 3))
    fig.clf()
    for j, (graph, target, output) in enumerate(zip(raw_graphs, targets, outputs)):
        if j >= h:
            break
        pos = nx.circular_layout(graph)
        ground_truth_node_prob = target["nodes"][:, -1]
        ground_truth_edge_prob = target["edges"][:, -1]

        # Ground truth.
        iax = j * (1 + num_steps_to_plot) + 1
        ax = draw_subplot(graph, fig, pos, node_size, h, w, iax, ground_truth_node_prob, ground_truth_edge_prob)

        ax.set_axis_on()
        ax.set_xticks([])
        ax.set_yticks([])
        try:
            ax.set_facecolor([0.9] * 3 + [1.0])
        except AttributeError:
            ax.set_axis_bgcolor([0.9] * 3 + [1.0])
        ax.grid(None)
        ax.set_title("Ground truth")

        # Prediction.
        for k, outp in enumerate(output):
            iax = j * (1 + num_steps_to_plot) + 2 + k
            node_prob = softmax_prob_last_dim(outp["nodes"])
            edge_prob = softmax_prob_last_dim(outp["edges"])
            ax = draw_subplot(graph, fig, pos, node_size, h, w, iax, node_prob, edge_prob)
            ax.set_title("Model-predicted\nStep {:02d} / {:02d}".format(
                step_indices[k] + 1, step_indices[-1] + 1))


def draw_graph(graph,
               pos,
               ax,
               node_size=200,
               node_color=(0.4, 0.8, 0.4),
               edge_color=(0.0, 0.0, 0.0),
               node_linewidth=1.0,
               edge_width=1.0):

    def _draw(draw_function, zorder=None, **kwargs):
        # draw_kwargs = self._make_draw_kwargs(**kwargs)
        _base_draw_kwargs = dict(G=graph, pos=pos, ax=ax)
        kwargs.update(_base_draw_kwargs)
        collection = draw_function(**kwargs)
        if collection is not None and zorder is not None:
            try:
                # This is for compatibility with older matplotlib.
                collection.set_zorder(zorder)
            except AttributeError:
                # This is for compatibility with newer matplotlib.
                collection[0].set_zorder(zorder)
        return collection

    # Plot nodes.
    c = [node_color[n] for n in graph.nodes()]
    _draw(nx.draw_networkx_nodes,
          node_size=node_size,
          node_color=c,
          linewidths=node_linewidth,
          alpha=[node_color[n][-1] for n in graph.nodes()],
          zorder=20)

    # Plot edges.
    e = [edge_color[(s, r)] for s, r in graph.edges]
    _draw(nx.draw_networkx_edges,
          edgelist=graph.edges,
          width=edge_width,
          zorder=10,
          edge_color=e
          )
