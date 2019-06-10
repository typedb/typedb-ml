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
    min_c = 0.3
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
    ncs = []
    for j, (graph, target, output) in enumerate(zip(raw_graphs, targets, outputs)):
        if j >= h:
            break
        # pos = get_node_dict(graph, "pos")
        pos = nx.circular_layout(graph)
        ground_truth = target["nodes"][:, -1]
        # Ground truth.
        iax = j * (1 + num_steps_to_plot) + 1
        ax = fig.add_subplot(h, w, iax)
        plotter = GraphPlotter(ax, graph, pos)
        color = {}
        for i, n in enumerate(plotter.nodes):
            color[n] = np.array([1.0 - ground_truth[i], 0.0, ground_truth[i], 1.0
                                 ]) * (1.0 - min_c) + min_c
        plotter.draw_graph_with_solution(node_size=node_size, node_color=color)
        ax.set_axis_on()
        ax.set_xticks([])
        ax.set_yticks([])
        try:
            ax.set_facecolor([0.9] * 3 + [1.0])
        except AttributeError:
            ax.set_axis_bgcolor([0.9] * 3 + [1.0])
        ax.grid(None)
        ax.set_title("Ground truth\nSolution length: {}".format(
            plotter.solution_length))
        # Prediction.
        for k, outp in enumerate(output):
            iax = j * (1 + num_steps_to_plot) + 2 + k
            ax = fig.add_subplot(h, w, iax)
            plotter = GraphPlotter(ax, graph, pos)
            color = {}
            prob = softmax_prob_last_dim(outp["nodes"])
            for i, n in enumerate(plotter.nodes):
                color[n] = np.array([1.0 - prob[n], 0.0, prob[n], 1.0
                                     ]) * (1.0 - min_c) + min_c
            plotter.draw_graph_with_solution(node_size=node_size, node_color=color)
            ax.set_title("Model-predicted\nStep {:02d} / {:02d}".format(
                step_indices[k] + 1, step_indices[-1] + 1))


class GraphPlotter(object):

    def __init__(self, ax, graph, pos):
        self._ax = ax
        self._graph = graph
        self._pos = pos
        self._base_draw_kwargs = dict(G=self._graph, pos=self._pos, ax=self._ax)
        self._solution_length = None
        self._nodes = None
        self._edges = None
        self._start_nodes = None
        self._end_nodes = None
        self._solution_nodes = None
        self._intermediate_solution_nodes = None
        self._solution_edges = None
        self._non_solution_nodes = None
        self._non_solution_edges = None
        self._ax.set_axis_off()

    @property
    def solution_length(self):
        if self._solution_length is None:
            self._solution_length = len(self._solution_edges)
        return self._solution_length

    @property
    def nodes(self):
        if self._nodes is None:
            self._nodes = self._graph.nodes()
        return self._nodes

    @property
    def edges(self):
        if self._edges is None:
            self._edges = self._graph.edges()
        return self._edges

    @property
    def start_nodes(self):
        if self._start_nodes is None:
            self._start_nodes = [
                n for n in self.nodes if self._graph.node[n].get("start", False)
            ]
        return self._start_nodes

    @property
    def end_nodes(self):
        if self._end_nodes is None:
            self._end_nodes = [
                n for n in self.nodes if self._graph.node[n].get("end", False)
            ]
        return self._end_nodes

    @property
    def solution_nodes(self):
        if self._solution_nodes is None:
            self._solution_nodes = [
                n for n in self.nodes if self._graph.node[n].get("solution", False)
            ]
        return self._solution_nodes

    @property
    def intermediate_solution_nodes(self):
        if self._intermediate_solution_nodes is None:
            self._intermediate_solution_nodes = [
                n for n in self.nodes
                if self._graph.node[n].get("solution", False) and
                   not self._graph.node[n].get("start", False) and
                   not self._graph.node[n].get("end", False)
            ]
        return self._intermediate_solution_nodes

    @property
    def solution_edges(self):
        if self._solution_edges is None:
            self._solution_edges = [
                e for e in self.edges
                if self._graph.get_edge_data(e[0], e[1]).get("solution", False)
            ]
        return self._solution_edges

    @property
    def non_solution_nodes(self):
        if self._non_solution_nodes is None:
            self._non_solution_nodes = [
                n for n in self.nodes
                if not self._graph.node[n].get("solution", False)
            ]
        return self._non_solution_nodes

    @property
    def non_solution_edges(self):
        if self._non_solution_edges is None:
            self._non_solution_edges = [
                e for e in self.edges
                if not self._graph.get_edge_data(e[0], e[1]).get("solution", False)
            ]
        return self._non_solution_edges

    def _make_draw_kwargs(self, **kwargs):
        kwargs.update(self._base_draw_kwargs)
        return kwargs

    def _draw(self, draw_function, zorder=None, **kwargs):
        draw_kwargs = self._make_draw_kwargs(**kwargs)
        collection = draw_function(**draw_kwargs)
        if collection is not None and zorder is not None:
            try:
                # This is for compatibility with older matplotlib.
                collection.set_zorder(zorder)
            except AttributeError:
                # This is for compatibility with newer matplotlib.
                collection[0].set_zorder(zorder)
        return collection

    def draw_nodes(self, **kwargs):
        """Useful kwargs: nodelist, node_size, node_color, linewidths."""
        if ("node_color" in kwargs and
                isinstance(kwargs["node_color"], collections.Sequence) and
                len(kwargs["node_color"]) in {3, 4} and
                not isinstance(kwargs["node_color"][0],
                               (collections.Sequence, np.ndarray))):
            num_nodes = len(kwargs.get("nodelist", self.nodes))
            kwargs["node_color"] = np.tile(
                np.array(kwargs["node_color"])[None], [num_nodes, 1])
        return self._draw(nx.draw_networkx_nodes, **kwargs)

    def draw_edges(self, **kwargs):
        """Useful kwargs: edgelist, width."""
        return self._draw(nx.draw_networkx_edges, **kwargs)

    def draw_graph(self,
                   node_size=200,
                   node_color=(0.4, 0.8, 0.4),
                   node_linewidth=1.0,
                   edge_width=1.0):
        # Plot nodes.
        self.draw_nodes(
            nodelist=self.nodes,
            node_size=node_size,
            node_color=node_color,
            linewidths=node_linewidth,
            zorder=20)
        # Plot edges.
        self.draw_edges(edgelist=self.edges, width=edge_width, zorder=10)

    def draw_graph_with_solution(self,
                                 node_size=200,
                                 node_color=(0.4, 0.8, 0.4),
                                 node_linewidth=1.0,
                                 edge_width=1.0,
                                 start_color="w",
                                 end_color="k",
                                 solution_node_linewidth=3.0,
                                 solution_edge_width=3.0):
        node_border_color = (0.0, 0.0, 0.0, 1.0)
        node_collections = {}
        # Plot start nodes.
        node_collections["start nodes"] = self.draw_nodes(
            nodelist=self.start_nodes,
            node_size=node_size,
            node_color=start_color,
            linewidths=solution_node_linewidth,
            edgecolors=node_border_color,
            zorder=100)
        # Plot end nodes.
        node_collections["end nodes"] = self.draw_nodes(
            nodelist=self.end_nodes,
            node_size=node_size,
            node_color=end_color,
            linewidths=solution_node_linewidth,
            edgecolors=node_border_color,
            zorder=90)
        # Plot intermediate solution nodes.
        if isinstance(node_color, dict):
            c = [node_color[n] for n in self.intermediate_solution_nodes]
        else:
            c = node_color
        node_collections["intermediate solution nodes"] = self.draw_nodes(
            nodelist=self.intermediate_solution_nodes,
            node_size=node_size,
            node_color=c,
            linewidths=solution_node_linewidth,
            edgecolors=node_border_color,
            zorder=80)
        # Plot solution edges.
        node_collections["solution edges"] = self.draw_edges(
            edgelist=self.solution_edges, width=solution_edge_width, zorder=70)
        # Plot non-solution nodes.
        if isinstance(node_color, dict):
            c = [node_color[n] for n in self.non_solution_nodes]
        else:
            c = node_color
        node_collections["non-solution nodes"] = self.draw_nodes(
            nodelist=self.non_solution_nodes,
            node_size=node_size,
            node_color=c,
            linewidths=node_linewidth,
            edgecolors=node_border_color,
            zorder=20)
        # Plot non-solution edges.
        node_collections["non-solution edges"] = self.draw_edges(
            edgelist=self.non_solution_edges, width=edge_width, zorder=10)
        # Set title as solution length.
        self._ax.set_title("Solution length: {}".format(self.solution_length))
        return node_collections
