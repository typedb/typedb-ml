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

import math

import graph_nets.utils_np as utils_np
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import kglib.kgcn_tensorflow.plot.draw as custom_nx


def plot_across_training(logged_iterations, losses_tr, losses_ge, corrects_tr, corrects_ge, solveds_tr, solveds_ge,
                         output_file='./learning.png'):
    # Plot results curves.
    fig = plt.figure(1, figsize=(18, 3))
    fig.clf()
    x = np.array(logged_iterations)
    # Loss.
    y_tr = losses_tr
    y_ge = losses_ge
    ax = fig.add_subplot(1, 3, 1)
    ax.plot(x, y_tr, "k", label="Training")
    ax.plot(x, y_ge, "k--", label="Test/generalization")
    ax.set_title("Loss across training")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Loss (binary cross-entropy)")
    ax.legend()
    # Correct.
    y_tr = corrects_tr
    y_ge = corrects_ge
    ax = fig.add_subplot(1, 3, 2)
    ax.plot(x, y_tr, "k", label="Training")
    ax.plot(x, y_ge, "k--", label="Test/generalization")
    ax.set_title("Fraction correct across training")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Fraction nodes/edges correct")
    # Solved.
    y_tr = solveds_tr
    y_ge = solveds_ge
    ax = fig.add_subplot(1, 3, 3)
    ax.plot(x, y_tr, "k", label="Training")
    ax.plot(x, y_ge, "k--", label="Test/generalization")
    ax.set_title("Fraction solved across training")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Fraction examples solved")

    plt.savefig(output_file, bbox_inches='tight')


def plot_predictions(raw_graphs, test_values, num_processing_steps_ge, solution_weights=(-0.5, 0.5, 0.5),
                     output_file='./graph.png'):

    # # Plot graphs and results after each processing step.
    # The white node is the start, and the black is the end. Other nodes are colored
    # from red to purple to blue, where red means the model is confident the node is
    # off the shortest path, blue means the model is confident the node is on the
    # shortest path, and purplish colors mean the model isn't sure.

    max_graphs_to_plot = 10
    num_steps_to_plot = 3
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
    w = num_steps_to_plot + 2
    fig = plt.figure(101, figsize=(18, h * 3))
    fig.clf()
    for j, (graph, target, output) in enumerate(zip(raw_graphs, targets, outputs)):
        if j >= h:
            break
        for s, r, d in graph.edges(data=True):
            d['weight'] = solution_weights[d['solution']]  # Looks good with high k
        pos = nx.spring_layout(graph, k=3 / math.sqrt(graph.number_of_nodes()), seed=1, weight='weight', iterations=50)
        # pos = nx.circular_layout(graph, scale=2)
        ground_truth_node_prob = target["nodes"][:, -1]
        ground_truth_edge_prob = target["edges"][:, -1]

        non_preexist_node_mask = mask_preexists(target["nodes"])
        non_preexist_edge_mask = mask_preexists(target["edges"])

        # Ground truth.
        iax = j * (2 + num_steps_to_plot) + 1
        ax = draw_subplot(graph, fig, pos, node_size, h, w, iax, ground_truth_node_prob, ground_truth_edge_prob, True)

        # Format the ground truth plot axes
        ax.set_axis_on()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['bottom'].set_color('blue')
        ax.spines['top'].set_color('blue')
        ax.spines['right'].set_color('blue')
        ax.spines['left'].set_color('blue')
        ax.grid(None)
        ax.set_title("Ground truth")

        # Prediction.
        for k, outp in enumerate(output):
            iax = j * (2 + num_steps_to_plot) + 2 + k
            node_prob = softmax_prob_last_dim(outp["nodes"]) * non_preexist_node_mask
            edge_prob = softmax_prob_last_dim(outp["edges"]) * non_preexist_edge_mask
            ax = draw_subplot(graph, fig, pos, node_size, h, w, iax, node_prob, edge_prob, False)
            ax.set_title("Model-predicted\nStep {:02d} / {:02d}".format(
                step_indices[k] + 1, step_indices[-1] + 1))

        # Class Winners
        # Displays whether the class represented by the last dimension was the winner
        node_prob = last_dim_was_class_winner(output[-1]["nodes"]) * non_preexist_node_mask
        edge_prob = last_dim_was_class_winner(output[-1]["edges"]) * non_preexist_edge_mask

        iax = j * (2 + num_steps_to_plot) + 2 + len(output)
        ax = draw_subplot(graph, fig, pos, node_size, h, w, iax, node_prob, edge_prob, False)

        # Format the class winners plot axes
        ax.set_axis_on()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['bottom'].set_color('green')
        ax.spines['top'].set_color('green')
        ax.spines['right'].set_color('green')
        ax.spines['left'].set_color('green')
        ax.grid(None)
        ax.set_title("Model-predicted winners")

    plt.savefig(output_file, bbox_inches='tight')


def mask_preexists(arr):
    return (arr[:, 0] == 0) * 1


def softmax_prob_last_dim(x):
    e = np.exp(x)
    return e[:, -1] / np.sum(e, axis=-1)


def last_dim_was_class_winner(x):
    return (np.argmax(x, axis=-1) == 2) * 1


def element_color(gt_plot, probability, element_props):
    """
    Determine the color values to use for a node/edge and its label
    gt plot:
    blue for existing elements, green for those to infer, red candidates

    output plot:
    blue for existing elements, green for those to infer, red for candidates, all with transparency
    """

    existing = 0
    candidate = 1
    to_infer = 2

    solution = element_props.get('solution')

    color_config = {
        to_infer: {'color': [0.0, 1.0, 0.0], 'gt_opacity': 1.0},
        candidate: {'color': [1.0, 0.0, 0.0], 'gt_opacity': 1.0},
        existing: {'color': [0.0, 0.0, 1.0], 'gt_opacity': 0.2}
    }

    chosen_config = color_config[solution]

    if gt_plot:
        opacity = chosen_config['gt_opacity']
    else:
        opacity = probability

    label = np.array([0.0, 0.0, 0.0] + [opacity])
    color = np.array(chosen_config['color'] + [opacity])

    return dict(element=color, label=label)


def draw_subplot(graph, fig, pos, node_size, h, w, iax, node_prob, edge_prob, gt_plot):
    ax = fig.add_subplot(h, w, iax)
    node_color = {}
    node_label_color = {}
    edge_color = {}
    edge_label_color = {}

    for i, (n, props) in enumerate(graph.nodes(data=True)):
        colors = element_color(gt_plot, node_prob[n], props)

        node_color[n] = colors['element']
        node_label_color[n] = colors['label']

    for n, (sender, receiver, props) in enumerate(graph.edges(data=True)):
        colors = element_color(gt_plot, edge_prob[n], props)

        edge_color[(sender, receiver)] = colors['element']
        edge_label_color[(sender, receiver)] = colors['label']

    draw_graph(graph, pos, ax, node_size=node_size, node_color=node_color, node_label_color=node_label_color,
               edge_color=edge_color, edge_label_color=edge_label_color)
    return ax


def draw_graph(graph,
               pos,
               ax,
               node_size=200,
               node_color=(0.4, 0.8, 0.4),
               node_label_color=None,
               edge_color=(0.0, 0.0, 0.0),
               edge_label_color=None,
               node_linewidth=1.0,
               edge_width=1.0,
               font_size=6):

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
          zorder=-10)

    # Plot edges.
    e = [edge_color[(s, r)] for s, r, k in graph.edges]
    _draw(nx.draw_networkx_edges,
          edgelist=graph.edges,
          width=edge_width,
          zorder=-20,
          edge_color=e
          )

    bbox_props = dict(boxstyle="square,pad=0.0", fc="none", ec="none", lw=1)
    labels_dict = {node_id: graph.nodes[node_id]['type'] for node_id in graph.nodes}
    edge_labels_dict = {(edge_id[0], edge_id[1]): graph.edges[edge_id]['type'] for edge_id in graph.edges}

    custom_nx.draw_networkx_labels(graph,
                                   pos,
                                   labels=labels_dict,
                                   font_size=font_size,
                                   font_color=node_label_color,
                                   alpha=[node_label_color[n][-1] for n in graph.nodes()])

    custom_nx.draw_networkx_edge_labels(graph,
                                        pos,
                                        edge_labels=edge_labels_dict,
                                        font_size=font_size,
                                        # font_color=np.array([0.0, 0.5, 0.0, 0.1]),
                                        font_color=edge_label_color,
                                        # alpha=0.2,
                                        alpha={n: edge_label_color[n][-1] for n in graph.edges()},
                                        bbox=bbox_props)
