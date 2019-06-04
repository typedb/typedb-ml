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

from graph_nets import utils_np
from graph_nets import utils_tf
import numpy as np
import tensorflow as tf

import experiment.data as data


def to_one_hot(indices, max_value, axis=-1):
    one_hot = np.eye(max_value)[indices]
    if axis not in (-1, one_hot.ndim):
        one_hot = np.moveaxis(one_hot, -1, axis)
    return one_hot


def graph_to_input_target(graph):
    """Returns 2 graphs with input and target feature vectors for training.

    Args:
    graph: An `nx.DiGraph` instance.

    Returns:
    The input `nx.DiGraph` instance.
    The target `nx.DiGraph` instance.

    Raises:
    ValueError: unknown node type
    """

    def create_feature(attr, fields):
        return np.hstack([np.array(attr[field], dtype=float) for field in fields])

    input_node_fields = ("input", "one_hot_type")
    input_edge_fields = ("input", "one_hot_type")
    target_node_fields = ("solution",)
    target_edge_fields = ("solution",)

    input_graph = graph.copy()
    target_graph = graph.copy()

    for node_index, node_feature in graph.nodes(data=True):
        input_graph.add_node(node_index, features=create_feature(node_feature, input_node_fields))
        target_node = to_one_hot(create_feature(node_feature, target_node_fields).astype(int), 2)[0]
        target_graph.add_node(node_index, features=target_node)

    for receiver, sender, edge_features in graph.edges(data=True):
        input_graph.add_edge(receiver, sender, features=create_feature(edge_features, input_edge_fields))
        target_edge = to_one_hot(create_feature(edge_features, target_edge_fields).astype(int), 2)[0]
        target_graph.add_edge(receiver, sender, features=target_edge)

    input_graph.graph["features"] = np.array([0.0])
    target_graph.graph["features"] = np.array([0.0])

    return input_graph, target_graph


def create_graphs():
    """
    Builds graphs ready to be used for training
    :return: the input graphs, the target (desired output) graphs, and the original_graphs
    """
    # graph_ids = (0, 1, 2, 3, 5, 6, 7, 8, 9, 10)
    graph_ids = (6, 0, 8, 10, 3, 9, 2, 7, 1, 5,)
    all_node_types = ['person', 'parentship', 'siblingship']
    all_edge_types = ['parent', 'child', 'sibling']

    input_graphs = []
    target_graphs = []
    graphs = []
    for i in graph_ids:
        graph = data.create_graph(i)
        data.encode_types_one_hot(graph, all_node_types, all_edge_types, attribute='one_hot_type',
                                  type_attribute='type')

        input_graph, target_graph = graph_to_input_target(graph)
        input_graphs.append(input_graph)
        target_graphs.append(target_graph)
        graphs.append(graph)

    return input_graphs, target_graphs, graphs


def create_placeholders():
    """
    Creates placeholders for the model training and evaluation.
    Returns:
    input_ph: The input graph's placeholders, as a graph namedtuple.
    target_ph: The target graph's placeholders, as a graph namedtuple.
    """
    # Create some example data for inspecting the vector sizes.
    input_graphs, target_graphs, _ = create_graphs()
    input_ph = utils_tf.placeholders_from_networkxs(input_graphs)
    target_ph = utils_tf.placeholders_from_networkxs(target_graphs)
    return input_ph, target_ph


def create_feed_dict(input_ph, target_ph):
    """Creates the feed dict for the placeholders for the model training and evaluation.

    Args:
    input_ph: The input graph's placeholders, as a graph namedtuple.
    target_ph: The target graph's placeholders, as a graph namedtuple.

    Returns:
    feed_dict: The feed `dict` of input and target placeholders and data.
    raw_graphs: The `dict` of raw networkx graphs.
    """
    inputs, targets, raw_graphs = create_graphs()
    input_graphs = utils_np.networkxs_to_graphs_tuple(inputs)
    target_graphs = utils_np.networkxs_to_graphs_tuple(targets)
    feed_dict = {input_ph: input_graphs, target_ph: target_graphs}
    return feed_dict, raw_graphs


def compute_accuracy(target, output, use_nodes=True, use_edges=False):
    """Calculate model accuracy.

    Returns the number of elements correctly predicted to exist, and the number of completely correct graphs
    (100% correct predictions).

    Args:
        target: A `graphs.GraphsTuple` that contains the target graph.
        output: A `graphs.GraphsTuple` that contains the output graph.
        use_nodes: A `bool` indicator of whether to compute node accuracy or not.
        use_edges: A `bool` indicator of whether to compute edge accuracy or not.

    Returns:
        correct: A `float` fraction of correctly labeled nodes/edges.
        solved: A `float` fraction of graphs that are completely correctly labeled.

    Raises:
        ValueError: Nodes or edges (or both) must be used
    """
    if not use_nodes and not use_edges:
        raise ValueError("Nodes or edges (or both) must be used")
    tdds = utils_np.graphs_tuple_to_data_dicts(target)
    odds = utils_np.graphs_tuple_to_data_dicts(output)
    cs = []
    ss = []
    for td, od in zip(tdds, odds):
        xn = np.argmax(td["nodes"], axis=-1)
        yn = np.argmax(od["nodes"], axis=-1)
        xe = np.argmax(td["edges"], axis=-1)
        ye = np.argmax(od["edges"], axis=-1)
        c = []
        if use_nodes:
            c.append(xn == yn)
        if use_edges:
            c.append(xe == ye)
        c = np.concatenate(c, axis=0)
        s = np.all(c)
        cs.append(c)
        ss.append(s)
    correct = np.mean(np.concatenate(cs, axis=0))
    solved = np.mean(np.stack(ss))
    return correct, solved


def create_loss_ops(target_op, output_ops):
    loss_ops = [
      tf.losses.softmax_cross_entropy(target_op.nodes, output_op.nodes) +
      tf.losses.softmax_cross_entropy(target_op.edges, output_op.edges)
      for output_op in output_ops
    ]
    return loss_ops


def make_all_runnable_in_session(*args):
    """Lets an iterable of TF graphs be output from a session as NP graphs."""
    return [utils_tf.make_runnable_in_session(a) for a in args]
