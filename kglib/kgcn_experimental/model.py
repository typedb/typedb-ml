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

import time

import networkx as nx
import tensorflow as tf
from graph_nets.demos import models

from kglib.kgcn_experimental.feed import create_feed_dict, create_placeholders
from kglib.kgcn_experimental.metrics import compute_accuracy
from kglib.kgcn_experimental.plotting import plot_input_vs_output, plot_across_training
from kglib.kgcn_experimental.prepare import make_all_runnable_in_session, create_input_target_graphs, \
    duplicate_edges_in_reverse


def create_loss_ops(target_op, output_ops):
    loss_ops = [
      tf.losses.softmax_cross_entropy(target_op.nodes, output_op.nodes) +
      tf.losses.softmax_cross_entropy(target_op.edges, output_op.edges)
      for output_op in output_ops
    ]
    return loss_ops


def model(concept_graphs,
          all_node_types,
          all_edge_types,
          tr_ge_split,
          num_processing_steps_tr=10,
          num_processing_steps_ge=10,
          num_training_iterations=10000,
          log_every_seconds=2):
    """
    Args:
        concept_graphs: In-memory graphs of Grakn concepts
        all_node_types: All of the node types present in the `concept_graphs`
        all_edge_types: All of the edge types present in the `concept_graphs`
        tr_ge_split: Integer at which to split the graphs between training and generalisation
        num_processing_steps_tr: Number of processing (message-passing) steps for training.
        num_processing_steps_ge: Number of processing (message-passing) steps for generalization.
        num_training_iterations: Number of training iterations
        log_every_seconds: # How much time between logging and printing the current results.

    Returns:

    """
    tf.reset_default_graph()

    def prepare(graph):
        graph = nx.convert_node_labels_to_integers(graph, label_attribute='concept')
        duplicate_edges_in_reverse(graph)
        return graph

    raw_graphs = [prepare(graph) for graph in concept_graphs]

    input_graphs, target_graphs = create_input_target_graphs(raw_graphs, all_node_types, all_edge_types)
    input_ph, target_ph = create_placeholders(input_graphs, target_graphs)

    # Connect the data to the model.
    # Instantiate the model.
    model = models.EncodeProcessDecode(edge_output_size=2, node_output_size=2)
    # A list of outputs, one per processing step.
    output_ops_tr = model(input_ph, num_processing_steps_tr)
    output_ops_ge = model(input_ph, num_processing_steps_ge)

    # Training loss.
    loss_ops_tr = create_loss_ops(target_ph, output_ops_tr)
    # Loss across processing steps.
    loss_op_tr = sum(loss_ops_tr) / num_processing_steps_tr
    # Test/generalization loss.
    loss_ops_ge = create_loss_ops(target_ph, output_ops_ge)
    loss_op_ge = loss_ops_ge[-1]  # Loss from final processing step.

    # Optimizer.
    learning_rate = 1e-3
    optimizer = tf.train.AdamOptimizer(learning_rate)
    step_op = optimizer.minimize(loss_op_tr)

    # Lets an iterable of TF graphs be output from a session as NP graphs.
    input_ph, target_ph = make_all_runnable_in_session(input_ph, target_ph)

    # Reset the Tensorflow session, but keep the same computational graph.

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    last_iteration = 0
    logged_iterations = []
    losses_tr = []
    corrects_tr = []
    solveds_tr = []
    losses_ge = []
    corrects_ge = []
    solveds_ge = []

    print("# (iteration number), T (elapsed seconds), "
          "Ltr (training loss), Lge (test/generalization loss), "
          "Ctr (training fraction nodes/edges labeled correctly), "
          "Str (training fraction examples solved correctly), "
          "Cge (test/generalization fraction nodes/edges labeled correctly), "
          "Sge (test/generalization fraction examples solved correctly)")

    start_time = time.time()
    last_log_time = start_time
    for iteration in range(last_iteration, num_training_iterations):
        feed_dict, _ = create_feed_dict("tr", tr_ge_split, input_ph, target_ph, input_graphs, target_graphs, raw_graphs)
        train_values = sess.run(
            {
                "step": step_op,
                "target": target_ph,
                "loss": loss_op_tr,
                "outputs": output_ops_tr
            },
            feed_dict=feed_dict)
        the_time = time.time()
        elapsed_since_last_log = the_time - last_log_time
        if elapsed_since_last_log > log_every_seconds:
            last_log_time = the_time
            feed_dict, ge_raw_graphs = create_feed_dict("ge", tr_ge_split, input_ph, target_ph, input_graphs, target_graphs, raw_graphs)
            test_values = sess.run(
                {
                    "target": target_ph,
                    "loss": loss_op_ge,
                    "outputs": output_ops_ge
                },
                feed_dict=feed_dict)
            correct_tr, solved_tr = compute_accuracy(
                train_values["target"], train_values["outputs"][-1], use_edges=True)
            correct_ge, solved_ge = compute_accuracy(
                test_values["target"], test_values["outputs"][-1], use_edges=True)
            elapsed = time.time() - start_time
            losses_tr.append(train_values["loss"])
            corrects_tr.append(correct_tr)
            solveds_tr.append(solved_tr)
            losses_ge.append(test_values["loss"])
            corrects_ge.append(correct_ge)
            solveds_ge.append(solved_ge)
            logged_iterations.append(iteration)
            print("# {:05d}, T {:.1f}, Ltr {:.4f}, Lge {:.4f}, Ctr {:.4f}, Str"
                  " {:.4f}, Cge {:.4f}, Sge {:.4f}".format(
                      iteration, elapsed, train_values["loss"], test_values["loss"],
                      correct_tr, solved_tr, correct_ge, solved_ge))

    plot_across_training(logged_iterations, losses_tr, losses_ge, corrects_tr, corrects_ge, solveds_tr, solveds_ge)
    plot_input_vs_output(ge_raw_graphs, test_values, num_processing_steps_ge)
