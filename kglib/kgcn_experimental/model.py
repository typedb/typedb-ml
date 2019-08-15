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

import numpy as np
import tensorflow as tf
from graph_nets.demos import models

from kglib.kgcn_experimental.feed import create_feed_dict, create_placeholders
from kglib.kgcn_experimental.metrics import compute_accuracy
from kglib.kgcn_experimental.plotting import plot_across_training, plot_input_vs_output
from kglib.kgcn_experimental.prepare import make_all_runnable_in_session, create_input_target_graphs


def loss_ops_from_difference(target_op, output_ops):
    """
    Loss operation which directly compares the target with the output over all nodes and edges
    Args:
        target_op: The target of the model
        output_ops: A list of the outputs of the model, one for each message-passing step

    Returns: The loss for each message-passing step

    """
    loss_ops = [
      tf.losses.softmax_cross_entropy(target_op.nodes, output_op.nodes) +
      tf.losses.softmax_cross_entropy(target_op.edges, output_op.edges)
      for output_op in output_ops
    ]
    return loss_ops


def loss_ops_preexisting_no_penalty(target_op, output_ops):
    """
    Loss operation which doesn't penalise the output values for pre-existing nodes and edges, treating them as slack
    variables

    Args:
        target_op: The target of the model
        output_ops: A list of the outputs of the model, one for each message-passing step

    Returns: The loss for each message-passing step

    """
    loss_ops = []
    for output_op in output_ops:

        node_mask_op = tf.math.reduce_any(tf.math.not_equal(target_op.nodes, tf.constant(np.array([0., 0., 1.]))), axis=1)
        target_nodes = tf.boolean_mask(target_op.nodes, node_mask_op)
        output_nodes = tf.boolean_mask(output_op.nodes, node_mask_op)

        edge_mask_op = tf.math.reduce_any(tf.math.not_equal(target_op.nodes, tf.constant(np.array([0., 0., 1.]))), axis=1)
        target_edges = tf.boolean_mask(target_op.nodes, edge_mask_op)
        output_edges = tf.boolean_mask(output_op.nodes, edge_mask_op)

        loss_op = (tf.losses.softmax_cross_entropy(target_nodes, output_nodes)
                   + tf.losses.softmax_cross_entropy(target_edges, output_edges))

        loss_ops.append(loss_op)

    return loss_ops


def softmax(x):
    return np.exp(x)/sum(np.exp(x))


def model(tr_graphs,
          ge_graphs,
          all_node_types,
          all_edge_types,
          num_processing_steps_tr=10,
          num_processing_steps_ge=10,
          num_training_iterations=10000,
          log_every_seconds=2):
    """
    Args:
        tr_graphs: In-memory graphs of Grakn concepts for training
        ge_graphs: In-memory graphs of Grakn concepts for generalisation
        all_node_types: All of the node types present in the `concept_graphs`, used for encoding
        all_edge_types: All of the edge types present in the `concept_graphs`, used for encoding
        num_processing_steps_tr: Number of processing (message-passing) steps for training.
        num_processing_steps_ge: Number of processing (message-passing) steps for generalization.
        num_training_iterations: Number of training iterations
        log_every_seconds: # How much time between logging and printing the current results.

    Returns:

    """
    tf.reset_default_graph()
    tf.set_random_seed(1)

    tr_input_graphs, tr_target_graphs = create_input_target_graphs(tr_graphs, all_node_types, all_edge_types)
    ge_input_graphs, ge_target_graphs = create_input_target_graphs(ge_graphs, all_node_types, all_edge_types)

    input_ph, target_ph = create_placeholders(tr_input_graphs, tr_target_graphs)

    # Connect the data to the model.
    # Instantiate the model.
    model = models.EncodeProcessDecode(edge_output_size=3, node_output_size=3)
    # A list of outputs, one per processing step.
    output_ops_tr = model(input_ph, num_processing_steps_tr)
    output_ops_ge = model(input_ph, num_processing_steps_ge)

    # Training loss.
    loss_ops_tr = loss_ops_from_difference(target_ph, output_ops_tr)
    # Loss across processing steps.
    loss_op_tr = sum(loss_ops_tr) / num_processing_steps_tr
    # Test/generalization loss.
    loss_ops_ge = loss_ops_from_difference(target_ph, output_ops_ge)
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
        feed_dict = create_feed_dict(input_ph, target_ph, tr_input_graphs, tr_target_graphs)
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
            feed_dict = create_feed_dict(input_ph, target_ph, ge_input_graphs, ge_target_graphs)
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
    plot_input_vs_output(ge_graphs, test_values, num_processing_steps_ge)

    return train_values, test_values
