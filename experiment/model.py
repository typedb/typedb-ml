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

import tensorflow as tf
from graph_nets import utils_tf
from graph_nets.demos import models

from experiment.feed import create_feed_dict, create_placeholders
from experiment.data import create_input_target_graphs, create_graph
from experiment.metrics import compute_accuracy
from experiment.plotting import plot_input_vs_output, plot_across_training


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


def main():
    tf.reset_default_graph()

    # Model parameters.
    # Number of processing (message-passing) steps.
    num_processing_steps_tr = 6
    num_processing_steps_ge = 6

    # Data / training parameters.
    num_training_iterations = 100

    # The value at which to split the data into training and evaluation sets
    tr_ge_split = 9

    # How much time between logging and printing the current results.
    log_every_seconds = 2

    # Data.
    # Input and target placeholders.

    # graph_ids = list(range(12))
    # random.seed(1)
    # random.shuffle(graph_ids)
    # print(f'Graphs are used in the order {graph_ids}')
    graph_ids = [7, 11, 0, 8, 5, 6, 3, 10, 4, 1, 9, 2]
    all_node_types = ['person', 'parentship', 'siblingship']
    all_edge_types = ['parent', 'child', 'sibling']
    raw_graphs = [create_graph(i) for i in graph_ids]
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


if __name__ == "__main__":
    main()
