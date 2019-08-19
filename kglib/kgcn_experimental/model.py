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
import sonnet as snt
import tensorflow as tf
from graph_nets import modules
from graph_nets import utils_tf
from tensorflow.python.layers.core import dense

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
        node_mask_op = tf.math.reduce_any(tf.math.not_equal(target_op.nodes, tf.constant(np.array([0., 0., 1.]))),
                                          axis=1)
        target_nodes = tf.boolean_mask(target_op.nodes, node_mask_op)
        output_nodes = tf.boolean_mask(output_op.nodes, node_mask_op)

        edge_mask_op = tf.math.reduce_any(tf.math.not_equal(target_op.nodes, tf.constant(np.array([0., 0., 1.]))),
                                          axis=1)
        target_edges = tf.boolean_mask(target_op.nodes, edge_mask_op)
        output_edges = tf.boolean_mask(output_op.nodes, edge_mask_op)

        loss_op = (tf.losses.softmax_cross_entropy(target_nodes, output_nodes)
                   + tf.losses.softmax_cross_entropy(target_edges, output_edges))

        loss_ops.append(loss_op)

    return loss_ops


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


def model(tr_graphs,
          ge_graphs,
          all_node_types,
          all_edge_types,
          num_processing_steps_tr=10,
          num_processing_steps_ge=10,
          num_training_iterations=10000,
          log_every_seconds=2,
          log_dir=None):
    """
    Args:
        tr_graphs: In-memory graphs of Grakn concepts for training
        ge_graphs: In-memory graphs of Grakn concepts for generalisation
        all_node_types: All of the node types present in the `concept_graphs`, used for encoding
        all_edge_types: All of the edge types present in the `concept_graphs`, used for encoding
        num_processing_steps_tr: Number of processing (message-passing) steps for training.
        num_processing_steps_ge: Number of processing (message-passing) steps for generalization.
        num_training_iterations: Number of training iterations
        log_every_seconds: The time to wait between logging and printing the next set of results.
        log_dir: Directory to store TensorFlow events files

    Returns:

    """
    tf.reset_default_graph()
    tf.set_random_seed(1)

    tr_input_graphs, tr_target_graphs = create_input_target_graphs(tr_graphs, all_node_types, all_edge_types)
    ge_input_graphs, ge_target_graphs = create_input_target_graphs(ge_graphs, all_node_types, all_edge_types)

    input_ph, target_ph = create_placeholders(tr_input_graphs, tr_target_graphs)

    # Connect the data to the model.
    # Instantiate the model.
    model = KGMessagePassingLearner(edge_output_size=3, node_output_size=3)
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

    if log_dir is not None:
        tf.summary.FileWriter(log_dir, sess.graph)

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


NUM_LAYERS = 2  # Hard-code number of layers in the edge/node/global models.
LATENT_SIZE = 16


class ThingModelManager(snt.AbstractModule):
    """
    Manages the models corresponding to different types of Thing in the Knowledge Graph. Assumes that the type of a
    Thing is given categorically as an integer value in the 0th position of its features
    """
    def __init__(self, encoders, feature_length, name="thing_model_manager"):
        """
        Args:
            encoders: List of encoders, one for each of the type indices that may be present. Expected to have a small
                set of instances reused throughout the list.
            feature_length: The length of features to output for matrix initialisation
            name: The name for this Module
        """
        super(ThingModelManager, self).__init__(name=name)
        self._feature_length = feature_length
        self._encoders = encoders

    def _build(self, things):

        encoded_things = np.zeros([things.shape[0], self._feature_length], dtype=np.float64)

        for i, thing in enumerate(things):
            encoder = self._encoders[int(thing[0])]  # Get the appropriate encoder for this type category
            encoded_things[i, :] = encoder(thing)

        return things


class KGIndependent(snt.AbstractModule):
    """
    A graph model that applies independent models to graph elements according to the types of those elements

    The inputs and outputs are graphs. The corresponding models are applied to
    each element of the graph (edges, nodes and globals) in parallel and
    independently of the other elements. It can be used to encode or
    decode the elements of a knowledge graph.
    """

    def __init__(self,
                 edge_model_fns=None,
                 node_model_fns=None,
                 global_model_fn=None,
                 name="kg_independent"):
        super(KGIndependent, self).__init__(name=name)

        with self._enter_variable_scope():
            # The use of snt.Module below is to ensure the ops and variables that
            # result from the edge/node/global_model_fns are scoped analogous to how
            # the Edge/Node/GlobalBlock classes do.
            if edge_model_fns is None:
                self._edge_model = lambda x: x
            else:
                self._edge_model = snt.Module(
                    lambda x: edge_model_fns()(x), name="edge_model")
            if node_model_fns is None:
                self._node_model = lambda x: x
            else:
                self._node_model = snt.Module(
                    lambda x: node_model_fns()(x), name="node_model")
            if global_model_fn is None:
                self._global_model = lambda x: x
            else:
                self._global_model = snt.Module(
                    lambda x: global_model_fn()(x), name="global_model")

    def _build(self, graph):
        """Connects the GraphIndependent.

        Args:
          graph: A `graphs.GraphsTuple` containing non-`None` edges, nodes and
            globals.

        Returns:
          An output `graphs.GraphsTuple` with updated edges, nodes and globals.

        """
        return graph.replace(
            edges=self._edge_model(graph.edges),
            nodes=self._node_model(graph.nodes),
            globals=self._global_model(graph.globals))


def make_mlp_model():
    """Instantiates a new MLP, followed by LayerNorm.

    The parameters of each new MLP are not shared with others generated by
    this function.

    Returns:
      A Sonnet module which contains the MLP and LayerNorm.
    """
    return snt.Sequential([
        snt.nets.MLP([LATENT_SIZE] * NUM_LAYERS, activate_final=True),
        snt.LayerNorm()
    ])


class MLPGraphIndependent(snt.AbstractModule):
    """GraphIndependent with MLP edge, node, and global models."""

    def __init__(self, name="MLPGraphIndependent"):
        super(MLPGraphIndependent, self).__init__(name=name)
        with self._enter_variable_scope():
            self._network = KGIndependent(
                edge_model_fns=make_mlp_model,
                node_model_fns=make_mlp_model,
                global_model_fn=make_mlp_model)

    def _build(self, inputs):
        return self._network(inputs)


class MLPGraphNetwork(snt.AbstractModule):
    """GraphNetwork with MLP edge, node, and global models."""

    def __init__(self, name="MLPGraphNetwork"):
        super(MLPGraphNetwork, self).__init__(name=name)
        with self._enter_variable_scope():
            self._network = modules.GraphNetwork(make_mlp_model, make_mlp_model,
                                                 make_mlp_model)

    def _build(self, inputs):
        return self._network(inputs)


class KGMessagePassingLearner(snt.AbstractModule):

    def __init__(self,
                 edge_output_size=None,
                 node_output_size=None,
                 global_output_size=None,
                 name="EncodeProcessDecode"):
        super(KGMessagePassingLearner, self).__init__(name=name)
        self._encoder = MLPGraphIndependent()
        self._core = MLPGraphNetwork()
        self._decoder = MLPGraphIndependent()
        # Transforms the outputs into the appropriate shapes.
        if edge_output_size is None:
            edge_fn = None
        else:
            edge_fn = lambda: snt.Linear(edge_output_size, name="edge_output")
        if node_output_size is None:
            node_fn = None
        else:
            node_fn = lambda: snt.Linear(node_output_size, name="node_output")
        if global_output_size is None:
            global_fn = None
        else:
            global_fn = lambda: snt.Linear(global_output_size, name="global_output")
        with self._enter_variable_scope():
            self._output_transform = modules.GraphIndependent(edge_fn, node_fn,
                                                              global_fn)

    def _build(self, input_op, num_processing_steps):
        latent = self._encoder(input_op)
        latent0 = latent
        output_ops = []
        for _ in range(num_processing_steps):
            core_input = utils_tf.concat([latent0, latent], axis=1)
            latent = self._core(core_input)
            decoded_op = self._decoder(latent)
            output_ops.append(self._output_transform(decoded_op))
        return output_ops
