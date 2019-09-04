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
from functools import partial

import numpy as np
import sonnet as snt
import tensorflow as tf
from graph_nets import modules
from graph_nets import utils_tf
from graph_nets.modules import GraphIndependent

from kglib.kgcn_experimental.graph_utils.encode import encode_solutions, augment_data_fields, \
    encode_categorically
from kglib.kgcn_experimental.graph_utils.iterate import multidigraph_edge_data_iterator, \
    multidigraph_node_data_iterator, multidigraph_data_iterator
from kglib.kgcn_experimental.graph_utils.prepare import make_all_runnable_in_session
from kglib.kgcn_experimental.network.embedding import common_embedding, node_embedding
from kglib.kgcn_experimental.network.feed import create_feed_dict, create_placeholders
from kglib.kgcn_experimental.network.loss import loss_ops_from_difference
from kglib.kgcn_experimental.network.metrics import compute_accuracy
from kglib.kgcn_experimental.plot.plotting import plot_across_training, plot_input_vs_output


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


class KGCN:
    def __init__(self,
                 all_node_types,
                 all_edge_types,
                 type_embedding_dim,
                 attr_embedding_dim,
                 attr_encoders,
                 latent_size=16, num_layers=2):
        self._attr_encoders = attr_encoders
        self._latent_size = latent_size
        self._num_layers = num_layers
        self._type_embedding_dim = type_embedding_dim
        self._attr_embedding_dim = attr_embedding_dim
        self.all_node_types = all_node_types
        self.all_edge_types = all_edge_types

    def input_fn(self, graphs):
        input_graphs = []
        target_graphs = []
        for graph in graphs:
            graph = encode_solutions(graph, solution_field="solution", encoded_solution_field="encoded_solution",
                                     encodings=np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]))

            node_iterator = multidigraph_node_data_iterator(graph)
            encode_categorically(node_iterator, self.all_node_types, 'type', 'categorical_type')

            edge_iterator = multidigraph_edge_data_iterator(graph)
            encode_categorically(edge_iterator, self.all_edge_types, 'type', 'categorical_type')

            features_field = "features"

            input_graph = graph.copy()
            augment_data_fields(multidigraph_data_iterator(input_graph),
                                ("input", "categorical_type", "value"),
                                features_field)
            input_graph.graph[features_field] = np.array([0.0] * 5, dtype=np.float32)

            target_graph = graph.copy()
            augment_data_fields(multidigraph_data_iterator(target_graph), ("encoded_solution",), features_field)
            target_graph.graph[features_field] = np.array([0.0] * 5, dtype=np.float32)

            input_graphs.append(input_graph)
            target_graphs.append(target_graph)
        return input_graphs, target_graphs

    def _build(self):

        def make_edge_model():
            common_embedding_module = snt.Module(
                partial(common_embedding, num_types=len(self.all_edge_types),
                        type_embedding_dim=self._type_embedding_dim)
            )

            return snt.Sequential([common_embedding_module,
                                   snt.nets.MLP([self._latent_size] * self._num_layers, activate_final=True),
                                   snt.LayerNorm()])

        def make_node_model():
            node_embedding_module = snt.Module(
                partial(node_embedding, num_types=len(self.all_node_types), type_embedding_dim=self._type_embedding_dim,
                        attr_encoders=self._attr_encoders, attr_embedding_dim=self._attr_embedding_dim)
            )
            return snt.Sequential([node_embedding_module,
                                   snt.nets.MLP([self._latent_size] * self._num_layers, activate_final=True),
                                   snt.LayerNorm()])

        kg_encoder = lambda: GraphIndependent(make_edge_model, make_node_model, make_mlp_model, name='kg_encoder')

        model = KGMessagePassingLearner(kg_encoder, edge_output_size=3, node_output_size=3)
        return model

    def __call__(self, tr_graphs, ge_graphs, num_processing_steps_tr=10, num_processing_steps_ge=10,
                 num_training_iterations=10000, log_every_seconds=2, log_dir=None):
        """
        Args:
            tr_graphs: In-memory graphs of Grakn concepts for training
            ge_graphs: In-memory graphs of Grakn concepts for generalisation
            num_processing_steps_tr: Number of processing (message-passing) steps for training.
            num_processing_steps_ge: Number of processing (message-passing) steps for generalization.
            num_training_iterations: Number of training iterations
            log_every_seconds: The time to wait between logging and printing the next set of results.
            log_dir: Directory to store TensorFlow events files

        Returns:

        """

        model = self._build()
        tf.set_random_seed(1)

        tr_input_graphs, tr_target_graphs = self.input_fn(tr_graphs)
        ge_input_graphs, ge_target_graphs = self.input_fn(ge_graphs)

        input_ph, target_ph = create_placeholders(tr_input_graphs, tr_target_graphs)

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

        # Optimizer
        learning_rate = 1e-3
        optimizer = tf.train.AdamOptimizer(learning_rate)
        step_op = optimizer.minimize(loss_op_tr)

        input_ph, target_ph = make_all_runnable_in_session(input_ph, target_ph)

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


def make_mlp_model(latent_size=16, num_layers=2):
    """Instantiates a new MLP, followed by LayerNorm.

    The parameters of each new MLP are not shared with others generated by
    this function.

    Returns:
      A Sonnet module which contains the MLP and LayerNorm.
    """
    return snt.Sequential([
        snt.nets.MLP([latent_size] * num_layers, activate_final=True),
        snt.LayerNorm()
    ])


class MLPGraphIndependent(snt.AbstractModule):
    """GraphIndependent with MLP edge, node, and global models."""

    def __init__(self, name="MLPGraphIndependent"):
        super(MLPGraphIndependent, self).__init__(name=name)
        with self._enter_variable_scope():
            self._network = GraphIndependent(
                edge_model_fn=make_mlp_model,
                node_model_fn=make_mlp_model,
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
                 kg_encoder,
                 edge_output_size=None,
                 node_output_size=None,
                 global_output_size=None,
                 name="KGMessagePassingLearner"):
        super(KGMessagePassingLearner, self).__init__(name=name)

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
            self._encoder = kg_encoder()
            self._core = MLPGraphNetwork()
            self._decoder = MLPGraphIndependent()
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
