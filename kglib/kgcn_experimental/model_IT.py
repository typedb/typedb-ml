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
import unittest

import networkx as nx
import numpy as np
import tensorflow as tf
from graph_nets.graphs import GraphsTuple

from kglib.kgcn_experimental.model import KGCN


class TestKGCN(unittest.TestCase):
    def test_kgcn_runs(self):
        graph = nx.MultiDiGraph()
        graph.add_node(0, type='person', input=1, solution=0)
        graph.add_edge(0, 1, type='employee', input=1, solution=0)
        graph.add_node(1, type='employment', input=1, solution=0)
        graph.add_edge(1, 2, type='employer', input=1, solution=0)
        graph.add_node(2, type='company', input=1, solution=0)

        kgcn = KGCN(['person', 'employment', 'company'],
                    ['employee', 'employer'],
                    5,
                    6,
                    attr_encoders={lambda x: tf.constant(np.zeros((3, 6))): [0, 1, 2]})
        kgcn([graph], [graph],
             num_processing_steps_tr=2,
             num_processing_steps_ge=2,
             num_training_iterations=50,
             log_every_seconds=0.5)


class TestModel(unittest.TestCase):
    def test_model_runs(self):
        tf.enable_eager_execution()

        input_graph = GraphsTuple(
            nodes=tf.convert_to_tensor(np.array([[1, 0], [1, 1], [1, 2]], dtype=np.float32)),
            edges=tf.convert_to_tensor(np.array([[1, 0], [1, 1]], dtype=np.float32)),
            globals=tf.convert_to_tensor(np.array([[0, 0, 0, 0, 0]], dtype=np.float32)),
            receivers=tf.convert_to_tensor(np.array([1, 2], dtype=np.int32)),
            senders=tf.convert_to_tensor(np.array([0, 1], dtype=np.int32)),
            n_node=tf.convert_to_tensor(np.array([3], dtype=np.int32)),
            n_edge=tf.convert_to_tensor(np.array([2], dtype=np.int32)))

        target_graph = GraphsTuple(
            nodes=tf.convert_to_tensor(np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=np.float32)),
            edges=tf.convert_to_tensor(np.array([[1, 0, 0], [1, 0, 0]], dtype=np.float32)),
            globals=tf.convert_to_tensor(np.array([[0, 0, 0, 0, 0]], dtype=np.float32)),
            receivers=tf.convert_to_tensor(np.array([1, 2], dtype=np.int32)),
            senders=tf.convert_to_tensor(np.array([0, 1], dtype=np.int32)),
            n_node=tf.convert_to_tensor(np.array([3], dtype=np.int32)),
            n_edge=tf.convert_to_tensor(np.array([2], dtype=np.int32)))

        kgcn = KGCN(['person', 'employment', 'company'], ['employee', 'employer'])
        model = kgcn._build()
        output_ops_tr = model(input_graph, 2)
        output_ops_ge = model(target_graph, 2)

    def test_model_runs_2(self):
        tf.enable_eager_execution()

        graph = GraphsTuple(nodes=tf.convert_to_tensor(np.array([[1, 0], [1, 1], [1, 2]], dtype=np.float32)),
                            edges=tf.convert_to_tensor(np.array([[1, 0], [1, 1]], dtype=np.float32)),
                            globals=tf.convert_to_tensor(np.array([[0, 0, 0, 0, 0]], dtype=np.float32)),
                            receivers=tf.convert_to_tensor(np.array([1, 2], dtype=np.int32)),
                            senders=tf.convert_to_tensor(np.array([0, 1], dtype=np.int32)),
                            n_node=tf.convert_to_tensor(np.array([3], dtype=np.int32)),
                            n_edge=tf.convert_to_tensor(np.array([2], dtype=np.int32)))

        kgcn = KGCN(['person', 'employment', 'company'], ['employee', 'employer'])
        model = kgcn._build()
        output1 = model(graph, 2)
        output2 = model(graph, 2)

    def test_model_runs_3(self):
        tf.enable_eager_execution()

        graph = GraphsTuple(nodes=tf.convert_to_tensor(np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=np.float32)),
                            edges=tf.convert_to_tensor(np.array([[1, 0, 0], [1, 0, 0]], dtype=np.float32)),
                            globals=tf.convert_to_tensor(np.array([[0, 0, 0, 0, 0]], dtype=np.float32)),
                            receivers=tf.convert_to_tensor(np.array([1, 2], dtype=np.int32)),
                            senders=tf.convert_to_tensor(np.array([0, 1], dtype=np.int32)),
                            n_node=tf.convert_to_tensor(np.array([3], dtype=np.int32)),
                            n_edge=tf.convert_to_tensor(np.array([2], dtype=np.int32)))

        kgcn = KGCN(['person', 'employment', 'company'], ['employee', 'employer'])
        model = kgcn._build()
        output1 = model(graph, 2)
        output2 = model(graph, 2)


class TestTensorScatterAdd(unittest.TestCase):

    def test_with_eager_execution(self):
        tf.enable_eager_execution()
        encoded_features = tf.zeros((2, 15), dtype=tf.float32)

        indices_to_encode = tf.convert_to_tensor(np.array([[0], [1]], dtype=np.int64))

        encoded_feats = tf.convert_to_tensor(np.array(
            [[-0.8854138, -0.8854138, -0.28424942, -0.8854138, -0.8854138, 0.15186566,
              -0.8854138, -0.09432995, 0.48828167, - 0.8854138, 1.7825887, 1.1054695,
              1.7577922, -0.8854138, 1.2904773],
             [-0.5648398, -0.5648398, -0.5648398, 0.11074328, -0.5648398, -0.5648398,
              -0.5648398, 1.2660778, -0.5648398, -0.5648398, 1.8472059, -0.5648398,
              -0.5648398, 2.5975735, -0.17320272]], dtype=np.float32))
        encoded_features = tf.tensor_scatter_add(encoded_features, indices_to_encode, encoded_feats)
        print(encoded_features)
        tf.disable_eager_execution()

    def test_with_session(self):

        encoded_features = np.zeros((2, 15), dtype=np.float32)

        indices_to_encode = np.array([[0]], dtype=np.int64)

        encoded_feats = np.array(
            [[-0.8854138, -0.8854138, -0.28424942, -0.8854138, -0.8854138, 0.15186566,
              -0.8854138, -0.09432995, 0.48828167, - 0.8854138, 1.7825887, 1.1054695,
              1.7577922, -0.8854138, 1.2904773],
             ], dtype=np.float32)

        encoded_features_ph = tf.placeholder(tf.float32, (2, 15))

        indices_to_encode_ph = tf.placeholder(tf.int64, indices_to_encode.shape)

        encoded_feats_ph = tf.placeholder(tf.float32, encoded_feats.shape)

        output_op = tf.tensor_scatter_add(encoded_features_ph, indices_to_encode_ph, encoded_feats_ph)

        feed_dict = {encoded_features_ph: encoded_features,
                     indices_to_encode_ph: indices_to_encode,
                     encoded_feats_ph: encoded_feats}

        with tf.Session() as sess:
            result = sess.run({"output": output_op}, feed_dict=feed_dict)
            print(result)


if __name__ == "__main__":
    tf.enable_eager_execution()
    unittest.main()
