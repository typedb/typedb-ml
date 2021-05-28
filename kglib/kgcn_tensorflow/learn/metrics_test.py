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

import unittest

import numpy as np
from graph_nets.graphs import GraphsTuple

from kglib.kgcn_tensorflow.learn.metrics import compute_accuracy, existence_accuracy


class TestComputeAccuracy(unittest.TestCase):

    def test_compute_accuracy_is_as_expected(self):

        t_nodes = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
        o_nodes = np.array([[0, 1], [1, 0], [1, 0]], dtype=np.float32)
        t_edges = np.array([[0, 1], [1, 0]], dtype=np.float32)
        o_edges = np.array([[1, 0], [1, 0]], dtype=np.float32)

        globals = None
        senders = np.array([0, 1])
        receivers = np.array([1, 2])
        n_node = np.array([3])
        n_edge = np.array([2])

        target = GraphsTuple(nodes=t_nodes,
                             edges=t_edges,
                             globals=globals,
                             receivers=receivers,
                             senders=senders,
                             n_node=n_node,
                             n_edge=n_edge)

        output = GraphsTuple(nodes=o_nodes,
                             edges=o_edges,
                             globals=globals,
                             receivers=receivers,
                             senders=senders,
                             n_node=n_node,
                             n_edge=n_edge)

        correct, solved = compute_accuracy(target, output)

        expected_correct = 2 / 5
        expected_solved = 0

        self.assertEqual(expected_correct, correct)
        self.assertEqual(expected_solved, solved)


class TestExistenceAccuracy(unittest.TestCase):

    def test_compute_accuracy_is_as_expected(self):

        t_nodes = np.array([[1, 0, 0], [0, 0, 1], [0, 0, 1]], dtype=np.float32)
        o_nodes = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        t_edges = np.array([[0, 1, 0], [1, 0, 0]], dtype=np.float32)
        o_edges = np.array([[1, 0, 0], [1, 0, 0]], dtype=np.float32)

        globals = None
        senders = np.array([0, 1])
        receivers = np.array([1, 2])
        n_node = np.array([3])
        n_edge = np.array([2])

        target = GraphsTuple(nodes=t_nodes,
                             edges=t_edges,
                             globals=globals,
                             receivers=receivers,
                             senders=senders,
                             n_node=n_node,
                             n_edge=n_edge)

        output = GraphsTuple(nodes=o_nodes,
                             edges=o_edges,
                             globals=globals,
                             receivers=receivers,
                             senders=senders,
                             n_node=n_node,
                             n_edge=n_edge)

        correct, solved = existence_accuracy(target, output)

        expected_correct = 2/3
        expected_solved = 0.0

        self.assertEqual(expected_correct, correct)
        self.assertEqual(expected_solved, solved)


if __name__ == "__main__":
    unittest.main()
