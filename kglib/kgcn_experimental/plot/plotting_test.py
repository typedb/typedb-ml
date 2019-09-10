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

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from graph_nets.graphs import GraphsTuple

from kglib.kgcn_experimental.plot.plotting import plot_input_vs_output


class TestPlotInputVsOutput(unittest.TestCase):
    def test_plotting(self):
        num_processing_steps_ge = 6

        graph = nx.MultiDiGraph(name=0)

        existing = dict(input=1, solution=0)
        to_infer = dict(input=0, solution=2)
        candidate = dict(input=0, solution=1)

        # people
        graph.add_node(0, type='person', **existing)
        graph.add_node(1, type='person', **candidate)

        # parentships
        graph.add_node(2, type='parentship', **to_infer)
        graph.add_edge(2, 0, type='parent', **to_infer)
        graph.add_edge(2, 1, type='child', **candidate)

        graph_tuple = GraphsTuple(nodes=np.array([[1., 0., 0.],
                                                  [1., 1., 0.],
                                                  [1., 0., 1.]]),
                                  edges=np.array([[1., 0., 0.],
                                                  [1., 1., 0.]]),
                                  receivers=np.array([1, 2], dtype=np.int32),
                                  senders=np.array([0, 1], dtype=np.int32),
                                  globals=np.array([[0., 0., 0., 0., 0.]], dtype=np.float32),
                                  n_node=np.array([3], dtype=np.int32),
                                  n_edge=np.array([2], dtype=np.int32))

        test_values = {"target": graph_tuple, "outputs": [graph_tuple for _ in range(6)]}
        plot_input_vs_output([graph], test_values, num_processing_steps_ge)
        plt.show()


if __name__ == "__main__":
    unittest.main()
