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
from graph_nets.utils_np import networkxs_to_graphs_tuple

from experiment.data import create_input_target_graphs
from experiment.plotting import plot_input_vs_output


class TestPlotInputVsOutput(unittest.TestCase):
    def test_plotting(self):
        num_processing_steps_ge = 6
        num_graphs = 3
        graph_ids = [7, 11, 0, 8, 5, 6, 3, 10, 4, 1, 9, 2]
        all_node_types = ['person', 'parentship', 'siblingship']
        all_edge_types = ['parent', 'child', 'sibling']
        inputs, targets, raw_graphs = create_input_target_graphs(graph_ids, all_node_types, all_edge_types)
        # input_graphs = utils_np.networkxs_to_graphs_tuple(inputs[:num_graphs])
        target_graphs = networkxs_to_graphs_tuple(targets[:num_graphs])
        test_values = {"target": target_graphs, "outputs": [target_graphs for _ in range(6)]}
        plot_input_vs_output(raw_graphs, test_values, num_processing_steps_ge)
        plt.show()


if __name__ == "__main__":
    unittest.main()
