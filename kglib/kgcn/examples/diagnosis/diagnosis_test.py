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

import grakn.client
import networkx as nx
import numpy as np
from mock import MagicMock

from kglib.kgcn.examples.diagnosis.diagnosis import write_predictions_to_grakn
from kglib.utils.grakn.object.thing import Thing


class TestWritePredictionsToGrakn(unittest.TestCase):
    def test_query_made_as_expected(self):
        graph = nx.MultiDiGraph()

        graph.add_node(0, concept=Thing('V123', 'person', 'entity'), probabilities=np.array([1.0, 0.0, 0.0]),
                       prediction=0)
        graph.add_node(1, concept=Thing('V1235', 'disease', 'entity'), probabilities=np.array([1.0, 0.0, 0.0]),
                       prediction=0)
        graph.add_node(2, concept=Thing('V6543', 'diagnosis', 'relation'), probabilities=np.array([0.0, 0.0071, 0.9927]),
                       prediction=2)

        graph.add_edge(2, 0)
        graph.add_edge(2, 1)

        graphs = [graph]
        tx = MagicMock(grakn.client.Transaction)

        tx.commit = MagicMock()
        tx.query = MagicMock()

        write_predictions_to_grakn(graphs, tx)

        expected_query = (f'match'
                          f'$p id V123;'
                          f'$d id V1235;'
                          f'insert'
                          f'$pd(predicted-patient: $p, predicted-diagnosed-disease: $d) isa predicted-diagnosis,'
                          f'has probability-exists 0.993,'
                          f'has probability-non-exists 0.007,'
                          f'has probability-preexists 0.000;')

        tx.query.assert_called_with(expected_query)

        tx.commit.assert_called()

    def test_query_made_only_if_relation_wins(self):
        graph = nx.MultiDiGraph()

        graph.add_node(0, concept=Thing('V123', 'person', 'entity'),
                       probabilities=np.array([1.0, 0.0, 0.0]), prediction=0)
        graph.add_node(1, concept=Thing('V1235', 'disease', 'entity'),
                       probabilities=np.array([1.0, 0.0, 0.0]), prediction=0)
        graph.add_node(2, concept=Thing('V6543', 'diagnosis', 'relation'),
                       probabilities=np.array([0.0, 0.0, 1.0]), prediction=1)

        graph.add_edge(2, 0)
        graph.add_edge(2, 1)

        graphs = [graph]
        tx = MagicMock(grakn.client.Transaction)

        tx.commit = MagicMock()
        tx.query = MagicMock()

        write_predictions_to_grakn(graphs, tx)

        tx.query.assert_not_called()

        tx.commit.assert_called()


if __name__ == "__main__":
    unittest.main()
