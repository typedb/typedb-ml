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
from unittest.mock import MagicMock

from typedb.api.query.query_manager import QueryManager
from typedb.client import *
import networkx as nx
import numpy as np

from kglib.kgcn_tensorflow.examples.diagnosis.diagnosis import write_predictions_to_typedb, obfuscate_labels
from kglib.utils.typedb.object.thing import Thing
from kglib.utils.graph.test.case import GraphTestCase


class TestWritePredictionsToTypeDB(unittest.TestCase):
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
        tx = MagicMock(TypeDBTransaction)

        tx.commit = MagicMock()
        tx.query.return_value = query = MagicMock(QueryManager)

        write_predictions_to_typedb(graphs, tx)

        expected_query = (f'match '
                          f'$p iid V123;'
                          f'$d iid V1235;'
                          f'$kgcn isa kgcn;'
                          f'insert '
                          f'$pd(patient: $p, diagnosed-disease: $d, diagnoser: $kgcn) isa diagnosis,'
                          f'has probability-exists 0.993,'
                          f'has probability-non-exists 0.007,'
                          f'has probability-preexists 0.000;')

        query.insert.assert_called_with(expected_query)

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
        tx = MagicMock(TypeDBTransaction)

        tx.commit = MagicMock()
        tx.query = MagicMock(QueryManager)

        write_predictions_to_typedb(graphs, tx)

        tx.query.assert_not_called()

        tx.commit.assert_called()


class TestObfuscateLabels(GraphTestCase):

    def test_labels_obfuscated_as_expected(self):

        graph = nx.MultiDiGraph()

        graph.add_node(0, type='person')
        graph.add_node(1, type='disease')
        graph.add_node(2, type='candidate-diagnosis')

        graph.add_edge(2, 0, type='candidate-patient')
        graph.add_edge(2, 1, type='candidate-diagnosed-disease')

        obfuscate_labels(graph, {'candidate-diagnosis': 'diagnosis',
                                 'candidate-patient': 'patient',
                                 'candidate-diagnosed-disease': 'diagnosed-disease'})

        expected_graph = nx.MultiDiGraph()
        expected_graph.add_node(0, type='person')
        expected_graph.add_node(1, type='disease')
        expected_graph.add_node(2, type='diagnosis')

        expected_graph.add_edge(2, 0, type='patient')
        expected_graph.add_edge(2, 1, type='diagnosed-disease')

        self.assertGraphsEqual(graph, expected_graph)


if __name__ == "__main__":
    unittest.main()
