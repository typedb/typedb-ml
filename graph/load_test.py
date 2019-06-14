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
import unittest.mock as mock

import grakn.client as client

import graph.load as load
import kglib.kgcn.core.ingest.traverse.data.context.neighbour as neighbour
from graph.mock.concept import MockType, MockThing


class TestConceptDictsFromQuery(unittest.TestCase):
    def test_concept_dicts_are_built_as_expected(self):
        query = 'match $x id V123; get;'

        mock_tx = mock.MagicMock(client.Transaction)

        answer_map = [{'x': MockThing('V123', 'ENTITY', MockType('V456', 'ENTITY', 'person'))}]

        mock_tx.query.return_value = answer_map
        concept_dicts = load.concept_dicts_from_query(query, mock_tx)
        mock_tx.query.assert_called_with(query)

        expected_concept_dicts = [{'x': neighbour.Thing('V123', 'person', 'entity')}]

        self.assertEqual(expected_concept_dicts, concept_dicts)

    def test_concept_dicts_are_built_as_expected_with_2_concepts(self):
        query = 'match $x id V123; get;'

        mock_tx = mock.MagicMock(client.Transaction)

        answer_map = [{
            'x': MockThing('V123', 'ENTITY', MockType('V456', 'ENTITY', 'person')),
            'y': MockThing('V789', 'RELATION', MockType('V765', 'RELATION', 'employment')),
        }]

        mock_tx.query.return_value = answer_map
        concept_dicts = load.concept_dicts_from_query(query, mock_tx)
        mock_tx.query.assert_called_with(query)

        expected_concept_dicts = [{
            'x': neighbour.Thing('V123', 'person', 'entity'),
            'y': neighbour.Thing('V789', 'employment', 'relation'),
        }]

        self.assertEqual(expected_concept_dicts, concept_dicts)
