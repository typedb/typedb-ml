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

import grakn.client

import kglib.kgcn.core.ingest.traverse.data.context.neighbour as neighbour
import kglib.kgcn.core.ingest.traverse.data.sample.sample as samp
import kglib.kgcn.core.ingest.traverse.data.context.builder as builder
import kglib.kgcn.core.ingest.traverse.data.context.builder_mocks as mocks


class TestUpdateDictLists(unittest.TestCase):

    def test_order_list_update(self):
        dict_to_update = {1: ['a'], 2: ['s'], 3: ['x']}
        dict_to_add = {1: ['b'], 2: ['t'], 3: ['y']}
        updated_dict = builder.update_dict_lists(dict_to_add, dict_to_update)
        expected_dict = {1: ['b', 'a'], 2: ['t', 's'], 3: ['y', 'x']}
        self.assertEqual(expected_dict, updated_dict)

    def test_key_not_overwritten(self):
        dict_to_update = {1: ['a']}
        dict_to_add = {1: ['b']}
        updated_dict = builder.update_dict_lists(dict_to_add, dict_to_update)
        expected_dict = {1: ['b', 'a']}
        self.assertEqual(expected_dict, updated_dict)

    def test_key_added(self):
        dict_to_update = {}
        dict_to_add = {1: ['a']}
        updated_dict = builder.update_dict_lists(dict_to_add, dict_to_update)
        expected_dict = {1: ['a']}
        self.assertEqual(expected_dict, updated_dict)

    def test_update_with_one_key_absent(self):
        dict_to_update = {1: ['a'], 2: ['s']}
        dict_to_add = {1: ['b']}
        updated_dict = builder.update_dict_lists(dict_to_add, dict_to_update)
        expected_dict = {1: ['b', 'a'], 2: ['s']}
        self.assertEqual(expected_dict, updated_dict)


class TestContextBuilder(unittest.TestCase):

    def test_neighbour_finder_called_with_root_node_id(self):

        tx_mock = mock.Mock(grakn.client.Transaction)
        sampler = mock.Mock(samp.Sampler)
        sampler.return_value = []

        starting_thing = mock.MagicMock(neighbour.Thing, id="0")
        mock_neighbour_finder = mock.MagicMock(neighbour.NeighbourFinder)

        context_builder = builder.ContextBuilder([sampler], neighbour_finder=mock_neighbour_finder)

        # The call to assess
        context_builder.build(tx_mock, starting_thing)

        mock_neighbour_finder.find.assert_called_once_with("0", tx_mock)

    def test_neighbour_finder_called_with_root_and_neighbour_ids(self):

        tx_mock = mock.Mock(grakn.client.Transaction)
        sampler = mock.Mock(samp.Sampler)
        sampler.return_value = mocks.gen([
            mock.MagicMock(neighbour.Connection, role_label="employmee", role_direction=1,
                           neighbour_thing=mock.MagicMock(neighbour.Thing, id="1")),
            mock.MagicMock(neighbour.Connection, role_label="@has-name-owner", role_direction=1,
                           neighbour_thing=mock.MagicMock(neighbour.Thing, id="3")),
        ])
        sampler2 = mock.Mock(samp.Sampler)
        sampler2.return_value = []

        starting_thing = mock.MagicMock(neighbour.Thing, id="0")
        mock_neighbour_finder = mock.MagicMock(neighbour.NeighbourFinder)

        context_builder = builder.ContextBuilder([sampler, sampler2], neighbour_finder=mock_neighbour_finder)

        # The call to assess
        context_builder.build(tx_mock, starting_thing)

        print(mock_neighbour_finder.find.mock_calls)
        mock_neighbour_finder.find.assert_has_calls(
            [mock.call("0", tx_mock), mock.call("1", tx_mock), mock.call("3", tx_mock)])


if __name__ == "__main__":
    unittest.main()
