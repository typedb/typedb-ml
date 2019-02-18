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

import datetime
import unittest

import numpy as np

import kglib.kgcn.core.ingest.traverse.data.context.neighbour as neighbour
import kglib.kgcn.core.ingest.traverse.data.sample.ordered as ordered
import kglib.kgcn.core.ingest.traverse.data.sample.sample as samp
import kglib.kgcn.core.ingest.traverse.data.context.builder as builder
import kglib.kgcn.core.ingest.traverse.data.context.builder_mocks as mock
import kglib.kgcn.core.ingest.traverse.data.context.array as array


class TestDetermineValuesToPut(unittest.TestCase):

    def test__determine_values_to_put_with_entity(self):
        role_label = 'employer'
        role_direction = neighbour.TARGET_PLAYS
        neighbour_type_label = 'company'
        neighbour_data_type = None
        neighbour_value = None
        values_dict = array.determine_values_to_put(role_label, role_direction, neighbour_type_label,
                                                    neighbour_data_type, neighbour_value)
        expected_result = {"role_type": 'employer',
                           'role_direction': role_direction,
                           'neighbour_type': 'company'
                           }
        self.assertEqual(values_dict, expected_result)

    def test__determine_values_to_put_with_string_attribute(self):
        role_label = '@has-name-value'
        role_direction = neighbour.NEIGHBOUR_PLAYS
        neighbour_type_label = 'name'
        neighbour_data_type = 'string'
        neighbour_value = 'Person\'s Name'
        values_dict = array.determine_values_to_put(role_label, role_direction, neighbour_type_label,
                                                    neighbour_data_type, neighbour_value)
        expected_result = {"role_type": '@has-name-value',
                           'role_direction': role_direction,
                           'neighbour_type': 'name',
                           'neighbour_data_type': 'string',
                           'neighbour_value_string': neighbour_value}
        self.assertEqual(expected_result, values_dict)


class TestContextArrayBuilderFromMockEntity(unittest.TestCase):

    def setUp(self):
        self._neighbourhood_sizes = (2, 3)
        self._num_example_things = 2

        self._builder = array.ArrayConverter(self._neighbourhood_sizes)

        self._expected_dims = [self._num_example_things] + list(reversed(self._neighbourhood_sizes)) + [1]

    def _check_dims(self, arrays):
        # We expect dimensions:
        # (2, 3, 2, 1)
        # (2, 2, 1)
        # (2, 1)
        exp = [[self._expected_dims[0]] + list(self._expected_dims[i+1:]) for i in range(len(self._expected_dims)-1)]
        for i in range(len(self._expected_dims) - 1):
            with self.subTest(exp[i]):
                self.assertEqual(arrays[i]['neighbour_type'].shape, tuple(exp[i]))

    def _thing_contexts_factory(self):
        return builder.convert_thing_contexts_to_neighbours(
            [mock.mock_traversal_output(), mock.mock_traversal_output()])

    def test_build_context_arrays(self):

        depthwise_arrays = self._builder.convert_to_array(self._thing_contexts_factory())
        self._check_dims(depthwise_arrays)
        with self.subTest('spot-check thing type'):
            self.assertEqual(depthwise_arrays[-1]['neighbour_type'][0, 0], 'person')
        with self.subTest('spot-check role type'):
            self.assertEqual(depthwise_arrays[0]['role_type'][0, 0, 0, 0], 'employer')
        with self.subTest('check role_type absent in final arrays'):
            self.assertFalse('role_type' in list(depthwise_arrays[-1].keys()))
        with self.subTest('check role_direction absent in final arrays'):
            self.assertFalse('role_direction' in list(depthwise_arrays[-1].keys()))

    def test_initialised_array_sizes(self):

        initialised_arrays = self._builder._initialise_arrays(self._num_example_things)
        self._check_dims(initialised_arrays)

    def test_array_values(self):
        depthwise_arrays = self._builder.convert_to_array(self._thing_contexts_factory())
        with self.subTest('role_type not empty'):
            self.assertFalse('' in depthwise_arrays[0]['role_type'])
        with self.subTest('neighbour_type not empty'):
            self.assertFalse('' in depthwise_arrays[0]['neighbour_type'])


class TestIntegrationsContextArrayBuilderFromEntity(unittest.TestCase):
    def setUp(self):
        import grakn
        entity_query = "match $x isa company, has name 'Google'; get;"
        uri = "localhost:48555"
        keyspace = "test_schema"
        client = grakn.Grakn(uri=uri)
        session = client.session(keyspace=keyspace)
        self._tx = session.transaction(grakn.TxType.WRITE)

        neighbour_sample_sizes = (6, 4, 4)
        sampling_method = ordered.ordered_sample

        samplers = []
        for sample_size in neighbour_sample_sizes:
            samplers.append(samp.Sampler(sample_size, sampling_method, limit=sample_size * 2))

        grakn_things = [answermap.get('x') for answermap in list(self._tx.query(entity_query))]

        things = [neighbour.build_thing(grakn_thing) for grakn_thing in grakn_things]

        context_builder = builder.ContextBuilder(samplers)

        neighbourhood_depths = [context_builder.build(self._tx, thing) for thing in things]

        neighbour_roles = builder.convert_thing_contexts_to_neighbours(neighbourhood_depths)

        # flat = builder.flatten_tree(neighbour_roles)
        # [builder.collect_to_tree(neighbourhood_depth) for neighbourhood_depth in neighbourhood_depths]

        ################################################################################################################
        # Context Array Building
        ################################################################################################################

        self._array_builder = array.ArrayConverter(neighbour_sample_sizes)
        self._context_arrays = self._array_builder.convert_to_array(neighbour_roles)

    def test_array_values(self):
        with self.subTest('role_type not empty'):
            self.assertFalse('' in self._context_arrays[0]['role_type'])
        with self.subTest('neighbour_type not empty'):
            self.assertFalse('' in self._context_arrays[0]['neighbour_type'])


class TestIntegrationsContextArrayBuilderWithMock(unittest.TestCase):
    def setUp(self):

        self._neighbour_sample_sizes = (2, 3)

        neighbourhood_depths = [mock.mock_traversal_output()]

        neighbour_roles = builder.convert_thing_contexts_to_neighbours(neighbourhood_depths)

        ################################################################################################################
        # Context Array Building
        ################################################################################################################

        self._array_builder = array.ArrayConverter(self._neighbour_sample_sizes)
        self._context_arrays = self._array_builder.convert_to_array(neighbour_roles)

    def test_role_type_not_empty(self):
        self.assertFalse('' in self._context_arrays[0]['role_type'])

    def test_neighbour_type_not_empty(self):
        self.assertFalse('' in self._context_arrays[0]['neighbour_type'])

    def test_string_values_not_empty(self):
        self.assertFalse('' in self._context_arrays[0]['neighbour_value_string'][0, :, 1, 0])

    def test_data_type_values_not_empty(self):
        self.assertFalse('' in self._context_arrays[0]['neighbour_data_type'][0, :, 1, 0])

    def test_shapes_as_expected(self):
        self.assertTupleEqual(self._context_arrays[0]['neighbour_type'].shape, (1, 3, 2, 1))
        self.assertTupleEqual(self._context_arrays[1]['neighbour_type'].shape, (1, 2, 1))
        self.assertTupleEqual(self._context_arrays[2]['neighbour_type'].shape, (1, 1))

    # def test_all_indices_visited(self):
    #     print(self._array_builder.indices_visited)
    #     self.assertEqual(len(self._array_builder.indices_visited), 6+2+1)

    def test_array_values(self):
        with self.subTest('role_type not empty'):
            self.assertFalse('' in self._context_arrays[0]['role_type'])
        with self.subTest('neighbour_type not empty'):
            self.assertFalse('' in self._context_arrays[0]['neighbour_type'])


class TestAttributeTypes(unittest.TestCase):
    def test_date(self):
        neighbour_roles = [builder.Neighbour(None, None, builder.ThingContext(
            neighbour.Thing("1", "start-date", "attribute", data_type='date',
                            value=datetime.datetime(day=1, month=1, year=2018)),
            mock.gen([])
        )), ]

        self._array_builder = array.ArrayConverter((2, 3))
        self._context_arrays = self._array_builder.convert_to_array(neighbour_roles)


class TestFillArrayWithRepeats(unittest.TestCase):
    def test_repeat_top_neighbour(self):
        shape = (2, 3, 2, 1)
        arr = np.array([[[[1],
                          [2]],
                         [[3],
                          [4]],
                         [[5],
                          [6]]],
                        [[[7],
                          [8]],
                         [[9],
                          [10]],
                         [[11],
                          [12]]]])
        arr[1, :, 1:, 0] = 0

        array.fill_array_with_repeats(arr, (1, ..., slice(1), 0), (1, ..., slice(1, None), 0))
        print(arr)

        expected_output = np.array([[[[1],
                                      [2]],
                                     [[3],
                                      [4]],
                                     [[5],
                                      [6]]],
                                    [[[7],
                                      [7]],
                                     [[9],
                                      [9]],
                                     [[11],
                                      [11]]]])
        np.testing.assert_array_equal(expected_output, arr)

    def test_repeat_top_neighbour_deep(self):
        shape = (2, 2, 2, 2, 1)
        arr = np.array([[[[[1],
                           [2]],
                          [[3],
                           [4]]],
                         [[[5],
                           [6]],
                          [[7],
                           [8]]]],
                        [[[[9],
                           [10]],
                          [[11],
                           [12]]],
                         [[[13],
                           [14]],
                          [[15],
                           [16]]]]])
        arr[1, :, :, 1:, 0] = 0

        array.fill_array_with_repeats(arr, (1, ..., slice(1), 0), (1, ..., slice(1, None), 0))
        print(arr)

        expected_output = np.array([[[[[1],
                                       [2]],
                                      [[3],
                                       [4]]],
                                     [[[5],
                                       [6]],
                                      [[7],
                                       [8]]]],
                                    [[[[9],
                                       [9]],
                                      [[11],
                                       [11]]],
                                     [[[13],
                                       [13]],
                                      [[15],
                                       [15]]]]])
        np.testing.assert_array_equal(expected_output, arr)

    def test_repeat_child_neighbour(self):
        shape = (2, 3, 2, 1)
        arr = np.array([[[[1],
                          [2]],
                         [[3],
                          [4]],
                         [[5],
                          [6]]],
                        [[[7],
                          [8]],
                         [[9],
                          [10]],
                         [[11],
                          [12]]]])
        arr[1, 1:, 1, 0] = 0

        array.fill_array_with_repeats(arr, ([1], ..., slice(1), [1], [0]), ([1], ..., slice(1, None), [1], [0]))
        print(arr)

        expected_output = np.array([[[[1],
                                      [2]],
                                     [[3],
                                      [4]],
                                     [[5],
                                      [6]]],
                                    [[[7],
                                      [8]],
                                     [[9],
                                      [8]],
                                     [[11],
                                      [8]]]])
        np.testing.assert_array_equal(expected_output, arr)

    def test_repeat_less_than_available(self):
        shape = (2, 3, 2, 1)
        arr = np.array([[[[1],
                          [2]],
                         [[3],
                          [4]],
                         [[5],
                          [6]]],
                        [[[7],
                          [8]],
                         [[9],
                          [10]],
                         [[11],
                          [12]]]])
        arr[0, 2:, 0, 0] = 0

        array.fill_array_with_repeats(arr, (0, ..., slice(2), 0, 0), (0, ..., slice(2, None), 0, 0))
        print(arr)

        expected_output = np.array([[[[1],
                                      [2]],
                                     [[3],
                                      [4]],
                                     [[1],
                                      [6]]],
                                    [[[7],
                                      [8]],
                                     [[9],
                                      [10]],
                                     [[11],
                                      [12]]]])
        np.testing.assert_array_equal(expected_output, arr)


if __name__ == "__main__":
    unittest.main()
