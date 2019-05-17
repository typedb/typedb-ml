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

import collections
import datetime
import unittest

import numpy as np

import kglib.kgcn.core.ingest.traverse.data.context.array as array
import kglib.kgcn.core.ingest.traverse.data.context.builder as builder
import kglib.kgcn.core.ingest.traverse.data.context.builder_mocks as mock
import kglib.kgcn.core.ingest.traverse.data.context.neighbour as neighbour

"""
Expected procedure:
- Take in context as a dict of lists, for a single example
- Convert to the values to put into the arrays, against the index to add at
- Initialise the arrays at different depths, for the data categories with default values
- Put the values for the example into the initialised array, without repetition (for simplicity, since this shouldn't 
have a serious effect on the model
- Combine with arrays to represent the list of examples
"""

# expected_output = collections.OrderedDict(
#             [('role_type', np.array([['employee']], dtype='U50')),
#              ('role_direction', np.array([[0]], dtype=np.int)),
#              ('neighbour_type', np.array([['person']], dtype=np.dtype('U50'))),
#              ('neighbour_data_type', np.array([['']], dtype=np.dtype('U10'))),
#              ('neighbour_value_long', np.array([[0]], dtype=np.int)),
#              ('neighbour_value_double', np.array([[0.0]], dtype=np.float)),
#              ('neighbour_value_boolean', np.array([[-1]], dtype=np.int)),
#              ('neighbour_value_date', np.array([['']], dtype='datetime64[s]')),
#              ('neighbour_value_string', np.array([['']], dtype=np.dtype('U50')))])


def mock_context():
    return {
        0: [builder.Node((), neighbour.Thing("0", "person", "entity"))],
        1: [builder.Node((0,), neighbour.Thing("1", "name", "attribute", data_type='string', value='Sundar Pichai'),
                         "has", neighbour.NEIGHBOUR_PLAYS),
            builder.Node((1,), neighbour.Thing("2", "employment", "relation"), "employee", neighbour.TARGET_PLAYS),
            ],
        2: [builder.Node((0, 0), neighbour.Thing("0", "person", "entity"), "has", neighbour.TARGET_PLAYS),
            # Note that (0, 1) is reversed compared to the natural expectation
            builder.Node((0, 1), neighbour.Thing("3", "company", "entity"), "employer", neighbour.NEIGHBOUR_PLAYS),
            builder.Node((1, 1), neighbour.Thing("0", "person", "entity"), "employee", neighbour.NEIGHBOUR_PLAYS),
            ]
    }


class TestContextToIndexedValues(unittest.TestCase):

    def test_context_to_indexed_values_ias_as_expected(self):
        context = mock_context()
        expected_indexed_values = {
            0: {(): {'neighbour_type': "person"}},
            1: {(0,): {'role_type': "has", 'role_direction': neighbour.NEIGHBOUR_PLAYS, 'neighbour_type': "name",
                       'neighbour_data_type': "string", 'neighbour_value_string': "Sundar Pichai"},
                (1,): {'role_type': "employee", 'role_direction': neighbour.TARGET_PLAYS,
                       'neighbour_type': "employment"},
                },
            2: {(0, 0): {'role_type': "has", 'role_direction': neighbour.TARGET_PLAYS, 'neighbour_type': "person"},
                (0, 1): {'role_type': "employer", 'role_direction': neighbour.NEIGHBOUR_PLAYS,
                         'neighbour_type': "company"},
                (1, 1): {'role_type': "employee", 'role_direction': neighbour.NEIGHBOUR_PLAYS,
                         'neighbour_type': "person"},
                },
        }
        vals = array.get_context_values_to_put(context)
        self.assertEqual(expected_indexed_values, vals)


class TestInitialiseArraysWithDefaultValues(unittest.TestCase):

    def test_arrays_are_initialised_as_expected(self):
        array_shape = (2, 1)

        initialised_arrays = array.initialise_arrays(array_shape,
                                                     role_type=(np.dtype('U50'), ''),
                                                     role_direction=(np.int, 0),
                                                     neighbour_type=(np.dtype('U50'), ''),
                                                     neighbour_data_type=(np.dtype('U10'), ''),
                                                     neighbour_value_long=(np.int, 0),
                                                     neighbour_value_double=(np.float, 0.0),
                                                     neighbour_value_boolean=(np.int, -1),
                                                     neighbour_value_date=('datetime64[s]', ''),
                                                     neighbour_value_string=(np.dtype('U50'), ''))

        expected_arrays_1_hop = {
            'role_type': np.array([[''], ['']], dtype=np.dtype('U50')),
            'role_direction': np.array([[0], [0]], dtype=np.int),
            'neighbour_type': np.array([[''], ['']], dtype=np.dtype('U50')),
            'neighbour_data_type': np.array([[''], ['']], dtype=np.dtype('U10')),
            'neighbour_value_long': np.array([[0], [0]], dtype=np.int),
            'neighbour_value_double': np.array([[0.0], [0.0]], dtype=np.int),
            'neighbour_value_boolean': np.array([[-1], [-1]], dtype=np.int),
            'neighbour_value_date': np.array([[''], ['']], dtype='datetime64[s]'),
            'neighbour_value_string': np.array([[''], ['']], dtype=np.dtype('U50'))
        }
        np.testing.assert_equal(expected_arrays_1_hop, initialised_arrays)

    def test_arrays_initialised_when_array_types_are_omitted(self):
        array_shape = (2, 1)

        initialised_arrays = array.initialise_arrays(array_shape,
                                                     neighbour_type=(np.dtype('U50'), ''),
                                                     )

        expected_arrays_1_hop = {
            'neighbour_type': np.array([[''], ['']], dtype=np.dtype('U50')),
        }
        np.testing.assert_equal(expected_arrays_1_hop, initialised_arrays)


class TestInitialiseArraysAtAllDepthsWithDefaultValues(unittest.TestCase):

    def test_arrays_are_initialised_at_all_depths_as_expected(self):
        deepest_array_shape = (2, 1)
        initialised_arrays = array.initialise_arrays_for_all_depths(deepest_array_shape,
                                                                    role_type=(np.dtype('U50'), ''),
                                                                    role_direction=(np.int, 0),
                                                                    neighbour_type=(np.dtype('U50'), ''),
                                                                    neighbour_data_type=(np.dtype('U10'), ''),
                                                                    neighbour_value_long=(np.int, 0),
                                                                    neighbour_value_double=(np.float, 0.0),
                                                                    neighbour_value_boolean=(np.int, -1),
                                                                    neighbour_value_date=('datetime64[s]', ''),
                                                                    neighbour_value_string=(np.dtype('U50'), ''))

        expected_arrays = {
            0: {
                'role_type': np.array([['']], dtype=np.dtype('U50')),
                'role_direction': np.array([[0]], dtype=np.int),
                'neighbour_type': np.array([['']], dtype=np.dtype('U50')),
                'neighbour_data_type': np.array([['']], dtype=np.dtype('U10')),
                'neighbour_value_long': np.array([[0]], dtype=np.int),
                'neighbour_value_double': np.array([[0.0]], dtype=np.int),
                'neighbour_value_boolean': np.array([[-1]], dtype=np.int),
                'neighbour_value_date': np.array([['']], dtype='datetime64[s]'),
                'neighbour_value_string': np.array([['']], dtype=np.dtype('U50'))
            },
            1: {
                'role_type': np.array([[''], ['']], dtype=np.dtype('U50')),
                'role_direction': np.array([[0], [0]], dtype=np.int),
                'neighbour_type': np.array([[''], ['']], dtype=np.dtype('U50')),
                'neighbour_data_type': np.array([[''], ['']], dtype=np.dtype('U10')),
                'neighbour_value_long': np.array([[0], [0]], dtype=np.int),
                'neighbour_value_double': np.array([[0.0], [0.0]], dtype=np.int),
                'neighbour_value_boolean': np.array([[-1], [-1]], dtype=np.int),
                'neighbour_value_date': np.array([[''], ['']], dtype='datetime64[s]'),
                'neighbour_value_string': np.array([[''], ['']], dtype=np.dtype('U50'))
            }
        }

        np.testing.assert_equal(expected_arrays, initialised_arrays)


class TestDetermineValuesToPut(unittest.TestCase):

    def test__get_values_to_put_with_entity(self):
        role_label = 'employer'
        role_direction = neighbour.TARGET_PLAYS
        neighbour_type_label = 'company'
        neighbour_data_type = None
        neighbour_value = None
        values_dict = array._get_values_to_put(role_label, role_direction, neighbour_type_label,
                                               neighbour_data_type, neighbour_value)
        expected_result = {"role_type": 'employer',
                           'role_direction': role_direction,
                           'neighbour_type': 'company'
                           }
        self.assertEqual(values_dict, expected_result)

    def test__get_values_to_put_with_string_attribute(self):
        role_label = '@has-name-value'
        role_direction = neighbour.NEIGHBOUR_PLAYS
        neighbour_type_label = 'name'
        neighbour_data_type = 'string'
        neighbour_value = 'Person\'s Name'
        values_dict = array._get_values_to_put(role_label, role_direction, neighbour_type_label,
                                               neighbour_data_type, neighbour_value)
        expected_result = {"role_type": '@has-name-value',
                           'role_direction': role_direction,
                           'neighbour_type': 'name',
                           'neighbour_data_type': 'string',
                           'neighbour_value_string': neighbour_value}
        self.assertEqual(expected_result, values_dict)


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

        arr = array.fill_array_with_repeats(arr, (1, ..., slice(1), 0), (1, ..., slice(1, None), 0))
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

        arr = array.fill_array_with_repeats(arr, (1, ..., slice(1), 0), (1, ..., slice(1, None), 0))
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

        arr = array.fill_array_with_repeats(arr, ([1], ..., slice(1), [1], [0]), ([1], ..., slice(1, None), [1], [0]))
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

        arr = array.fill_array_with_repeats(arr, (0, ..., slice(2), 0, 0), (0, ..., slice(2, None), 0, 0))
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


class TestUpdateDepthwiseArraysWithNeighbour(unittest.TestCase):
    def setUp(self):
        self.current_indices = [0, 0]
        self.depth = 0
        context = builder.ThingContext(neighbour.Thing('1234', 'person', 'entity'), [])
        self.nb = builder.Neighbour('employee', 0, context)

        self.depthwise_arrays = [collections.OrderedDict(
            [('role_type', np.array([['']], dtype='U50')),
             ('role_direction', np.array([[0]], dtype=np.int)),
             ('neighbour_type', np.array([['']], dtype=np.dtype('U50'))),
             ('neighbour_data_type', np.array([['']], dtype=np.dtype('U10'))),
             ('neighbour_value_long', np.array([[0]], dtype=np.int)),
             ('neighbour_value_double', np.array([[0.0]], dtype=np.float)),
             ('neighbour_value_boolean', np.array([[-1]], dtype=np.int)),
             ('neighbour_value_date', np.array([['']], dtype='datetime64[s]')),
             ('neighbour_value_string', np.array([['']], dtype=np.dtype('U50')))])]

    def test_output_as_expected(self):
        expected_output = [collections.OrderedDict(
            [('role_type', np.array([['employee']], dtype='U50')),
             ('role_direction', np.array([[0]], dtype=np.int)),
             ('neighbour_type', np.array([['person']], dtype=np.dtype('U50'))),
             ('neighbour_data_type', np.array([['']], dtype=np.dtype('U10'))),
             ('neighbour_value_long', np.array([[0]], dtype=np.int)),
             ('neighbour_value_double', np.array([[0.0]], dtype=np.float)),
             ('neighbour_value_boolean', np.array([[-1]], dtype=np.int)),
             ('neighbour_value_date', np.array([['']], dtype='datetime64[s]')),
             ('neighbour_value_string', np.array([['']], dtype=np.dtype('U50')))])]
        array._update_depthwise_arrays_with_neighbour(self.depthwise_arrays, self.current_indices,
                                                                   self.depth, self.nb)
        np.testing.assert_array_equal(expected_output, self.depthwise_arrays)


class TestPutValuesIntoArray(unittest.TestCase):

    def setUp(self):
        self.values_to_put = collections.OrderedDict(
            [('role_type', 'employee'),
             ('role_direction', 0),
             ('neighbour_type', 'person'),
             ('neighbour_data_type', ''),
             ('neighbour_value_long', 0),
             ('neighbour_value_double', 0.0),
             ('neighbour_value_boolean', -1),
             ('neighbour_value_date', ''),
             ('neighbour_value_string', '')])

        self.current_indices = (0, 0)

        self.arrays_at_this_depth = collections.OrderedDict(
            [('role_type', np.array([['']], dtype='U50')),
             ('role_direction', np.array([[0]], dtype=np.int)),
             ('neighbour_type', np.array([['']], dtype=np.dtype('U50'))),
             ('neighbour_data_type', np.array([['']], dtype=np.dtype('U10'))),
             ('neighbour_value_long', np.array([[0]], dtype=np.int)),
             ('neighbour_value_double', np.array([[0.0]], dtype=np.float)),
             ('neighbour_value_boolean', np.array([[-1]], dtype=np.int)),
             ('neighbour_value_date', np.array([['']], dtype='datetime64[s]')),
             ('neighbour_value_string', np.array([['']], dtype=np.dtype('U50')))])

    def test_output_as_expected(self):
        expected_output = collections.OrderedDict(
            [('role_type', np.array([['employee']], dtype='U50')),
             ('role_direction', np.array([[0]], dtype=np.int)),
             ('neighbour_type', np.array([['person']], dtype=np.dtype('U50'))),
             ('neighbour_data_type', np.array([['']], dtype=np.dtype('U10'))),
             ('neighbour_value_long', np.array([[0]], dtype=np.int)),
             ('neighbour_value_double', np.array([[0.0]], dtype=np.float)),
             ('neighbour_value_boolean', np.array([[-1]], dtype=np.int)),
             ('neighbour_value_date', np.array([['']], dtype='datetime64[s]')),
             ('neighbour_value_string', np.array([['']], dtype=np.dtype('U50')))])

        output = array._put_values_into_array(self.arrays_at_this_depth, self.current_indices, self.values_to_put)
        np.testing.assert_array_equal(expected_output, output)

    def test_no_side_effects(self):
        output = array._put_values_into_array(self.arrays_at_this_depth, self.current_indices, self.values_to_put)
        self.assertNotEqual(id(self.arrays_at_this_depth), id(output))

    def test_input_unchanged(self):
        copy = self.arrays_at_this_depth.copy()
        output = array._put_values_into_array(self.arrays_at_this_depth, self.current_indices, self.values_to_put)
        np.testing.assert_array_equal(copy, self.arrays_at_this_depth)


if __name__ == "__main__":
    unittest.main()
