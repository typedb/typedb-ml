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

import kglib.kgcn.core.ingest.traverse.data.context.array as array
from kglib.kgcn.core.ingest.traverse.data.context import builder as builder, neighbour as neighbour

"""
Expected procedure:
- Take in context as a dict of lists, for a single example
- Convert to the values to put into the arrays, against the index to add at
- Initialise the arrays at different depths, for the data categories with default values
- Put the values for the example into the initialised array, without repetition (for simplicity, since this shouldn't 
have a serious effect on the model
- Combine with arrays to represent the list of examples
"""


def mock_context():
    return {
        2: [builder.Node((), neighbour.Thing("0", "person", "entity"))],
        1: [builder.Node((0,), neighbour.Thing("1", "name", "attribute", data_type='string', value='Sundar Pichai'),
                         "has", neighbour.NEIGHBOUR_PLAYS),
            builder.Node((1,), neighbour.Thing("2", "employment", "relation"), "employee", neighbour.TARGET_PLAYS),
            ],
        0: [builder.Node((0, 0), neighbour.Thing("0", "person", "entity"), "has", neighbour.TARGET_PLAYS),
            # Note that (0, 1) is reversed compared to the natural expectation
            builder.Node((0, 1), neighbour.Thing("3", "company", "entity"), "employer", neighbour.NEIGHBOUR_PLAYS),
            builder.Node((1, 1), neighbour.Thing("0", "person", "entity"), "employee", neighbour.NEIGHBOUR_PLAYS),
            ]
    }


class TestContextToIndexedValues(unittest.TestCase):

    def test_context_to_indexed_values_is_as_expected(self):
        context = mock_context()
        expected_indexed_values = {
            2: {(): {'neighbour_type': "person"}},
            1: {(0,): {'role_type': "has", 'role_direction': neighbour.NEIGHBOUR_PLAYS, 'neighbour_type': "name",
                       'neighbour_data_type': "string", 'neighbour_value_string': "Sundar Pichai"},
                (1,): {'role_type': "employee", 'role_direction': neighbour.TARGET_PLAYS,
                       'neighbour_type': "employment"},
                },
            0: {(0, 0): {'role_type': "has", 'role_direction': neighbour.TARGET_PLAYS, 'neighbour_type': "person"},
                (0, 1): {'role_type': "employer", 'role_direction': neighbour.NEIGHBOUR_PLAYS,
                         'neighbour_type': "company"},
                (1, 1): {'role_type': "employee", 'role_direction': neighbour.NEIGHBOUR_PLAYS,
                         'neighbour_type': "person"},
                },
        }
        vals = array.get_context_values_to_put(context)
        self.assertEqual(expected_indexed_values, vals)


class TestBatchIndexedValues(unittest.TestCase):
    def test_outcome_is_as_expected(self):
        indexed_values = {
                1: {(): {'neighbour_type': "person"}},
                0: {(0,): {'role_type': "has", 'role_direction': neighbour.NEIGHBOUR_PLAYS, 'neighbour_type': "name",
                           'neighbour_data_type': "string", 'neighbour_value_string': "Sundar Pichai"},
                    (1,): {'role_type': "employee", 'role_direction': neighbour.TARGET_PLAYS,
                           'neighbour_type': "employment"},
                    }
            }
        batch_indexed_values = [indexed_values] * 2

        batched = array.batch_values_to_put(batch_indexed_values)

        expected_result = {
                1: {(0,): {'neighbour_type': "person"},
                    (1,): {'neighbour_type': "person"}},
                0: {(0, 0,): {'role_type': "has", 'role_direction': neighbour.NEIGHBOUR_PLAYS, 'neighbour_type': "name",
                           'neighbour_data_type': "string", 'neighbour_value_string': "Sundar Pichai"},
                    (0, 1,): {'role_type': "employee", 'role_direction': neighbour.TARGET_PLAYS,
                           'neighbour_type': "employment"},
                    (1, 0,): {'role_type': "has", 'role_direction': neighbour.NEIGHBOUR_PLAYS, 'neighbour_type': "name",
                           'neighbour_data_type': "string", 'neighbour_value_string': "Sundar Pichai"},
                    (1, 1,): {'role_type': "employee", 'role_direction': neighbour.TARGET_PLAYS,
                           'neighbour_type': "employment"},
                    }
            }
        self.assertEqual(expected_result, batched)


class TestInitialiseArraysWithDefaultValues(unittest.TestCase):

    def test_arrays_are_initialised_as_expected(self):
        array_shape = (2, 1)

        initialised_arrays = array.initialise_arrays(array_shape,
                                                     role_type=(np.dtype('U50'), ''),
                                                     role_direction=(np.int, -1),
                                                     neighbour_type=(np.dtype('U50'), ''),
                                                     neighbour_data_type=(np.dtype('U10'), ''),
                                                     neighbour_value_long=(np.int, 0),
                                                     neighbour_value_double=(np.float, 0.0),
                                                     neighbour_value_boolean=(np.int, -1),
                                                     neighbour_value_date=('datetime64[s]', ''),
                                                     neighbour_value_string=(np.dtype('U50'), ''))

        expected_arrays_1_hop = {
            'role_type': np.array([[''], ['']], dtype=np.dtype('U50')),
            'role_direction': np.array([[-1], [-1]], dtype=np.int),
            'neighbour_type': np.array([[''], ['']], dtype=np.dtype('U50')),
            'neighbour_data_type': np.array([[''], ['']], dtype=np.dtype('U10')),
            'neighbour_value_long': np.array([[0], [0]], dtype=np.int),
            'neighbour_value_double': np.array([[0.0], [0.0]], dtype=np.int),
            'neighbour_value_boolean': np.array([[-1], [-1]], dtype=np.int),
            'neighbour_value_date': np.array([[''], ['']], dtype='datetime64[s]'),
            'neighbour_value_string': np.array([[''], ['']], dtype=np.dtype('U50'))
        }
        np.testing.assert_equal(expected_arrays_1_hop, initialised_arrays)

    def test_batch_arrays_are_initialised_as_expected(self):
        # Exactly the same as the previous test, but for our use-case where we use the first index to indicate the
        # example of the batch
        array_shape = (1, 2, 1)

        initialised_arrays = array.initialise_arrays(array_shape,
                                                     role_type=(np.dtype('U50'), ''),
                                                     role_direction=(np.int, -1),
                                                     neighbour_type=(np.dtype('U50'), ''),
                                                     neighbour_data_type=(np.dtype('U10'), ''),
                                                     neighbour_value_long=(np.int, 0),
                                                     neighbour_value_double=(np.float, 0.0),
                                                     neighbour_value_boolean=(np.int, -1),
                                                     neighbour_value_date=('datetime64[s]', ''),
                                                     neighbour_value_string=(np.dtype('U50'), ''))

        expected_arrays_1_hop = {
            'role_type': np.array([[[''], ['']]], dtype=np.dtype('U50')),
            'role_direction': np.array([[[-1], [-1]]], dtype=np.int),
            'neighbour_type': np.array([[[''], ['']]], dtype=np.dtype('U50')),
            'neighbour_data_type': np.array([[[''], ['']]], dtype=np.dtype('U10')),
            'neighbour_value_long': np.array([[[0], [0]]], dtype=np.int),
            'neighbour_value_double': np.array([[[0.0], [0.0]]], dtype=np.int),
            'neighbour_value_boolean': np.array([[[-1], [-1]]], dtype=np.int),
            'neighbour_value_date': np.array([[[''], ['']]], dtype='datetime64[s]'),
            'neighbour_value_string': np.array([[[''], ['']]], dtype=np.dtype('U50'))
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

    def test_exception_throw_if_all_types_are_omitted(self):
        array_shape = (2, 1)

        with self.assertRaises(ValueError) as context:
            array.initialise_arrays(array_shape)

        self.assertEqual('At least one array dtype and default value must be provided', str(context.exception))


class TestDepthArraySizes(unittest.TestCase):
    def test_batch_array_sizes_determined_correctly(self):
        max_hops_shape = (2, 3, 2, 1)

        depth_array_shapes = array.get_depth_array_sizes(max_hops_shape)

        expected_depth_array_shapes = [(2, 3, 2, 1), (2, 2, 1), (2, 1)]
        self.assertListEqual(expected_depth_array_shapes, depth_array_shapes)


class TestInitialiseArraysAtAllDepthsWithDefaultValues(unittest.TestCase):

    def test_arrays_are_initialised_at_all_depths_as_expected(self):
        max_hops_shape = (1, 2, 1)
        initialised_arrays = array.initialise_arrays_for_all_depths(max_hops_shape,
                                                                    role_type=(np.dtype('U50'), ''),
                                                                    role_direction=(np.int, 0),
                                                                    neighbour_type=(np.dtype('U50'), ''),
                                                                    neighbour_data_type=(np.dtype('U10'), ''),
                                                                    neighbour_value_long=(np.int, 0),
                                                                    neighbour_value_double=(np.float, 0.0),
                                                                    neighbour_value_boolean=(np.int, -1),
                                                                    neighbour_value_date=('datetime64[s]', ''),
                                                                    neighbour_value_string=(np.dtype('U50'), ''))

        expected_arrays = [
            {
                'role_type': np.array([[[''], ['']]], dtype=np.dtype('U50')),
                'role_direction': np.array([[[0], [0]]], dtype=np.int),
                'neighbour_type': np.array([[[''], ['']]], dtype=np.dtype('U50')),
                'neighbour_data_type': np.array([[[''], ['']]], dtype=np.dtype('U10')),
                'neighbour_value_long': np.array([[[0], [0]]], dtype=np.int),
                'neighbour_value_double': np.array([[[0.0], [0.0]]], dtype=np.int),
                'neighbour_value_boolean': np.array([[[-1], [-1]]], dtype=np.int),
                'neighbour_value_date': np.array([[[''], ['']]], dtype='datetime64[s]'),
                'neighbour_value_string': np.array([[[''], ['']]], dtype=np.dtype('U50'))
            },
            {
                'neighbour_type': np.array([['']], dtype=np.dtype('U50')),
                'neighbour_data_type': np.array([['']], dtype=np.dtype('U10')),
                'neighbour_value_long': np.array([[0]], dtype=np.int),
                'neighbour_value_double': np.array([[0.0]], dtype=np.int),
                'neighbour_value_boolean': np.array([[-1]], dtype=np.int),
                'neighbour_value_date': np.array([['']], dtype='datetime64[s]'),
                'neighbour_value_string': np.array([['']], dtype=np.dtype('U50'))
            }
        ]

        np.testing.assert_equal(expected_arrays, initialised_arrays)


class TestFillingArrays(unittest.TestCase):
    def test_array_filled_as_expected(self):
        initialised_arrays = [
            {
                'role_type': np.full((1, 2, 1), fill_value='', dtype=np.dtype('U50')),
                'role_direction': np.full((1, 2, 1), fill_value=-1, dtype=np.int),
                'neighbour_type': np.full((1, 2, 1), fill_value='', dtype=np.dtype('U50')),
                'neighbour_data_type': np.full((1, 2, 1), fill_value='', dtype=np.dtype('U10')),
                'neighbour_value_long': np.full((1, 2, 1), fill_value=0, dtype=np.int),
                'neighbour_value_double': np.full((1, 2, 1), fill_value=0.0, dtype=np.int),
                'neighbour_value_boolean': np.full((1, 2, 1), fill_value=-1, dtype=np.int),
                'neighbour_value_date': np.full((1, 2, 1), fill_value='', dtype='datetime64[s]'),
                'neighbour_value_string': np.full((1, 2, 1), fill_value='', dtype=np.dtype('U50'))
            },
            {
                'neighbour_type': np.full((1, 1), fill_value='', dtype=np.dtype('U50')),
                'neighbour_data_type': np.full((1, 1), fill_value='', dtype=np.dtype('U10')),
                'neighbour_value_long': np.full((1, 1), fill_value=0, dtype=np.int),
                'neighbour_value_double': np.full((1, 1), fill_value=0.0, dtype=np.int),
                'neighbour_value_boolean': np.full((1, 1), fill_value=-1, dtype=np.int),
                'neighbour_value_date': np.full((1, 1), fill_value='', dtype='datetime64[s]'),
                'neighbour_value_string': np.full((1, 1), fill_value='', dtype=np.dtype('U50'))
            }
        ]

        batch_values = {
            1: {(0,): {'neighbour_type': "person"},
                },
            0: {(0, 0): {'role_type': "has", 'role_direction': neighbour.NEIGHBOUR_PLAYS, 'neighbour_type': "name",
                         'neighbour_data_type': "string", 'neighbour_value_string': "Sundar Pichai"},
                (0, 1): {'role_type': "employee", 'role_direction': neighbour.TARGET_PLAYS,
                         'neighbour_type': "employment"},
                }
        }

        expected_filled_arrays = [
            {
                'role_type': np.array([[['has'], ['employee']]], dtype=np.dtype('U50')),
                'role_direction': np.array([[[neighbour.NEIGHBOUR_PLAYS], [neighbour.TARGET_PLAYS]]], dtype=np.int),
                'neighbour_type': np.array([[['name'], ['employment']]], dtype=np.dtype('U50')),
                'neighbour_data_type': np.array([[['string'], ['']]], dtype=np.dtype('U10')),
                'neighbour_value_long': np.array([[[0], [0]]], dtype=np.int),
                'neighbour_value_double': np.array([[[0.0], [0.0]]], dtype=np.int),
                'neighbour_value_boolean': np.array([[[-1], [-1]]], dtype=np.int),
                'neighbour_value_date': np.array([[[''], ['']]], dtype='datetime64[s]'),
                'neighbour_value_string': np.array([[['Sundar Pichai'], ['']]], dtype=np.dtype('U50'))
            },
            {
                'neighbour_type': np.array([['person']], dtype=np.dtype('U50')),
                'neighbour_data_type': np.array([['']], dtype=np.dtype('U10')),
                'neighbour_value_long': np.array([[0]], dtype=np.int),
                'neighbour_value_double': np.array([[0.0]], dtype=np.int),
                'neighbour_value_boolean': np.array([[-1]], dtype=np.int),
                'neighbour_value_date': np.array([['']], dtype='datetime64[s]'),
                'neighbour_value_string': np.array([['']], dtype=np.dtype('U50'))
            }
        ]

        filled_arrays = array.fill_arrays_at_all_depths(initialised_arrays, batch_values)
        np.testing.assert_equal(filled_arrays, expected_filled_arrays)

    def test_array_filled_as_expected_2_hop(self):

        initialised_arrays = [
            {
                'neighbour_type': np.full((1, 3, 2, 1), fill_value='', dtype=np.dtype('U50')),
            },
            {
                'neighbour_type': np.full((1, 2, 1), fill_value='', dtype=np.dtype('U50')),
            },
            {
                'neighbour_type': np.full((1, 1), fill_value='', dtype=np.dtype('U50')),
            }
        ]

        batch_values = {
            0: {
                (0, 0, 0): {'neighbour_type': "a"},
                (0, 1, 0): {'neighbour_type': "a"},
                (0, 2, 0): {'neighbour_type': "a"},
                (0, 0, 1): {'neighbour_type': "a"},
                (0, 1, 1): {'neighbour_type': "a"},
                (0, 2, 1): {'neighbour_type': "a"},
                },
            1: {(0, 0,): {'neighbour_type': "name"},
                (0, 1,): {'neighbour_type': "employment"},
                },
            2: {(0,): {'neighbour_type': "person"},
                }

        }

        expected_filled_arrays = [
            {
                'neighbour_type': np.array([[[['a'], ['a']], [['a'], ['a']], [['a'], ['a']]]], dtype=np.dtype('U50')),
            },
            {
                'neighbour_type': np.array([[['name'], ['employment']]], dtype=np.dtype('U50')),
            },
            {
                'neighbour_type': np.array([['person']], dtype=np.dtype('U50')),
            }
        ]

        filled_arrays = array.fill_arrays_at_all_depths(initialised_arrays, batch_values)
        np.testing.assert_equal(filled_arrays, expected_filled_arrays)


if __name__ == "__main__":
    unittest.main()
