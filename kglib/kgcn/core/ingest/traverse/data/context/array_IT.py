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

from kglib.kgcn.core.ingest.traverse.data.context import builder as builder, neighbour as neighbour, array as array


class TestBuildingArraysFromContextBatch(unittest.TestCase):
    def test_outcome_is_as_expected(self):
        context_batch = [{
            1: [builder.Node((), neighbour.Thing("0", "person", "entity"))],
            0: [builder.Node((0,), neighbour.Thing("1", "name", "attribute", data_type='string', value='Sundar Pichai'),
                             "has", neighbour.NEIGHBOUR_PLAYS),
                builder.Node((1,), neighbour.Thing("2", "employment", "relation"), "employee", neighbour.TARGET_PLAYS),
                ]
        }]

        max_hops_shape = (1, 2, 1)

        filled_arrays = array.convert_context_batch_to_arrays(context_batch,
                                                              max_hops_shape,
                                                              role_type=(np.dtype('U50'), ''),
                                                              role_direction=(np.int, -1),
                                                              neighbour_type=(np.dtype('U50'), ''),
                                                              neighbour_data_type=(np.dtype('U10'), ''),
                                                              neighbour_value_long=(np.int, 0),
                                                              neighbour_value_double=(np.float, 0.0),
                                                              neighbour_value_boolean=(np.int, -1),
                                                              neighbour_value_date=('datetime64[s]', ''),
                                                              neighbour_value_string=(np.dtype('U50'), ''))

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
                'role_type': np.array([['']], dtype=np.dtype('U50')),
                'role_direction': np.array([[-1]], dtype=np.int),
                'neighbour_type': np.array([['person']], dtype=np.dtype('U50')),
                'neighbour_data_type': np.array([['']], dtype=np.dtype('U10')),
                'neighbour_value_long': np.array([[0]], dtype=np.int),
                'neighbour_value_double': np.array([[0.0]], dtype=np.int),
                'neighbour_value_boolean': np.array([[-1]], dtype=np.int),
                'neighbour_value_date': np.array([['']], dtype='datetime64[s]'),
                'neighbour_value_string': np.array([['']], dtype=np.dtype('U50'))
            }
        ]

        np.testing.assert_equal(filled_arrays, expected_filled_arrays)


if __name__ == "__main__":
    unittest.main()
