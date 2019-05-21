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

import collections
import numpy as np
import tensorflow as tf

import kglib.kgcn.core.ingest.encode.schema as se
import kglib.kgcn.core.ingest.traverse.data.context.array as array

schema_traversal = collections.OrderedDict((('animal', ['animal']),
                                            ('dog', ['animal', 'dog']),
                                            ('collie', ['animal', 'dog', 'collie']),
                                            ('fish', ['animal', 'fish']),
                                            ('border collie', ['animal', 'dog', 'collie', 'border collie']),
                                            ))


class TestBuildAdjacencyMatrix(unittest.TestCase):

    def test_input_output(self):
        print(schema_traversal.items())

        adj = se._build_adjacency_matrix(schema_traversal)

        with self.subTest('type adjacency matrix'):
            desired_output = np.identity(len(schema_traversal), dtype=np.int64)
            desired_output[:, 0] = 1
            desired_output[[2, 4], 1] = 1
            desired_output[4, 2] = 1

            print("output:")
            print(adj)
            np.testing.assert_array_equal(adj, desired_output)


class TestEncodeSchemaTypes(unittest.TestCase):
    def test_input_output(self):

        ###############################################################
        # Example data
        ###############################################################
        schema_types = ['animal', 'dog', 'collie', 'fish', 'border collie']

        multi_hot_embeddings = np.identity(len(schema_types), dtype=np.int64)
        multi_hot_embeddings[:, 0] = 1
        multi_hot_embeddings[[2, 4], 1] = 1
        multi_hot_embeddings[4, 2] = 1

        schema_type_features = ['dog', 'dog', 'border collie', 'animal', 'fish', 'fish']

        expected_type_indices = np.reshape(np.array([schema_types.index(label) for label in schema_type_features]),
                                           (2, 3, 1))

        expected_result = np.squeeze(np.take(multi_hot_embeddings, expected_type_indices, axis=0))

        schema_type_features = np.reshape(np.array(schema_type_features), (2, 3, 1))

        print(multi_hot_embeddings)
        print(schema_types)
        print("Look up:")
        print(schema_type_features)
        print("Expected result:")
        print(expected_result)

        ###############################################################
        # Test output
        ###############################################################
        tf.enable_eager_execution()
        encoder = se.MultiHotSchemaTypeEncoder(schema_traversal)
        embeddings = encoder(tf.convert_to_tensor(schema_type_features, dtype=tf.string))
        print("\nResult:")
        print(embeddings.numpy())
        with self.subTest("Embedding correctness"):
            np.testing.assert_array_equal(embeddings.numpy(), expected_result)

        with self.subTest("Embedding shape"):
            np.testing.assert_array_equal(embeddings.numpy().shape, (2, 3, 5))

        # Omitted since the type_indices are no longer returned along with the embeddings
        # with self.subTest("Type indices correctness"):
        #     np.testing.assert_array_equal(type_indices.numpy(), expected_type_indices)

    def test_encoding_schema_for_an_input_array_works_as_expected(self):
        array_shape = (3, 2, 1)
        example_arrays = array.initialise_arrays(array_shape, neighbour_type=('U25', 'collie'))

        example = np.array(example_arrays['neighbour_type'], dtype=str)
        tf.enable_eager_execution()
        encoder = se.MultiHotSchemaTypeEncoder(schema_traversal)
        embeddings = encoder(tf.convert_to_tensor(example, tf.string))

    def test_raise_exception_when_schema_traversal_empty(self):
        tf.enable_eager_execution()
        empty_schema_traversal = collections.OrderedDict()

        with self.assertRaises(ValueError) as context:
            se.MultiHotSchemaTypeEncoder(empty_schema_traversal)

        self.assertTrue('The schema traversal supplied cannot be empty' in str(context.exception))


if __name__ == "__main__":
    unittest.main()
