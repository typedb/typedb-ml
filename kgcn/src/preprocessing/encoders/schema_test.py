import unittest

import tensorflow as tf
import numpy as np

import kgcn.src.preprocessing.encoders.schema as se
import collections

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
        embeddings, type_indices = encoder(tf.convert_to_tensor(schema_type_features, dtype=tf.string))
        print("\nResult:")
        print(embeddings.numpy())
        with self.subTest("Embedding correctness"):
            np.testing.assert_array_equal(embeddings.numpy(), expected_result)

        with self.subTest("Embedding shape"):
            np.testing.assert_array_equal(embeddings.numpy().shape, (2, 3, 5))

        with self.subTest("Type indices correctness"):
            np.testing.assert_array_equal(type_indices.numpy(), expected_type_indices)
