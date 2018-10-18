import unittest

import tensorflow as tf
import numpy as np

import kgcn.src.preprocessing.encoders.schema as se


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

        schema_type_features = ['dog', 'dog', 'border collie', 'animal', 'fish', 'fish', 'fish']

        expected_result = np.empty((len(schema_type_features), len(schema_types)))
        for i, schema_type_feature in enumerate(schema_type_features):
            j = schema_types.index(schema_type_feature)
            expected_result[i, :] = multi_hot_embeddings[j, :]

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
        encoder = se.SchemaTypeEncoder(tf.convert_to_tensor(schema_types, dtype=tf.string),
                                       tf.convert_to_tensor(multi_hot_embeddings, dtype=tf.int64))
        result = encoder(tf.convert_to_tensor(schema_type_features, dtype=tf.string))
        print("\nResult:")
        print(result.numpy())
        np.testing.assert_array_equal(result.numpy(), expected_result)
