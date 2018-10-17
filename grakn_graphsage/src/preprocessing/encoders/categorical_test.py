import unittest
import tensorflow as tf
import numpy as np

import grakn_graphsage.src.preprocessing.encoders.categorical as cat


class TestIndicesFromCategories(unittest.TestCase):
    def test_input_output(self):
        tf.enable_eager_execution()
        categories = np.array(['cat', 'dog', 'sheep', 'cow', 'collie', 'animal'], dtype='U5')

        raw_features = np.full((2, 3, 2, 1), '', dtype='U5')
        raw_features[0, 0, 0, 0] = 'dog'
        raw_features[0, 2, 1, 0] = 'cow'
        raw_features[1, 1, 1, 0] = 'cat'
        print(raw_features)

        # check output

        # output[0, 0, 0, 0] = 1
        # output[0, 2, 1, 0] = 3
        # output[1, 1, 1, 0] = 0
        # output[0, 0, 0, 1] = -1
        output = cat.indices_from_categories(tf.convert_to_tensor(raw_features, dtype=tf.string),
                                             tf.convert_to_tensor(categories, dtype=tf.string))

        desired_output = np.array([[[[1], [-1]],
                                    [[-1], [-1]],
                                    [[-1], [3]]],

                                   [[[-1], [-1]],
                                    [[-1], [0]],
                                    [[-1], [-1]]]], dtype=np.int32)

        print(output)
        np.testing.assert_array_equal(output.numpy(), desired_output)
