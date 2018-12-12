import unittest

import numpy as np

import kgcn.src.encoder.tf_hub as tf_hub
import tensorflow as tf


def check_shape(test: unittest.TestCase, array, result, expected_vector_length):
    with test.subTest('test output shape'):
        expected_shape = list(array.shape)
        expected_shape[-1] = expected_vector_length
        expected_shape = tuple(expected_shape)
        actual_shape = result.shape
        test.assertTupleEqual(expected_shape, actual_shape)


class TestTFHub(unittest.TestCase):
    def setUp(self):
        self._arr = np.array(["A long sentence.",
                              "single-word",
                              "http://example.com",
                              "Accipitridae",
                              "Haliaeetus",
                              "North America",
                              "GB",
                              "SG",
                              "specimens",
                              "S",
                              "W",
                              "Oryx dammah",
                              "trophies",
                              "seeds",
                              "Sclerocactus papyracanthus",
                              "kg",
                              "ml",
                              ""])
        self._arr = self._arr[..., np.newaxis]

    def test_1D_array(self):

        with tf.Graph().as_default():
            encoder = tf_hub.TensorFlowHubEncoder("https://tfhub.dev/google/nnlm-en-dim128-with-normalization/1", 128)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.tables_initializer())
                result = encoder(tf.convert_to_tensor(self._arr))
                sess.run(result)
                print(result.eval().shape)
                check_shape(self, self._arr, result.eval(), 128)

    def test_depth_array(self):

        arr = np.reshape(self._arr, (3, 3, 2, 1))

        with tf.Graph().as_default():
            encoder = tf_hub.TensorFlowHubEncoder("https://tfhub.dev/google/nnlm-en-dim128-with-normalization/1", 128)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.tables_initializer())
                result = encoder(tf.convert_to_tensor(arr))
                sess.run(result)
                print(result.eval().shape)
                check_shape(self, arr, result.eval(), 128)

