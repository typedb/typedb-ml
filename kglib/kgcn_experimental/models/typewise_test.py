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
import tensorflow as tf
from mock import Mock

from kglib.utils.test.utils import get_call_args
from kglib.kgcn_experimental.models.typewise import TypewiseEncoder


class TestTypewiseEncoder(unittest.TestCase):
    def setUp(self):
        tf.enable_eager_execution()

    def test_types_encoded_by_expected_functions(self):
        things = np.array([[0, 0], [1, 0], [2, 0.5673]], dtype=np.float32)

        mock_entity_relation_encoder = Mock(return_value=np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float32))

        mock_attribute_encoder = Mock(return_value=np.array([[0.9527, 0.2367, 0.7582]], dtype=np.float32))

        encoders_for_types = {lambda: mock_entity_relation_encoder: [0, 1], lambda: mock_attribute_encoder: [2]}

        tm = TypewiseEncoder(encoders_for_types, 3)
        encoding = tm(things)  # The function under test

        np.testing.assert_array_equal([[np.array([[0], [0]], dtype=np.float32)]],
                                      get_call_args(mock_entity_relation_encoder))

        np.testing.assert_array_equal([[np.array([[0.5673]], dtype=np.float32)]], get_call_args(mock_attribute_encoder))

        expected_encoding = np.array([[0, 0, 0], [0, 0, 0], [0.9527, 0.2367, 0.7582]], dtype=np.float32)
        np.testing.assert_array_equal(expected_encoding, encoding.numpy())

    def test_basic_encoding(self):
        things = np.array([[0], [1], [2]], dtype=np.float32)

        mock_entity_relation_encoder = Mock(return_value=np.array([[0.1, 0, 0], [0.1, 0, 0], [0.1, 0, 0]], dtype=np.float32))

        encoders_for_types = {lambda: mock_entity_relation_encoder: [0, 1, 2]}

        tm = TypewiseEncoder(encoders_for_types, 3)
        encoding = tm(things)  # The function under test

        expected_encoding = np.array([[0.1, 0, 0], [0.1, 0, 0], [0.1, 0, 0]], dtype=np.float32)
        np.testing.assert_array_equal(expected_encoding, encoding.numpy())

    def test_encoders_do_not_fulfil_classes(self):
        mock_entity_relation_encoder = Mock()

        encoders_for_types = {lambda: mock_entity_relation_encoder: [0, 2]}

        with self.assertRaises(ValueError) as context:
            TypewiseEncoder(encoders_for_types, 3)

        self.assertEqual('Encoder categories are inconsistent. Expected [0, 1, 2], but got [0, 2]',
                         str(context.exception))


if __name__ == '__main__':
    unittest.main()


class TestTensorScatterAdd(unittest.TestCase):

    def test_with_eager_execution(self):
        tf.enable_eager_execution()
        encoded_features = tf.zeros((2, 15), dtype=tf.float32)

        indices_to_encode = tf.convert_to_tensor(np.array([[0], [1]], dtype=np.int64))

        encoded_feats = tf.convert_to_tensor(np.array(
            [[-0.8854138, -0.8854138, -0.28424942, -0.8854138, -0.8854138, 0.15186566,
              -0.8854138, -0.09432995, 0.48828167, - 0.8854138, 1.7825887, 1.1054695,
              1.7577922, -0.8854138, 1.2904773],
             [-0.5648398, -0.5648398, -0.5648398, 0.11074328, -0.5648398, -0.5648398,
              -0.5648398, 1.2660778, -0.5648398, -0.5648398, 1.8472059, -0.5648398,
              -0.5648398, 2.5975735, -0.17320272]], dtype=np.float32))
        encoded_features = tf.tensor_scatter_add(encoded_features, indices_to_encode, encoded_feats)
        print(encoded_features)
        tf.disable_eager_execution()

    def test_with_session(self):

        encoded_features = np.zeros((2, 15), dtype=np.float32)

        indices_to_encode = np.array([[0]], dtype=np.int64)

        encoded_feats = np.array(
            [[-0.8854138, -0.8854138, -0.28424942, -0.8854138, -0.8854138, 0.15186566,
              -0.8854138, -0.09432995, 0.48828167, - 0.8854138, 1.7825887, 1.1054695,
              1.7577922, -0.8854138, 1.2904773],
             ], dtype=np.float32)

        encoded_features_ph = tf.placeholder(tf.float32, (2, 15))

        indices_to_encode_ph = tf.placeholder(tf.int64, indices_to_encode.shape)

        encoded_feats_ph = tf.placeholder(tf.float32, encoded_feats.shape)

        output_op = tf.tensor_scatter_add(encoded_features_ph, indices_to_encode_ph, encoded_feats_ph)

        feed_dict = {encoded_features_ph: encoded_features,
                     indices_to_encode_ph: indices_to_encode,
                     encoded_feats_ph: encoded_feats}

        with tf.Session() as sess:
            result = sess.run({"output": output_op}, feed_dict=feed_dict)
            print(result)