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
import sonnet as snt
import tensorflow as tf
from mock import Mock

from kglib.kgcn_experimental.test.utils import get_call_args


class TypewiseEncoder(snt.AbstractModule):
    """
    Orchestrates encoding elements according to their types. Defers encoding of each feature to the appropriate encoder
    for the type of that feature. Assumes that the type is given categorically as an integer value in the 0th position
    of the provided features Tensor.
    """
    def __init__(self, encoders_for_types, feature_length, name="typewise_encoder"):
        """
        Args:
            encoders_for_types: Dict - keys: encoders; values: a list of type categories the encoder should be used for
            feature_length: The length of features to output for matrix initialisation
            name: The name for this Module
        """
        super(TypewiseEncoder, self).__init__(name=name)

        types_considered = []
        for a in encoders_for_types.values():
            types_considered.extend(a)
        types_considered.sort()

        expected_types = list(range(max(types_considered) + 1))

        if types_considered != expected_types:
            raise ValueError(
                f'Encoder categories are inconsistent. Expected {expected_types}, but got {types_considered}')

        self._feature_length = feature_length
        self._encoders_for_types = encoders_for_types

    def _build(self, features):

        shape = tf.stack([tf.shape(features)[0], self._feature_length])

        encoded_features = tf.zeros(shape, dtype=tf.float32)

        for encoder, types in self._encoders_for_types.items():

            feat_types = tf.cast(features[:, 0], tf.int32)  # The types for each feature, as integers

            # Expand dimensions ready for element-wise equality comparison
            exp_types = tf.expand_dims(types, axis=0)
            exp_feat_types = tf.expand_dims(feat_types, axis=1)

            elementwise_equality = tf.equal(exp_feat_types, exp_types)

            # Use this encoder when the feat_type matches any of the types
            applicable_types_mask = tf.reduce_any(elementwise_equality, axis=1)
            indices_to_encode = tf.where(applicable_types_mask)
            feats_to_encode = tf.squeeze(tf.gather(features[:, 1:], indices_to_encode), axis=1)
            encoded_feats = encoder()(feats_to_encode)

            encoded_features += tf.scatter_nd(tf.cast(indices_to_encode, dtype=tf.int32), encoded_feats, shape)

        return encoded_features


class TestAttributeEncoder(unittest.TestCase):
    def test_attribute_encoding_stages_are_as_expected(self):

        def op_mock():
            return Mock(return_value=np.array([0.121, 1.621, 1.437, -0.194, -0.216], dtype=np.float32))

        def attr_mock():
            return Mock(return_value=np.array([0.22632198, 0.29790161, 0.44993045], dtype=np.float32))

        encode = AttributeEncoder(5, 0, op=op_mock, attr_op=attr_mock)
        encoding = encode(np.array([2, 0.1234], dtype=np.float32))

        op_mock_call_args = get_call_args(op_mock)
        expected_intermediate_encoding = np.array([0, 0, 1, 0, 0, 0.22632198, 0.29790161, 0.44993045], dtype=np.float32)
        np.testing.assert_array_equal(op_mock_call_args, [[expected_intermediate_encoding]])

        attr_mock_call_args = get_call_args(attr_mock)
        expected_attribute_value = np.array([0.1234], dtype=np.float32)
        np.testing.assert_array_equal(attr_mock_call_args, [[expected_attribute_value]])

        expected_encoding = np.array([0.121, 1.621, 1.437, -0.194, -0.216], dtype=np.float32)
        np.testing.assert_array_equal(expected_encoding, encoding)