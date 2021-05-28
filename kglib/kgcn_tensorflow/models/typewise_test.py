#
#  Copyright (C) 2021 Vaticle
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
from unittest.mock import Mock

from kglib.utils.test.utils import get_call_args
from kglib.kgcn_tensorflow.models.typewise import TypewiseEncoder


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
