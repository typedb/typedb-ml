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

from unittest.mock import Mock, patch

from kglib.kgcn_tensorflow.models.attribute import CategoricalAttribute
import tensorflow as tf

from kglib.utils.test.utils import get_call_args


class TestCategoricalAttribute(tf.test.TestCase):

    def setUp(self):
        self._mock_embed_instance = Mock(return_value=tf.zeros((3, 1, 5), dtype=tf.float32))
        self._mock_embed_class = Mock(return_value=self._mock_embed_instance)
        self._patcher = patch('kglib.kgcn_tensorflow.models.attribute.snt.Embed', new=self._mock_embed_class,
                              spec=True)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_embed_invoked_correctly(self):
        attr_embedding_dim = 5
        cat = CategoricalAttribute(2, 5)
        cat(tf.zeros((3, 1), tf.float32))
        self._mock_embed_class.assert_called_once_with(2, attr_embedding_dim)

    def test_output_is_as_expected(self):
        inp = tf.zeros((3, 1), dtype=tf.float32)
        expected_output = tf.zeros((3, 5), dtype=tf.float32)
        cat = CategoricalAttribute(2, 5)
        output = cat(inp)
        self.assertAllClose(expected_output, output)
        self.assertEqual(expected_output.dtype, output.dtype)

    def test_embed_instance_called_correctly(self):
        inp = tf.zeros((3, 1), dtype=tf.float32)
        cat = CategoricalAttribute(2, 5)
        cat(inp)
        self.assertAllClose(get_call_args(self._mock_embed_instance), [[tf.zeros((3, 1), dtype=tf.int32)]])
        self.assertEqual(get_call_args(self._mock_embed_instance)[0][0].dtype, tf.int32)


if __name__ == "__main__":
    unittest.main()
