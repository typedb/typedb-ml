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

from mock import Mock, patch

from kglib.kgcn_experimental.network.attribute import CategoricalAttribute
import tensorflow as tf


class TestCategoricalAttribute(unittest.TestCase):

    def setUp(self):
        self._mock_embed_instance = Mock()
        self._mock_embed_class = Mock(return_value=self._mock_embed_instance)
        self._patcher = patch('kglib.kgcn_experimental.network.attribute.snt.Embed', new=self._mock_embed_class,
                              spec=True)
        self._patcher.start()

        self._mock_cast = Mock()
        self._cast_patcher = patch('kglib.kgcn_experimental.network.attribute.tf.cast', new=self._mock_cast, spec=True)
        self._cast_patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self._cast_patcher.stop()

    def test_embed_invoked_correctly(self):
        attr_embedding_dim = 5
        cat = CategoricalAttribute([0, 1, 2], 5)
        cat(Mock())
        self._mock_embed_class.assert_called_once_with(3, attr_embedding_dim)

    def test_output_is_as_expected(self):

        output_mock = Mock()
        self._mock_embed_instance.return_value = output_mock

        cat = CategoricalAttribute([0, 1, 2], 5)
        output = cat(Mock())
        self.assertEqual(output_mock, output)

    def test_embed_instance_called_with_input(self):
        inp = tf.zeros((3, 1), dtype=tf.float32)
        casted_inp = Mock()
        self._mock_cast.return_value = casted_inp

        cat = CategoricalAttribute([0, 1, 2], 5)
        cat(inp)

        self._mock_cast.assert_called_once_with(inp, dtype=tf.int32)
        self._mock_embed_instance.assert_called_once_with(casted_inp)


if __name__ == "__main__":
    unittest.main()
