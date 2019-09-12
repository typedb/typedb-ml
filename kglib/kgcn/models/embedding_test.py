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
from unittest.mock import patch
from kglib.kgcn.models.embedding import common_embedding, attribute_embedding, node_embedding
from kglib.utils.test.utils import get_call_args


class TestCommonEmbedding(unittest.TestCase):
    def test_embedding_output_shape_as_expected(self):
        tf.enable_eager_execution()
        features = np.array([[1, 0, 0.7], [1, 2, 0.7], [0, 1, 0.5]], dtype=np.float32)
        type_embedding_dim = 5
        output = common_embedding(features, 3, type_embedding_dim)

        np.testing.assert_array_equal(np.array([3, 6]), output.shape)


class TestAttributeEmbedding(unittest.TestCase):

    def test_embedding_is_typewise(self):
        features = np.array([[1, 0, 0.7], [1, 2, 0.7], [0, 1, 0.5]])

        mock_instance = Mock()
        mock = Mock(return_value=mock_instance)
        patcher = patch('kglib.kgcn.models.embedding.TypewiseEncoder', spec=True, new=mock)
        mock_class = patcher.start()

        attr_encoders = Mock()
        attr_embedding_dim = Mock()

        attribute_embedding(features, attr_encoders, attr_embedding_dim)  # Function under test

        mock_class.assert_called_once_with(attr_encoders, attr_embedding_dim)
        call_args = get_call_args(mock_instance)

        np.testing.assert_array_equal([[np.array([[0, 0.7], [2, 0.7], [1, 0.5]])]], call_args)

        patcher.stop()


class TestNodeEmbedding(unittest.TestCase):

    def test_embedding_is_typewise(self):
        tf.enable_eager_execution()

        features = Mock()
        num_types = Mock()
        type_embedding_dim = Mock()
        attr_encoders = Mock()
        attr_embedding_dim = Mock()

        mock_attribute_embedding = Mock(return_value=np.ones((3, 5)))

        mock_common_embedding = Mock(return_value=np.ones((3, 4)))

        patcher_attr = patch('kglib.kgcn.models.embedding.attribute_embedding', spec=True,
                             new=mock_attribute_embedding)
        patcher_attr.start()

        patcher_common = patch('kglib.kgcn.models.embedding.common_embedding', spec=True,
                               new=mock_common_embedding)
        patcher_common.start()

        embedding = node_embedding(features, num_types, type_embedding_dim, attr_encoders, attr_embedding_dim)

        np.testing.assert_array_equal(np.ones((3, 9)), embedding.numpy())

        patcher_attr.stop()
        patcher_common.stop()


if __name__ == "__main__":
    unittest.main()
