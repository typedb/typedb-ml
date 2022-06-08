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

from kglib.kgcn_tensorflow.models.embedding import construct_categorical_embedders, construct_continuous_embedders, \
    construct_non_attribute_embedders


def construct_embedders(node_types, attr_embedding_dim, categorical_attributes, continuous_attributes):
    attr_embedders = dict()

    if categorical_attributes is not None:
        attr_embedders.update(construct_categorical_embedders(node_types, attr_embedding_dim, categorical_attributes))

    if continuous_attributes is not None:
        attr_embedders.update(construct_continuous_embedders(node_types, attr_embedding_dim, continuous_attributes))

    attr_embedders.update(construct_non_attribute_embedders(node_types, attr_embedding_dim, categorical_attributes,
                                                            continuous_attributes))
    return attr_embedders


class TestConstructingEmbedders(unittest.TestCase):

    def test_all_types_encoded(self):
        node_types = ['a', 'b', 'c']
        attr_embedding_dim = 5
        categorical_attributes = {'a': ['option1', 'option2']}
        continuous_attributes = {'b': (0, 1)}

        attr_embedders = construct_embedders(node_types, attr_embedding_dim, categorical_attributes,
                                             continuous_attributes)
        all_types = [l for el in list(attr_embedders.values()) for l in el]

        expected_types = [0, 1, 2]

        self.assertListEqual(expected_types, all_types)

    def test_multiple_categorical_embedders(self):
        node_types = ['a', 'b', 'c']
        attr_embedding_dim = 5
        categorical_attributes = {'a': ['option1', 'option2'], 'c': ['option3', 'option4']}
        continuous_attributes = {'b': (0, 1)}

        attr_embedders = construct_embedders(node_types, attr_embedding_dim, categorical_attributes,
                                             continuous_attributes)

        all_types = [l for el in list(attr_embedders.values()) for l in el]
        all_types.sort()

        expected_types = [0, 1, 2]
        print(attr_embedders)

        self.assertListEqual(expected_types, all_types)

        for types in attr_embedders.values():
            self.assertNotEqual(types, [])


if __name__ == "__main__":
    unittest.main()
