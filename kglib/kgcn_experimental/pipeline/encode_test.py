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

from kglib.kgcn_experimental.pipeline.encode import augment_data_fields


class TestAugmentDataFields(unittest.TestCase):

    def test_numpy_fields_augmented_as_expected(self):
        data = [dict(attr1=np.array([0, 1, 0]), attr2=np.array([5]))]

        augment_data_fields(data, ('attr1', 'attr2'), 'features')

        expected_data = [dict(attr1=np.array([0, 1, 0]), attr2=np.array([5]), features=np.array([0, 1, 0, 5]))]

        np.testing.assert_equal(expected_data, data)

    def test_augmenting_non_numpy_numeric(self):
        data = [dict(attr1=np.array([0, 1, 0]), attr2=5)]

        augment_data_fields(data, ('attr1', 'attr2'), 'features')

        expected_data = [dict(attr1=np.array([0, 1, 0]), attr2=5, features=np.array([0, 1, 0, 5]))]

        np.testing.assert_equal(expected_data, data)


if __name__ == "__main__":
    unittest.main()
