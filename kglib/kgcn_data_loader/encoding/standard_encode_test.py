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

from kglib.kgcn_data_loader.encoding.standard_encode import stack_features


class TestAugmentDataFields(unittest.TestCase):

    def test_numpy_fields_augmented_as_expected(self):
        features = [np.array([0, 1, 0]), np.array([5])]

        stacked = stack_features(features)

        expected = np.array([0, 1, 0, 5])

        np.testing.assert_equal(expected, stacked)

    def test_augmenting_non_numpy_numeric(self):
        data = [np.array([0, 1, 0]), 5]

        stacked = stack_features(data)

        expected = np.array([0, 1, 0, 5])

        np.testing.assert_equal(stacked, expected)


if __name__ == "__main__":
    unittest.main()
