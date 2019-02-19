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

import kglib.kgcn.core.ingest.encode.boolean as bool_enc

tf.enable_eager_execution()


class TestOneHotBooleanEncoding(unittest.TestCase):
    def test_input_output(self):
        input_data = np.full((2, 3, 1), -1, dtype=np.int64)

        input_data[1, 1] = 0
        input_data[1, 0] = 1
        input_data[0, 2] = 1

        print(f'Input:\n{input_data}')
        expected_output = np.full((2, 3, 2), 0, dtype=np.int64)
        expected_output[1, 1, 0] = 1
        expected_output[1, 0, 1] = 1
        expected_output[0, 2, 1] = 1

        print(f'\nExpected output\n{expected_output}')

        output = bool_enc.one_hot_boolean_encode(tf.convert_to_tensor(input_data, dtype=tf.int64))

        print(f'output:\n{output.numpy()}')
        with self.subTest('Correctness'):
            np.testing.assert_array_equal(output.numpy(), expected_output)


if __name__ == "__main__":
    unittest.main()
