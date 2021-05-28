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

from kglib.kgcn_tensorflow.models.attribute import CategoricalAttribute
import tensorflow as tf
import numpy as np


class ITCategoricalAttribute(unittest.TestCase):
    def test_output_tensorspec(self):
        cat = CategoricalAttribute(2, 5)
        inp = tf.zeros((3, 1), dtype=tf.float32)
        output = cat(inp)
        np.testing.assert_array_equal(tf.TensorShape([3, 5]), output.shape)
        np.testing.assert_equal(output.dtype, tf.float32)


if __name__ == "__main__":
    unittest.main()
