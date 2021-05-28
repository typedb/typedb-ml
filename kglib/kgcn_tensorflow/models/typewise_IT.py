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
from tensorflow.python.framework.ops import EagerTensor

from kglib.kgcn_tensorflow.models.typewise import TypewiseEncoder


class ITTypewiseEncoder(unittest.TestCase):

    def setUp(self):
        tf.enable_eager_execution()

    def test_with_tensors(self):
        tf.reset_default_graph()
        tf.set_random_seed(1)

        things = tf.convert_to_tensor(np.array([[0, 0], [1, 0], [2, 0.5673]], dtype=np.float32))

        entity_relation = lambda x: x
        continuous_attribute = lambda x: x

        encoders_for_types = {lambda: entity_relation: [0, 1], lambda: continuous_attribute: [2]}

        tm = TypewiseEncoder(encoders_for_types, 1)
        encoded_things = tm(things)  # The function under test

        # Check that tensorflow was actually used
        self.assertEqual(EagerTensor, type(encoded_things))


if __name__ == '__main__':
    unittest.main()
