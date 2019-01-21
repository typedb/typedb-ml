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

import tensorflow as tf

import kgcn.models.downstream as downstream

tf.enable_eager_execution()


class TestSupervisedLoss(unittest.TestCase):

    def test_loss_rank_1(self):

        label_values = [1, [1], [1, 0, 1]]

        for label_value in label_values:
            self.subTest()
            with self.subTest(labels=str(label_value)):
                num_samples = 30
                raw_labels = [label_value for _ in range(num_samples)]
                labels = tf.convert_to_tensor(raw_labels, dtype=tf.float32)
                predictions = tf.convert_to_tensor(raw_labels, dtype=tf.float32)

                loss = downstream.supervised_loss(predictions, labels, 0.0, True)
                print(loss)

                self.assertEqual(loss.shape, ())


if __name__ == "__main__":
    unittest.main()
