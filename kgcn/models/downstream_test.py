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
