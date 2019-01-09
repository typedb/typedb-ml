import unittest

import numpy as np
import tensorflow as tf

import kgcn.src.models.learners as base
import kgcn.src.models.manager as manager
import tensorflow.contrib.layers as layers

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')
flags.DEFINE_integer('classes_length', 2, 'Number of classes')
flags.DEFINE_integer('features_length', 8, 'Number of features after encoding')
flags.DEFINE_integer('aggregated_length', 20, 'Length of aggregated representation of neighbours, a hidden dimension')
flags.DEFINE_integer('output_length', 32, 'Length of the output of "combine" operation, taking place at each depth, '
                                          'and the final length of the embeddings')

flags.DEFINE_integer('max_training_steps', 100, 'Max number of gradient steps to take during gradient descent')
flags.DEFINE_string('log_dir', './out', 'directory to use to store data from training')


def trial_data():
    num_samples = 30
    neighbourhood_sizes = (4, 3)
    feature_length = 8
    neighbourhood_shape = list(reversed(neighbourhood_sizes)) + [feature_length]
    shapes = [[num_samples] + neighbourhood_shape[i:] for i in range(len(neighbourhood_shape))]

    raw_neighbourhood_depths = [np.ones(shape, dtype=np.float32) for shape in shapes]

    label_value = [1, 0]
    raw_labels = [label_value for _ in range(num_samples)]
    labels = raw_labels
    return num_samples, neighbourhood_sizes, raw_neighbourhood_depths, labels


class TestLearningManager(unittest.TestCase):
    def test_train(self):
        num_samples, neighbourhood_sizes, neighbourhoods_depths, labels = trial_data()

        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        learner = base.SupervisedAccumulationLearner(FLAGS.classes_length, FLAGS.features_length,
                                                     FLAGS.aggregated_length,
                                                     FLAGS.output_length, neighbourhood_sizes, optimizer,
                                                     sigmoid_loss=True,
                                                     regularisation_weight=0.0, classification_dropout_keep_prob=1.0,
                                                     classification_activation=tf.nn.relu,
                                                     classification_regularizer=layers.l2_regularizer(scale=0.1),
                                                     classification_kernel_initializer=
                                                     tf.contrib.layers.xavier_initializer())
        sess = tf.Session()
        learning_manager = manager.LearningManager(learner, max_training_steps=FLAGS.max_training_steps,
                                                   log_dir=FLAGS.log_dir)

        # Build the placeholders for the neighbourhood_depths for each feature type
        raw_array_placeholders = manager.build_array_placeholders(num_samples, neighbourhood_sizes,
                                                                  FLAGS.features_length, tf.float32)
        # Build the placeholder for the labels
        labels_placeholder = manager.build_labels_placeholder(num_samples, FLAGS.classes_length)

        learning_manager(sess, raw_array_placeholders, labels_placeholder)

        feed_dict = {labels_placeholder: labels}
        for raw_array_placeholder, raw_array in zip(raw_array_placeholders, neighbourhoods_depths):
                feed_dict[raw_array_placeholder] = raw_array

        learning_manager.train(sess, feed_dict)


if __name__ == "__main__":
    unittest.main()
