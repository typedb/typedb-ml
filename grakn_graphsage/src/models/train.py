import time

import numpy as np
import tensorflow as tf

import grakn_graphsage.src.models.base as base
import tensorflow.contrib.layers as layers

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')
flags.DEFINE_integer('training_batch_size', 30, 'Training batch size')
flags.DEFINE_integer('neighbourhood_size_depth_1', 3, 'Neighbourhood size for depth 1')
flags.DEFINE_integer('neighbourhood_size_depth_2', 4, 'Neighbourhood size for depth 2')
flags.DEFINE_integer('neighbourhood_size_depth_3', 5, 'Neighbourhood size for depth 3')
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
    neighbourhood_shape = list(neighbourhood_sizes) + [feature_length]
    shapes = [[num_samples] + neighbourhood_shape[i:] for i in range(len(neighbourhood_shape))]

    raw_neighbourhood_depths = [np.ones(shape) for shape in shapes]

    label_value = [1, 0]
    raw_labels = [label_value for _ in range(num_samples)]
    labels = raw_labels
    return raw_neighbourhood_depths, labels


def main():
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

    neighbourhood_sizes = (FLAGS.neighbourhood_size_depth_2, FLAGS.neighbourhood_size_depth_1)
    model = base.SupervisedModel(FLAGS.classes_length, FLAGS.features_length, FLAGS.aggregated_length,
                                 FLAGS.output_length, neighbourhood_sizes, optimizer, sigmoid_loss=True,
                                 regularisation_weight=0.0, classification_dropout=0.3,
                                 classification_activation=tf.nn.relu,
                                 classification_regularizer=layers.l2_regularizer(scale=0.1),
                                 classification_kernel_initializer=tf.contrib.layers.xavier_initializer())

    neighbourhood_placeholders = [tf.placeholder(tf.float64, shape=(30, 4, 3, 8)),
                                  tf.placeholder(tf.float64, shape=(30, 3, 8)),
                                  tf.placeholder(tf.float64, shape=(30, 8))]
    labels_placeholder = tf.placeholder(tf.float64, shape=(30, 2))
    train_op, loss = model.train(neighbourhood_placeholders, labels_placeholder)

    # Build the summary Tensor based on the TF collection of Summaries.
    summary = tf.summary.merge_all()

    # Add the variable initializer Op.
    init = tf.global_variables_initializer()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

    # Run the Op to initialize the variables.
    sess.run(init)

    neighbourhoods_depths, labels = trial_data()

    feed_dict = {labels_placeholder: labels}
    for neighbourhood_placeholder, neighbourhood_depth in zip(neighbourhood_placeholders, neighbourhoods_depths):
        feed_dict[neighbourhood_placeholder] = neighbourhood_depth

    for step in range(FLAGS.max_training_steps):
        start_time = time.time()
        _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
        duration = time.time() - start_time

        if step % int(FLAGS.max_training_steps / 20) == 0:
            # Print status to stdout.
            print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
            # Update the events file.
            summary_str = sess.run(summary, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()


if __name__ == "__main__":
    main()
