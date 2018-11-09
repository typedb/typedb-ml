import time
import typing as typ

import tensorflow as tf
import tensorflow.contrib.layers as layers

import kgcn.src.models.learners as base

flags = tf.app.flags
FLAGS = flags.FLAGS


def build_array_placeholders(batch_size, neighbourhood_sizes, features_length,
                             feature_types: typ.Union[typ.List[typ.MutableMapping[str, tf.DType]], tf.DType]):
    array_neighbourhood_sizes = list(reversed(neighbourhood_sizes))
    neighbourhood_placeholders = []
    for i in range(len(array_neighbourhood_sizes) + 1):
        shape = [batch_size] + list(array_neighbourhood_sizes[i:]) + [features_length]

        try:
            phs = tf.placeholder(feature_types, shape=shape)
        except:
            phs = {name: tf.placeholder(type, shape=shape) for name, type in feature_types[i].items()}

        neighbourhood_placeholders.append(phs)
    return neighbourhood_placeholders


def build_labels_placeholder(batch_size, classes_length):
    return tf.placeholder(tf.float32, shape=(batch_size, classes_length))


def supervised_train(neighbourhood_sizes, neighbourhoods_depths, labels):
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

    learner = base.SupervisedAccumulationLearner(FLAGS.classes_length, FLAGS.features_length, FLAGS.aggregated_length,
                                                 FLAGS.output_length, neighbourhood_sizes, optimizer, sigmoid_loss=True,
                                                 regularisation_weight=0.0, classification_dropout=0.3,
                                                 classification_activation=tf.nn.relu,
                                                 classification_regularizer=layers.l2_regularizer(scale=0.1),
                                                 classification_kernel_initializer=tf.contrib.layers.xavier_initializer())

    # Build the placeholders for the neighbourhood_depths
    neighbourhood_placeholders = build_array_placeholders(FLAGS.training_batch_size, neighbourhood_sizes,
                                                          FLAGS.features_length, tf.float32)

    # Build the placeholder for the labels
    labels_placeholder = build_labels_placeholder(FLAGS.training_batch_size, FLAGS.classes_length)

    # train_op, loss = learner.train(neighbourhood_placeholders, labels_placeholder)
    train_op, loss, class_predictions, precision, recall, f1_score = learner.train_and_evaluate(
        neighbourhood_placeholders, labels_placeholder)

    # Build the summary Tensor based on the TF collection of Summaries.
    summary = tf.summary.merge_all()

    # Add the variable initializer Op.
    init_global = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()  # Added to initialise tf.metrics.recall

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

    # Run the Op to initialize the variables.
    sess.run(init_global)
    sess.run(init_local)

    feed_dict = {labels_placeholder: labels}
    for neighbourhood_placeholder, neighbourhood_depth in zip(neighbourhood_placeholders, neighbourhoods_depths):
        feed_dict[neighbourhood_placeholder] = neighbourhood_depth

    print("\n\n========= Training and Evaluation =========")
    for step in range(FLAGS.max_training_steps):
        start_time = time.time()

        if step % int(FLAGS.max_training_steps / 20) == 0:
            _, loss_value, precision_value, recall_value, f1_score_value = sess.run(
                [train_op, loss, precision, recall, f1_score], feed_dict=feed_dict)

            duration = time.time() - start_time
            print(f'Step {step}: loss {loss_value:.2f}, precision {precision_value}, '
                  f'recall {recall_value}, f1-score {f1_score_value}     ({duration:.3f} sec)')

            summary_str = sess.run(summary, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()
        else:
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

    print("\n\n========= Prediction =========")
    class_prediction_values = sess.run([class_predictions], feed_dict=feed_dict)
    print(f'predictions: \n{class_prediction_values}')
