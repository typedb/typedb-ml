import time
import typing as typ

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


# TODO Update and move now this isn't used here
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


class LearningManager:

    def __init__(self, learner):
        self._learner = learner

    def __call__(self, sess, neighbourhoods_input, labels_input):

        # Build the placeholders for the neighbourhood_depths
        # neighbourhood_placeholders = build_array_placeholders(FLAGS.training_batch_size, neighbourhood_sizes,
        #                                                       FLAGS.features_length, tf.float32)

        # Build the placeholder for the labels
        # labels_placeholder = build_labels_placeholder(FLAGS.training_batch_size, FLAGS.classes_length)

        # train_op, loss = learner.train(neighbourhood_placeholders, labels_placeholder)
        self.train_op, self.loss, self.class_predictions, self.precision, self.recall, self.f1_score = \
            self._learner.train_and_evaluate(neighbourhoods_input, labels_input)

        # Build the summary Tensor based on the TF collection of Summaries.
        self.summary = tf.summary.merge_all()

        # Add the variable initializer Op.
        init_global = tf.global_variables_initializer()
        init_local = tf.local_variables_initializer()  # Added to initialise tf.metrics.recall

        # Instantiate a SummaryWriter to output summaries and the Graph.
        self.summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        # Run the Op to initialize the variables.
        sess.run(init_global)
        sess.run(init_local)

    def train(self, sess, feed_dict):
        print("\n\n========= Training and Evaluation =========")
        for step in range(FLAGS.max_training_steps):
            start_time = time.time()

            if step % int(FLAGS.max_training_steps / 20) == 0:
                _, loss_value, precision_value, recall_value, f1_score_value = sess.run(
                    [self.train_op, self.loss, self.precision, self.recall, self.f1_score], feed_dict=feed_dict)

                duration = time.time() - start_time
                print(f'Step {step}: loss {loss_value:.2f}, precision {precision_value}, '
                      f'recall {recall_value}, f1-score {f1_score_value}     ({duration:.3f} sec)')

                summary_str = sess.run(self.summary, feed_dict=feed_dict)
                self.summary_writer.add_summary(summary_str, step)
                self.summary_writer.flush()
            else:
                _, loss_value = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                return _, loss_value

    def predict(self, sess, feed_dict):
        print("\n\n========= Prediction =========")
        class_prediction_values = sess.run([self.class_predictions], feed_dict=feed_dict)
        print(f'predictions: \n{class_prediction_values}')

