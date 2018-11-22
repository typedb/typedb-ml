import collections
import time
import typing as typ

import tensorflow as tf


def build_array_placeholders(batch_size, neighbourhood_sizes, features_length,
                             feature_types: typ.Union[typ.List[typ.MutableMapping[str, tf.DType]], tf.DType],
                             name=None):
    array_neighbourhood_sizes = list(reversed(neighbourhood_sizes))
    neighbourhood_placeholders = []
    for i in range(len(array_neighbourhood_sizes) + 1):
        shape = [None] + list(array_neighbourhood_sizes[i:]) + [features_length]

        try:
            phs = tf.placeholder(feature_types, shape=shape, name=name)
        except:
            phs = {name: tf.placeholder(type, shape=shape, name=name) for name, type in feature_types[i].items()}

        neighbourhood_placeholders.append(phs)
    return neighbourhood_placeholders


# TODO Update and move now this isn't used here


def build_labels_placeholder(batch_size, classes_length, name=None):
    return tf.placeholder(tf.float32, shape=(batch_size, classes_length), name=name)


class LearningManager:

    def __init__(self, learner, max_training_steps, log_dir):
        self._learner = learner
        self._max_training_steps = max_training_steps
        self._log_dir = log_dir

    def __call__(self, sess, neighbourhoods_input, labels_input):
        self.train_op, self.loss, self.class_predictions, self.micro_precisions, self.micro_precisions_update, \
        self.micro_recalls, self.micro_recalls_update, self.f1_score, self.update_f1_score, \
        self.confusion_matrix = self._learner.train_and_evaluate(
            neighbourhoods_input, labels_input)

        # Build the summary Tensor based on the TF collection of Summaries.
        self.summary = tf.summary.merge_all()

        # Add the variable initializer Op.
        init_global = tf.global_variables_initializer()
        init_local = tf.local_variables_initializer()  # Added to initialise tf.metrics.recall
        init_tables = tf.tables_initializer()

        # Instantiate a SummaryWriter to output summaries and the Graph.
        self.summary_writer = tf.summary.FileWriter(self._log_dir, sess.graph)

        # Run the Op to initialize the variables.
        sess.run(init_global)
        sess.run(init_local)
        sess.run(init_tables)

    def train(self, sess, feed_dict):
        print("\n\n========= Training and Evaluation =========")
        for step in range(self._max_training_steps):
            start_time = time.time()

            if step % int(self._max_training_steps / 20) == 0:
                _, loss_value, micro_precision_values, _, micro_recall_values, _, f1_score_value, _, confusion_matrix_value = \
                    sess.run([self.train_op, self.loss, self.micro_precisions, self.micro_precisions_update,
                              self.micro_recalls, self.micro_recalls_update, self.f1_score, self.update_f1_score,
                              self.confusion_matrix], feed_dict=feed_dict)

                duration = time.time() - start_time
                print(f'Step {step}: loss {loss_value:.2f}, micro precision {micro_precision_values:.2f}, '
                      f'micro recall {micro_recall_values:.2f}, micro f1-score {f1_score_value:.2f}'
                      f'     ({duration:.3f} sec)\n'
                      f'Confusion matrix:\n'
                      f'{confusion_matrix_value}')

                summary_str = sess.run(self.summary, feed_dict=feed_dict)
                self.summary_writer.add_summary(summary_str, step)
                self.summary_writer.flush()
            else:
                _, loss_value = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        print("\n\n========= Training and Evaluation Complete =========\n\n")

    def evaluate(self, sess, feed_dict):
        micro_precision_values, _, micro_recall_values, _, f1_score_value, _, confusion_matrix_value, \
        class_prediction_values = sess.run(
            [self.micro_precisions, self.micro_precisions_update,
             self.micro_recalls, self.micro_recalls_update, self.f1_score,
             self.update_f1_score, self.confusion_matrix, self.class_predictions],
            feed_dict=feed_dict)

        print("\n\n========= Evaluation =========")

        print(f'Micro precision {micro_precision_values:.2f},\n'
              f'Micro recall {micro_recall_values:.2f},\n'
              f'Micro f1-score {f1_score_value:.2f}\n'
              f'Confusion matrix:\n'
              f'{confusion_matrix_value}')
        print(f'Class assignments: \n{class_prediction_values}')

        print("\n\n========= Evaluation Complete =========")

    def predict(self, sess, feed_dict):
        print("\n\n========= Prediction =========")
        class_prediction_values = sess.run([self.class_predictions], feed_dict=feed_dict)
        print(f'predictions: \n{class_prediction_values}')
        print("\n\n========= Prediction Complete =========\n\n")
