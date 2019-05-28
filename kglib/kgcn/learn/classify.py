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
import abc

import tensorflow as tf
import tensorflow.contrib.layers as layers

import kglib.kgcn.learn.metrics.report as metrics
import kglib.kgcn.core.model as model


class SupervisedKGCNMultiClassClassifier(abc.ABC):
    def __init__(
            self,
            kgcn: model.KGCN,
            optimizer,
            num_classes,
            log_dir,
            max_training_steps=10000,
            regularisation_weight=0.0,
            classification_dropout_keep_prob=0.7,
            use_bias=True,
            classification_activation=lambda x: x,
            classification_regularizer=layers.l2_regularizer(scale=0.1),
            classification_kernel_initializer=tf.contrib.layers.xavier_initializer()):

        self._log_dir = log_dir
        self._write_summary = self._log_dir is not None
        self._kgcn = kgcn
        self._optimizer = optimizer
        self._num_classes = num_classes
        self._max_training_steps = max_training_steps
        self._regularisation_weight = regularisation_weight
        self._classification_dropout_keep_prob = classification_dropout_keep_prob
        self._use_bias = use_bias
        self._classification_activation = classification_activation
        self._classification_regularizer = classification_regularizer
        self._classification_kernel_initializer = classification_kernel_initializer

        ################################################################################################################
        # KGCN Embeddings
        ################################################################################################################
        self.labels_placeholder = tf.placeholder(tf.float32, shape=(None, num_classes), name='labels_input')
        labels_dataset = tf.data.Dataset.from_tensor_slices(self.labels_placeholder)

        self.embeddings, next_batch, self.dataset_initializer, self.neighbourhood_placeholders = self._kgcn.embed(
            labels_dataset)
        self.labels = next_batch[0]

        ################################################################################################################
        # Downstream of embeddings - classification
        ################################################################################################################
        classification_layer = tf.layers.Dense(self._num_classes,
                                               activation=self._classification_activation,
                                               use_bias=self._use_bias,
                                               kernel_regularizer=self._classification_regularizer,
                                               kernel_initializer=self._classification_kernel_initializer,
                                               name='classification_dense_layer')

        # tf.summary.histogram('classification/dense/kernel', classification_layer.kernel)  # TODO figure out why
        #  this is throwing an error
        # tf.summary.histogram('classification/dense/bias', classification_layer.bias)

        class_scores = classification_layer(self.embeddings)
        tf.summary.histogram('classification/dense/class_scores', class_scores)

        regularised_class_scores = tf.nn.dropout(class_scores,
                                                 self._classification_dropout_keep_prob,
                                                 name='classification_dropout')

        tf.summary.histogram('evaluate/regularised_class_scores', regularised_class_scores)

        self._class_scores = regularised_class_scores

        self._loss_op = self.loss(class_scores, self.labels)
        self._train_op = self.optimise(self._loss_op)

        self.tf_session = None
        self.summary_writer = None
        self.summary = None

        # Subclasses need to initialise various TensorFlow components with:
        # self._initialise_computation_graph_components()

    def _initialise_computation_graph_components(self):
        ################################################################################################################
        # Graph initialisation tasks - run after the whole graph has been built
        ################################################################################################################
        self.tf_session = tf.Session()
        # Add the variable initializer Op.
        init_global = tf.global_variables_initializer()
        init_local = tf.local_variables_initializer()  # Added to initialise tf.metrics.recall
        init_tables = tf.tables_initializer()

        # Instantiate a SummaryWriter to output summaries and the Graph.
        if self._write_summary:
            self.summary_writer = tf.summary.FileWriter(self._log_dir, self.tf_session.graph)

        # Run the Op to initialize the variables.
        self.tf_session.run(init_global)
        self.tf_session.run(init_local)
        self.tf_session.run(init_tables)
        self.summary = tf.summary.merge_all()

    @abc.abstractmethod
    def _per_class_loss(self, logits, labels):
        return

    @abc.abstractmethod
    def _objective_loss(self, per_class_loss):
        return

    def loss(self, logits, labels=None):

        with tf.name_scope('loss') as scope:
            # Get the losses from the various layers
            regularisation_loss = tf.cast(self._regularisation_weight * tf.losses.get_regularization_loss(), tf.float32)
            tf.summary.scalar('regularisation_loss', regularisation_loss)

            per_class_loss = self._per_class_loss(logits, labels)
            objective_loss = self._objective_loss(per_class_loss)

            loss = objective_loss + regularisation_loss

            tf.summary.histogram('loss/per_class_loss', per_class_loss)
            tf.summary.scalar('loss/final_loss', loss)

        return loss

    def optimise(self, loss):
        grads_and_vars = self._optimizer.compute_gradients(loss)

        for grad, var in grads_and_vars:
            tf.summary.histogram('gradients/' + var.name, grad)

        opt_op = self._optimizer.apply_gradients(grads_and_vars), loss

        return opt_op



    def predict(self, feed_dict):
        print("========= Evaluation =========")
        _ = self.tf_session.run(self.dataset_initializer, feed_dict=feed_dict)

        loss_value, class_scores_values, predictions_class_winners_values = self.tf_session.run(
            [self._loss_op, self._class_scores, self._predictions])

        print(class_scores_values)
        print(f'Loss: {loss_value:.2f}')
        print("========= Evaluation Complete =========\n\n")

    def get_feed_dict(self, session, concepts, labels=None):

        # Possibly save/load context arrays here instead
        context_arrays = self._kgcn.input_fn(session, concepts)

        feed_dict = build_feed_dict(self.neighbourhood_placeholders,
                                    context_arrays,
                                    labels_placeholder=self.labels_placeholder,
                                    labels=labels)
        return feed_dict


class SupervisedKGCNMultiClassSingleLabelClassifier(SupervisedKGCNMultiClassClassifier):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._labels_winners = tf.argmax(self.labels, -1)
        self._predictions = tf.argmax(self._class_scores, -1)
        self._confusion_matrix = tf.confusion_matrix(self._labels_winners,
                                                     self._predictions,
                                                     num_classes=self._num_classes)

        self._initialise_computation_graph_components()

    def _per_class_loss(self, logits, labels):
        return tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)

    def _objective_loss(self, per_class_loss):
        return tf.reduce_mean(per_class_loss)

    def train(self, feed_dict):
        print("========= Training =========")
        _ = self.tf_session.run(self.dataset_initializer, feed_dict=feed_dict)
        for step in range(self._max_training_steps):
            _, loss_value, confusion_matrix, class_scores_values, predictions_class_winners_values, \
                labels_winners_values = self.tf_session.run(
                    [self._train_op, self._loss_op, self._confusion_matrix, self._class_scores,
                     self._predictions, self._labels_winners])

            summary_str = self.tf_session.run(self.summary, feed_dict=feed_dict)
            if self._write_summary:
                self.summary_writer.add_summary(summary_str, step)
                self.summary_writer.flush()
            if step % int(self._max_training_steps / 20) == 0:
                print(f'\n-----')
                print(f'Step {step}')
                print(f'Loss: {loss_value:.2f}')
                metrics.report_multiclass_metrics(labels_winners_values, predictions_class_winners_values)
        print("========= Training Complete =========\n\n")

    def eval(self, feed_dict):
        print("========= Evaluation =========")
        _ = self.tf_session.run(self.dataset_initializer, feed_dict=feed_dict)

        loss_value, confusion_matrix, class_scores_values, predictions_class_winners_values, labels_winners_values = \
            self.tf_session.run(
                [self._loss_op, self._confusion_matrix, self._class_scores, self._predictions,
                 self._labels_winners])

        print(f'Loss: {loss_value:.2f}')
        metrics.report_multiclass_metrics(labels_winners_values, predictions_class_winners_values)
        print("========= Evaluation Complete =========\n\n")


class SupervisedKGCNMultiClassMultiLabelClassifier(SupervisedKGCNMultiClassClassifier):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        predictions = tf.cast(self._class_scores, tf.float32)
        threshold = 0.5
        self._predictions = tf.cast(tf.greater(predictions, threshold), tf.int64)

        self._initialise_computation_graph_components()

    def _per_class_loss(self, logits, labels):
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)

    def _objective_loss(self, per_class_loss):
        # the loss is summed across classes before it is averaged over the batch
        # https://github.com/tensorflow/skflow/issues/113#issuecomment-397631386
        return tf.reduce_mean(tf.reduce_sum(per_class_loss, axis=1))

    def train(self, feed_dict):
        print("========= Training =========")
        _ = self.tf_session.run(self.dataset_initializer, feed_dict=feed_dict)
        for step in range(self._max_training_steps):
            _, loss_value, class_scores_values, predictions_class_winners_values, labels_values = self.tf_session.run(
                [self._train_op, self._loss_op, self._class_scores, self._predictions, self.labels])

            summary_str = self.tf_session.run(self.summary, feed_dict=feed_dict)

            if self._write_summary:
                self.summary_writer.add_summary(summary_str, step)
                self.summary_writer.flush()

            if step % int(self._max_training_steps / 20) == 0:

                print(f'\n-----')
                print(f'Step {step}')
                print(f'Loss: {loss_value:.2f}')

                metrics.report_multilabel_metrics(labels_values, predictions_class_winners_values)
        print("========= Training Complete =========\n\n")

    def eval(self, feed_dict):
        print("========= Evaluation =========")
        _ = self.tf_session.run(self.dataset_initializer, feed_dict=feed_dict)

        loss_value, class_scores_values, predictions_class_winners_values, labels_values = self.tf_session.run(
            [self._loss_op, self._class_scores, self._predictions, self.labels])

        print(f'Loss: {loss_value:.2f}')
        metrics.report_multilabel_metrics(labels_values, predictions_class_winners_values)
        print("========= Evaluation Complete =========\n\n")


def build_feed_dict(context_array_placeholders, context_array_depths, labels_placeholder=None, labels=None):
    feed_dict = {}
    if labels is not None:
        feed_dict[labels_placeholder] = labels

    for context_array_placeholder, context_arrays_dict in zip(context_array_placeholders, context_array_depths):
        for feature_type_name in list(context_arrays_dict.keys()):
            feed_dict[context_array_placeholder[feature_type_name]] = context_arrays_dict[feature_type_name]

    return feed_dict
