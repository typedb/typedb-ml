import abc

import tensorflow as tf
import tensorflow.contrib.layers as layers

import kgcn.src.models.aggregation as agg
import kgcn.src.models.initialise as init
import tf_metrics


class AccumulationLearner:
    def __init__(self, feature_lengths, aggregated_length, output_length, neighbourhood_sizes,
                 normalisation=tf.nn.l2_normalize):

        self._neighbourhood_sizes = neighbourhood_sizes
        self._aggregated_length = aggregated_length
        self._feature_lengths = feature_lengths
        self._output_length = output_length
        self._combined_lengths = [feature_length + self._aggregated_length for feature_length in self._feature_lengths]
        self._normalisation = normalisation

    def embedding(self, neighbourhoods):

        # TODO pass through params for aggregators, combiners and normalisers
        aggregators = [agg.Aggregate(self._aggregated_length, name=f'aggregate_{i}') for i in range(len(self._neighbourhood_sizes))]

        combiners = []
        for i in range(len(self._neighbourhood_sizes)):

            weights = init.initialise_glorot_weights((self._combined_lengths[i], self._output_length),
                                                     name=f'weights_{i}')

            if i + 1 == len(self._neighbourhood_sizes):
                combiner = agg.Combine(weights, activation=lambda x: x, name=f'combine_{i}_linear')
            else:
                combiner = agg.Combine(weights, activation=tf.nn.relu, name=f'combine_{i}_relu')
            combiners.append(combiner)

        normalisers = [agg.normalise for _ in range(len(self._neighbourhood_sizes))]

        full_representation = agg.chain_aggregate_combine(neighbourhoods, aggregators, combiners, normalisers)
        # full_representation = tf.nn.l2_normalize(full_representation, -1)
        return full_representation

    @abc.abstractmethod
    def loss(self, predictions, labels=None):
        pass


class SupervisedAccumulationLearner(AccumulationLearner):

    def __init__(self, labels_length, feature_lengths, aggregated_length, output_length, neighbourhood_sizes, optimizer,
                 sigmoid_loss=True, regularisation_weight=0.0, classification_dropout_keep_prob=0.9,
                 classification_activation=tf.nn.tanh, classification_regularizer=layers.l2_regularizer(scale=0.1),
                 classification_kernel_initializer=tf.contrib.layers.xavier_initializer(), **kwargs):
        super().__init__(feature_lengths, aggregated_length, output_length, neighbourhood_sizes, **kwargs)
        self._optimizer = optimizer
        self._regularisation_weight = regularisation_weight
        self._sigmoid_loss = sigmoid_loss
        self._classification_dropout_keep_prob = classification_dropout_keep_prob
        self._labels_length = labels_length
        self._classification_activation = classification_activation
        self._classification_regularizer = classification_regularizer
        self._classification_kernel_initializer = classification_kernel_initializer
        self._class_predictions = None

    def inference(self, embeddings):
        classification_layer = tf.layers.Dense(self._labels_length, activation=self._classification_activation,
                                               use_bias=True, kernel_regularizer=self._classification_regularizer,
                                               kernel_initializer=self._classification_kernel_initializer,
                                               name='classification_dense_layer')

        class_predictions = classification_layer(embeddings)
        tf.summary.histogram('classification/dense/kernel', classification_layer.kernel)
        # tf.summary.histogram('classification/dense/bias', classification_layer.bias)
        tf.summary.histogram('classification/dense/output', class_predictions)

        regularised_class_predictions = tf.nn.dropout(class_predictions, self._classification_dropout_keep_prob,
                                                      name='classification_dropout')

        self._class_predictions = regularised_class_predictions

        return regularised_class_predictions

    def loss(self, logits, labels=None):

        loss = supervised_loss(logits, labels, regularisation_weight=self._regularisation_weight,
                               sigmoid_loss=self._sigmoid_loss)
        tf.summary.scalar('loss/final_loss', loss)
        return loss

    def predict(self, prediction):
        # prediction = tf.Print(prediction, [prediction], name='print_class_predictions',
        #                       message='Class predictions: ', summarize=15 * 3)
        if self._sigmoid_loss:
            sigmoid_class_prediction = tf.nn.sigmoid(prediction)
            tf.summary.histogram('sigmoid_class_prediction', sigmoid_class_prediction)
            return sigmoid_class_prediction
        else:
            softmax_class_prediction = tf.nn.softmax(prediction)
            tf.summary.histogram('softmax_class_prediction', softmax_class_prediction)
            return tf.cast(tf.round(softmax_class_prediction), dtype=tf.int32)

    def optimise(self, loss):
        grads_and_vars = self._optimizer.compute_gradients(loss)
        # clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
        #         for grad, var in grads_and_vars]
        # tf.summary.histogram('clipped_grads_and_vars', clipped_grads_and_vars[0])
        # # grad, _ = clipped_grads_and_vars[0]
        # opt_op = self._optimizer.apply_gradients(clipped_grads_and_vars), loss

        for grad, var in grads_and_vars:
            tf.summary.histogram('gradients/' + var.name, grad)

        opt_op = self._optimizer.apply_gradients(grads_and_vars), loss

        # opt_op = self._optimizer.minimize(loss)

        return opt_op

    def train(self, neighbourhoods, labels):
        embeddings = self.embedding(neighbourhoods)
        class_predictions = self.inference(embeddings)

        loss = self.loss(class_predictions, labels)
        return self.optimise(loss)

    def discretise_class_predictions(self, class_predictions):
        if self._sigmoid_loss:
            return class_predictions
        else:
            return tf.cast(tf.round(class_predictions), dtype=tf.int32)

    def train_and_evaluate(self, neighbourhoods, labels):
        embeddings = self.embedding(neighbourhoods)
        tf.summary.histogram('evaluate/embeddings', embeddings)
        class_predictions = self.inference(embeddings)

        tf.summary.histogram('evaluate/class_predictions', class_predictions)
        loss = self.loss(class_predictions, labels)

        with tf.name_scope('metrics'):
            discrete_class_predictions = self.discretise_class_predictions(class_predictions)

            # class_precisions = []
            # class_recalls = []
            # class_f1_scores = []
            # print(f'predictions shape: {discrete_class_predictions.shape}')
            # print(f'labels shape: {discrete_class_predictions.shape}')
            # for i in range(self._labels_length):
            #     class_precision, update_class_precision = tf.metrics.precision(labels[..., i], discrete_class_predictions[..., i])
            #     class_precisions.append((class_precision, update_class_precision))
            #
            #     class_recall, update_class_recall = tf.metrics.recall(labels[..., i], discrete_class_predictions[..., i])
            #     class_recalls.append((class_recall, update_class_recall))
            #
            #     with tf.name_scope('f1_score') as scope:
            #         class_f1_score = (2 * class_precision * class_recall) / (class_precision + class_recall)
            #         class_f1_scores.append(class_f1_score)
            #
            #     tf.summary.scalar(f'evaluate/class_{i}_precision', class_precision)
            #     tf.summary.scalar(f'evaluate/class_{i}_recall', class_recall)
            #     tf.summary.scalar(f'evaluate/class_{i}_f1_score', class_f1_score)

            average = 'micro'

            winning_labels = tf.argmax(labels, -1)
            winning_discrete_class_predictions = tf.argmax(discrete_class_predictions, -1)

            micro_precision, update_precision = tf_metrics.precision(winning_labels, winning_discrete_class_predictions,
                                                                           self._labels_length, average=average)
            micro_recall, update_recall = tf_metrics.recall(winning_labels, winning_discrete_class_predictions, self._labels_length,
                                                                  average=average)
            micro_f1_score, update_f1_score = tf_metrics.recall(winning_labels, winning_discrete_class_predictions, self._labels_length,
                                                                  average=average)

            tf.summary.scalar('evaluate/micro_precision', micro_precision)
            tf.summary.scalar('evaluate/micro_recall', micro_recall)
            tf.summary.scalar('evaluate/micro_f1_score', micro_f1_score)

        return self.optimise(loss), loss, self.predict(class_predictions), update_precision, \
               micro_precision, micro_recall, update_recall, micro_f1_score, update_f1_score, \
               tf.confusion_matrix(cm_prep(labels), cm_prep(discrete_class_predictions))


def cm_prep(labels):
    return tf.argmax(labels, -1)


def supervised_loss(logits, labels, regularisation_weight=0.5, sigmoid_loss=True):
    with tf.name_scope('loss') as scope:
        # Get the losses from the various layers
        loss = tf.cast(regularisation_weight * tf.losses.get_regularization_loss(), tf.float32)
        tf.summary.scalar('regularisation_loss', loss)
        # classification loss
        if sigmoid_loss:
            loss_fn = tf.nn.sigmoid_cross_entropy_with_logits
        else:
            loss_fn = tf.nn.softmax_cross_entropy_with_logits

        raw_loss = loss_fn(logits=logits, labels=labels)
        # raw_loss = tf.Print(raw_loss, [raw_loss], name='raw_loss',
        #                       message='Raw loss: ', summarize=15 * 3)  # Print deprecated in r1.12
        tf.summary.histogram('loss/raw_loss', raw_loss)
        if sigmoid_loss:
            class_summed_loss = tf.reduce_sum(raw_loss, axis=1)
            tf.summary.histogram('loss/class_summed_loss', class_summed_loss)
            raw_loss = class_summed_loss
        loss += tf.reduce_mean(raw_loss)
        return loss
