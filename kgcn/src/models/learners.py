import abc

import tensorflow as tf
import tensorflow.contrib.layers as layers

import kgcn.src.models.aggregation as agg
import kgcn.src.models.initialise as init


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
                 sigmoid_loss=True, regularisation_weight=0.0, classification_dropout_keep_prob=1.0,
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
        tf.summary.histogram('classification/dense/bias', classification_layer.bias)
        tf.summary.histogram('classification/dense/output', class_predictions)

        regularised_class_predictions = tf.nn.dropout(class_predictions, self._classification_dropout_keep_prob,
                                                      name='classification_dropout')

        self._class_predictions = regularised_class_predictions

        return regularised_class_predictions

    def loss(self, predictions, labels=None):

        loss = supervised_loss(predictions, labels, regularisation_weight=self._regularisation_weight,
                               sigmoid_loss=self._sigmoid_loss)
        tf.summary.scalar('loss', loss)
        return loss

    def predict(self, prediction):
        if self._sigmoid_loss:
            sigmoid_class_prediction = tf.nn.sigmoid(prediction)
            tf.summary.histogram('sigmoid_class_prediction', sigmoid_class_prediction)
            return sigmoid_class_prediction
        else:
            softmax_class_prediction = tf.nn.softmax(prediction)
            tf.summary.histogram('softmax_class_prediction', softmax_class_prediction)
            return softmax_class_prediction

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

    def train_and_evaluate(self, neighbourhoods, labels):
        embeddings = self.embedding(neighbourhoods)
        tf.summary.histogram('evaluate/embeddings', embeddings)
        class_predictions = self.inference(embeddings)
        tf.summary.histogram('evaluate/class_predictions', class_predictions)
        loss = self.loss(class_predictions, labels)
        precision, _ = tf.metrics.precision(labels, class_predictions)
        recall, _ = tf.metrics.recall(labels, class_predictions)
        with tf.name_scope('f1_score') as scope:
            f1_score = (2 * precision * recall) / (precision + recall)

        tf.summary.scalar('evaluate/precision', precision)
        tf.summary.scalar('evaluate/recall', recall)
        tf.summary.scalar('evaluate/f1_score', f1_score)

        return self.optimise(loss), loss, self.predict(class_predictions), precision, recall, f1_score


def supervised_loss(predictions, labels, regularisation_weight=0.5, sigmoid_loss=True):
    with tf.name_scope('loss') as scope:
        # Get the losses from the various layers
        loss = tf.cast(regularisation_weight * tf.losses.get_regularization_loss(), tf.float32)
        tf.summary.scalar('regularisation_loss', loss)
        # classification loss
        if sigmoid_loss:
            loss_fn = tf.nn.sigmoid_cross_entropy_with_logits
        else:
            loss_fn = tf.nn.softmax_cross_entropy_with_logits

        raw_loss = loss_fn(logits=predictions, labels=labels)
        tf.summary.histogram('raw_loss', raw_loss)
        loss += tf.reduce_mean(raw_loss)
        return loss
