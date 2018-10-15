import tensorflow as tf
import grakn_graphsage.src.models.aggregation as agg
import grakn_graphsage.src.models.initialise as init
import tensorflow.contrib.layers as layers
import abc


class Model:
    def __init__(self, feature_length, aggregated_length, output_length, neighbourhood_sizes,
                 normalisation=tf.nn.l2_normalize):

        self._neighbourhood_sizes = neighbourhood_sizes
        self._aggregated_length = aggregated_length
        self._feature_length = feature_length
        self._output_length = output_length
        self._combined_length = self._feature_length + self._aggregated_length
        self._normalisation = normalisation

    def embedding(self, neighbourhoods):

        # TODO pass through params for aggregators, combiners and normalisers
        aggregators = [agg.Aggregate(self._aggregated_length) for _ in range(len(self._neighbourhood_sizes))]

        combiners = []
        for i in range(len(self._neighbourhood_sizes)):

            weights = init.initialise_glorot_weights((self._combined_length, self._output_length))

            if i + 1 == len(self._neighbourhood_sizes):
                combiner = agg.Combine(weights, activation=lambda x: x, name='combine_linear')
            else:
                combiner = agg.Combine(weights, activation=tf.nn.relu)
            combiners.append(combiner)

        normalisers = [agg.normalise for _ in range(len(self._neighbourhood_sizes))]

        full_representation = agg.chain_aggregate_combine(neighbourhoods, aggregators, combiners, normalisers)
        full_representation = tf.nn.l2_normalize(full_representation, -1)
        return full_representation

    @abc.abstractmethod
    def loss(self, predictions, labels=None):
        pass


class SupervisedModel(Model):

    def __init__(self, labels_length, feature_length, aggregated_length, output_length, neighbourhood_sizes, sigmoid_loss=True, regularisation_weight=0.0, classification_dropout=0.3,
                 classification_activation=tf.nn.relu, classification_regularizer=layers.l2_regularizer(scale=0.1),
                 kernel_initializer=tf.contrib.layers.xavier_initializer(), **kwargs):
        super().__init__(feature_length, aggregated_length, output_length, neighbourhood_sizes, **kwargs)
        self._regularisation_weight = regularisation_weight
        self._sigmoid_loss = sigmoid_loss
        self._classification_dropout = classification_dropout
        self._labels_length = labels_length
        self._classification_activation = classification_activation
        self._classification_regularizer = classification_regularizer
        self._kernel_initializer = kernel_initializer

    def inference(self, neighbourhoods):
        embeddings = self.embedding(neighbourhoods)
        dense_output = tf.layers.dense(embeddings, self._labels_length, activation=self._classification_activation,
                                       use_bias=False, kernel_regularizer=self._classification_regularizer,
                                       kernel_initializer=self._kernel_initializer,
                                       name='classification_dense_layer')
        regularised_output = tf.nn.dropout(dense_output, self._classification_dropout)

        return regularised_output

    def loss(self, predictions, labels=None):

        loss = supervised_loss(predictions, labels, regularisation_weight=self._regularisation_weight,
                               sigmoid_loss=self._sigmoid_loss)
        tf.summary.scalar('loss', loss)
        return loss

    def predict(self, prediction):
        if self._sigmoid_loss:
            return tf.nn.sigmoid(prediction)
        else:
            return tf.nn.softmax(prediction)


def supervised_loss(predictions, labels, regularisation_weight=0.0, sigmoid_loss=True):
    # Get the losses from the various layers
    loss = tf.cast(regularisation_weight * tf.losses.get_regularization_loss(), tf.float64)
    # classification loss
    if sigmoid_loss:
        loss_fn = tf.nn.sigmoid_cross_entropy_with_logits
    else:
        loss_fn = tf.nn.softmax_cross_entropy_with_logits

    loss += tf.reduce_mean(loss_fn(logits=predictions, labels=labels))

    tf.summary.scalar('loss', loss)
    return loss
