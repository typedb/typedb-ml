import tensorflow as tf


def aggregator(neighbour_features, aggregated_length, reduction=tf.reduce_max, activation=tf.nn.relu, dropout=0.3,
               name=None):
    """
    Take a tensor that describes the features (aggregated or otherwise) of a set of neighbours and aggregate
    them through a dense layer and order-independent pooling/reduction

    :param neighbour_features: the neighbours' features, shape (num_neighbours, neighbour_feat_length)
    :param aggregated_length: the number of elements in the representation created
    :param reduction: order-independent method of pooling the response for each neighbour
    :param activation: activation function for the included dense layer
    :param dropout: quantity of dropout regularisation on the output of the included dense layer
    :param name: this op's name for use in graph visualisation
    :return: aggregated representation of neighbours, shape (aggregated_length, 1)
    """

    with tf.name_scope(name, "aggregator") as scope:
        dense_output = tf.layers.dense(neighbour_features, aggregated_length, activation, use_bias=False,
                                       name='dense_layer')

        # Use dropout on output from the dense layer to prevent overfitting
        regularised_output = tf.nn.dropout(dense_output, dropout)

        # Use max-pooling (or similar) to aggregate the results for each neighbour. This is an important operation
        # since the order of the neighbours isn't considered, which is a property we need Note that this is reducing
        # not pooling, which is equivalent to having a pool size of num_neighbours
        reduced_output = reduction(regularised_output, axis=0)

        # Get the output from shape (1, neighbour_feat_length) to (neighbour_feat_length, 1)
        final_output = tf.transpose(reduced_output)

        return final_output
