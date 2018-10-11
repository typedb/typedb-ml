import tensorflow as tf


def aggregate(neighbour_features, aggregated_length, reduction=tf.reduce_max, activation=tf.nn.relu, dropout=0.3,
              name=None):
    """
    Take a tensor that describes the features (aggregated or otherwise) of a set of neighbours and aggregate
    them through a dense layer and order-independent pooling/reduction

    :param neighbour_features: the neighbours' features, shape (num_neighbours, neighbour_feat_length)
    :param aggregated_length: the number of elements in the representation created
    :param reduction: order-independent method of pooling the response for each neighbour
    :param activation: activation function for the included dense layer
    :param dropout: quantity of dropout regularisation on the output of the included dense layer
    :param name: Name for the operation (optional).
    :return: aggregated representation of neighbours, shape (1, aggregated_length)
    """

    with tf.name_scope(name, default_name="aggregate") as scope:
        dense_output = tf.layers.dense(neighbour_features, aggregated_length, activation, use_bias=False,
                                       name='dense_layer')

        # Use dropout on output from the dense layer to prevent overfitting
        regularised_output = tf.nn.dropout(dense_output, dropout)

        # Use max-pooling (or similar) to aggregate the results for each neighbour. This is an important operation
        # since the order of the neighbours isn't considered, which is a property we need Note that this is reducing
        # not pooling, which is equivalent to having a pool size of num_neighbours
        reduced_output = reduction(regularised_output, axis=0)

        # # Get the output from shape (1, neighbour_feat_length) to (neighbour_feat_length, 1)
        # final_output = tf.transpose(reduced_output)

        return reduced_output


def combine(target_features, neighbour_representations, weights, activation=tf.nn.relu, name=None):
    """
    Combine the results of neighbour aggregation with the features of target nodes. Combine using concatenation,
    multiplication with a weight matrix (GCN approach) and process with some activation function
    :param target_features: the features of the target nodes
    :param neighbour_representations: the representations of the neighbours of the target nodes, one representation
    for each target
    :param weights: weight matrix, shape (combined_length, output_length)
    :param activation: activation function performed on the
    :param name: Name for the operation (optional).
    :return: full representations of target nodes
    """
    with tf.name_scope(name, default_name="combine") as scope:
        concatenated_features = tf.concat([target_features, neighbour_representations], axis=1)

        weighted_output = tf.matmul(concatenated_features, weights, name='apply_weights')

        return activation(weighted_output)
