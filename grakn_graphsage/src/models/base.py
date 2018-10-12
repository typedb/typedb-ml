import tensorflow as tf
import grakn_graphsage.src.models.aggregation as agg
import grakn_graphsage.src.models.initialise as init


class Model:
    def __init__(self, neighbourhood_sizes, feature_length, aggregated_length, output_length,
                 normalisation=tf.nn.l2_normalize):

        self._neighbourhood_sizes = neighbourhood_sizes
        self._aggregated_length = aggregated_length
        self._feature_length = feature_length
        self._output_length = output_length
        self._combined_length = self._output_length + self._aggregated_length
        self._combined_length = self._feature_length + self._aggregated_length
        self._normalisation = normalisation

    def inference(self, neighbourhood):

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
        full_representation = agg.chain_aggregate_combine(neighbourhood, aggregators, combiners, normalisers)

        return full_representation