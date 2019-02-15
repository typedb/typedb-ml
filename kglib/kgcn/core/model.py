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

import tensorflow as tf

import kglib.kgcn.core.ingest.encode.encode as encode
import kglib.kgcn.core.ingest.traverse.data.context as context
import kglib.kgcn.core.ingest.traverse.data.neighbour as neighbour
import kglib.kgcn.core.nn.embed as embed
import kglib.kgcn.core.ingest.traverse.data.sample.ordered as ordered
import kglib.kgcn.core.ingest.traverse.data.sample.sample as sample
import kglib.kgcn.core.ingest.preprocess.preprocess as preprocess
import kglib.kgcn.core.ingest.preprocess.context_array as context_array


class KGCN:
    def __init__(self,
                 neighbour_sample_sizes,
                 features_size,
                 example_concepts_features_size,
                 aggregated_size,
                 embedding_size,
                 schema_encoding_transaction,
                 batch_size,
                 embedding_normalisation=tf.nn.l2_normalize,
                 neighbour_sampling_method=ordered.ordered_sample,
                 neighbour_sampling_limit_factor=1,
                 formatters={'neighbour_value_date': preprocess.datetime_to_unixtime},
                 features_to_exclude=()):

        self._embedding_normalisation = embedding_normalisation
        self.embedding_size = embedding_size
        self.aggregated_size = aggregated_size
        self.neighbour_sample_sizes = neighbour_sample_sizes

        self.feature_sizes = [features_size] * len(self.neighbour_sample_sizes)
        self.feature_sizes[-1] = example_concepts_features_size
        print(f'feature sizes: {self.feature_sizes}')

        self._schema_encoding_transaction = schema_encoding_transaction
        self._encode = encode.Encoder(self._schema_encoding_transaction)

        self.batch_size = batch_size
        self._formatters = formatters
        self._features_to_exclude = features_to_exclude

        traversal_samplers = []
        for sample_size in neighbour_sample_sizes:
            traversal_samplers.append(
                sample.Sampler(sample_size, neighbour_sampling_method, limit=int(sample_size * neighbour_sampling_limit_factor)))

        self._array_builder = context_array.ContextArrayBuilder(neighbour_sample_sizes)

        self._context_builder = context.ContextBuilder(traversal_samplers)

        self._embed = embed.Embedder(self.feature_sizes, self.aggregated_size, self.embedding_size,
                                     self.neighbour_sample_sizes, normalisation=self._embedding_normalisation)

        features_to_exclude = {feat_name: None for feat_name in self._features_to_exclude}
        self.neighbourhood_dataset, self.array_placeholders = preprocess.build_dataset(self.neighbour_sample_sizes,
                                                                                       **features_to_exclude)

    def input_fn(self, session, concepts):
        context_batch = self._context_builder.build_batch(session, concepts)
        context_arrays = self._array_builder.build_context_arrays(context_batch)
        formatted_neighbourhoods = preprocess.apply_operations(context_arrays, self._formatters)
        return formatted_neighbourhoods

    def embed(self, *additional_datasets):

        datasets = list(additional_datasets)
        datasets.insert(0, self.neighbourhood_dataset)

        combined_dataset = tf.data.Dataset.zip(tuple(datasets))

        dataset_initializer, dataset_iterator = _shuffle_and_batch_dataset(combined_dataset, self.batch_size)

        # TODO This should be called in a loop when using more than one batch
        next_batch = dataset_iterator.get_next()

        # The neighbourhood_dataset will be the first item since that's the order they were zipped
        encoded_arrays = self._encode(next_batch[0])

        embeddings = self._embed(encoded_arrays)
        tf.summary.histogram('evaluate/embeddings', embeddings)

        return embeddings, next_batch[1:], dataset_initializer, self.array_placeholders


def _shuffle_and_batch_dataset(dataset, batch_size, seed=5):
    dataset = dataset.shuffle(buffer_size=batch_size, seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size=batch_size).repeat()

    dataset_iterator = dataset.make_initializable_iterator()
    dataset_initializer = dataset_iterator.initializer
    return dataset_initializer, dataset_iterator
