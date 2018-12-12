import tensorflow as tf

import kgcn.src.encoder.encode as encode
import kgcn.src.models.embedding as learners
import kgcn.src.neighbourhood.data.sampling.ordered as ordered
import kgcn.src.neighbourhood.data.sampling.sampler as samp
import kgcn.src.neighbourhood.data.traversal
import kgcn.src.preprocess.preprocess
import kgcn.src.preprocess.preprocess as preprocess


class KGCN:
    def __init__(self,
                 neighbour_sample_sizes,
                 features_length,
                 starting_concepts_features_length,
                 aggregated_length,
                 output_length,
                 schema_transaction,
                 batch_size,
                 buffer_size,
                 embedding_normalisation=tf.nn.l2_normalize,
                 sampling_method=ordered.ordered_sample,
                 sampling_limit_factor=1,
                 formatters={'neighbour_value_date': preprocess.datetime_to_unixtime},
                 features_to_exclude=(),
                 include_implicit=False,
                 include_metatypes=False,
                 ):

        self._embedding_normalisation = embedding_normalisation
        self.output_length = output_length
        self.aggregated_length = aggregated_length
        self.neighbour_sample_sizes = neighbour_sample_sizes

        self.feature_lengths = [features_length] * len(self.neighbour_sample_sizes)
        self.feature_lengths[-1] = starting_concepts_features_length
        print(f'feature lengths: {self.feature_lengths}')

        self._schema_transaction = schema_transaction

        self.include_metatypes = include_metatypes
        self.include_implicit = include_implicit
        self.batch_size = batch_size
        self._buffer_size = buffer_size
        self._formatters = formatters
        self._features_to_exclude = features_to_exclude

        traversal_samplers = []
        for sample_size in neighbour_sample_sizes:
            traversal_samplers.append(
                samp.Sampler(sample_size, sampling_method, limit=int(sample_size * sampling_limit_factor)))

        self._traverser = kgcn.src.preprocess.preprocess.Traverser(traversal_samplers)

    def input_fn(self, session, concepts):
        raw_array_depths = self._traverser(session, concepts)
        raw_array_depths = preprocess.apply_operations(raw_array_depths, self._formatters)
        return raw_array_depths

    def build_dataset(self):
        features_to_exclude = {feat_name: None for feat_name in self._features_to_exclude}
        dataset_builder = preprocess.DatasetBuilder(self.neighbour_sample_sizes, **features_to_exclude)
        arrays_dataset, placeholders = dataset_builder()
        return arrays_dataset, placeholders

    def batch_dataset(self, dataset):
        # buffer_size = batch_size = tf.cast(self._batch_size, tf.int64)
        dataset = dataset.shuffle(buffer_size=self._buffer_size, seed=5, reshuffle_each_iteration=True)
        dataset = dataset.batch(batch_size=self.batch_size).repeat()

        dataset_iterator = dataset.make_initializable_iterator()
        dataset_initializer = dataset_iterator.initializer
        return dataset_initializer, dataset_iterator

    def encode(self, arrays):
        encoder = encode.Encoder(self._schema_transaction, self.include_implicit, self.include_metatypes)
        return encoder(arrays)

    def embed(self, encoded_arrays):
        embedder = learners.Embedder(self.feature_lengths, self.aggregated_length, self.output_length,
                                     self.neighbour_sample_sizes, normalisation=self._embedding_normalisation)
        return embedder(encoded_arrays)

    def embed_with_labels(self, num_classes):
        labels_placeholder = tf.placeholder(tf.float32, shape=(self.batch_size, num_classes), name='labels_input')
        labels_dataset = tf.data.Dataset.from_tensor_slices(labels_placeholder)

        # Pipeline
        arrays_dataset, array_placeholders = self.build_dataset()

        combined_dataset = tf.data.Dataset.zip((arrays_dataset, labels_dataset))

        dataset_initializer, dataset_iterator = self.batch_dataset(combined_dataset)

        # TODO This should be called in a loop when using more than one batch
        batch_arrays, labels = dataset_iterator.get_next()

        encoded_arrays = self.encode(batch_arrays)

        embeddings = self.embed(encoded_arrays)
        tf.summary.histogram('evaluate/embeddings', embeddings)

        return embeddings, labels, dataset_initializer, array_placeholders, labels_placeholder
