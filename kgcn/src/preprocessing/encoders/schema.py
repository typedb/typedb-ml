import tensorflow as tf


class SchemaTypeEncoder:
    def __init__(self, schema_types, multi_hot_embeddings):
        self._lookup_table = tf.contrib.lookup.index_table_from_tensor(schema_types)
        self._multi_hot_embeddings = multi_hot_embeddings

    def __call__(self, schema_type_features):

        # First go from string features to the indices of those types
        type_indices = self._lookup_table.lookup(schema_type_features)

        # Then look up the row of the multi-hot embeddings to use for each
        embeddings = tf.nn.embedding_lookup(self._multi_hot_embeddings, type_indices)
        embeddings = tf.squeeze(embeddings, axis=-2)
        return embeddings, type_indices
