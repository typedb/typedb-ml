import tensorflow as tf
import numpy as np


def _build_adjacency_matrix(schema_traversal):
    adj = np.zeros((len(schema_traversal), len(schema_traversal)))
    schema_concepts = list(schema_traversal.keys())
    for schema_concept_index, (schema_concept, super_types) in enumerate(schema_traversal.items()):
        for super_type in super_types:
            super_type_index = schema_concepts.index(super_type)
            adj[schema_concept_index, super_type_index] = 1
    return adj


class MultiHotSchemaTypeEncoder:
    def __init__(self, schema_traversal):

        schema_concept_type_labels = list(schema_traversal.keys())
        self._lookup_table = tf.contrib.lookup.index_table_from_tensor(mapping=schema_concept_type_labels,
                                                                       num_oov_buckets=0, default_value=-1)
        self._multi_hot_embeddings = _build_adjacency_matrix(schema_traversal)

    def __call__(self, schema_type_features):

        # First go from string features to the indices of those types
        type_indices = self._lookup_table.lookup(schema_type_features)

        # Then look up the row of the multi-hot embeddings to use for each
        embeddings = tf.nn.embedding_lookup(self._multi_hot_embeddings, type_indices)
        embeddings = tf.squeeze(embeddings, axis=-2)
        return embeddings, type_indices
