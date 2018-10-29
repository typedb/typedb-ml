import tensorflow as tf
import kgcn.src.neighbourhood.schema.traversal as trav
import kgcn.src.neighbourhood.schema.strategy as schema_strat
import numpy as np


def _build_adjacency_matrix(schema_concept_super_types):
    adj = np.zeros((len(schema_concept_super_types), len(schema_concept_super_types)))
    schema_concepts = list(schema_concept_super_types.keys())
    for schema_concept_index, (schema_concept, super_types) in enumerate(schema_concept_super_types.items()):
        for super_type in super_types:
            super_type_index = schema_concepts.index(super_type)
            adj[schema_concept_index, super_type_index] = 1
    return schema_concepts, adj


class MultiHotSchemaTypeEncoder:
    def __init__(self, traversal_executor, schema_strategy: schema_strat.SchemaTraversalStrategy):

        if schema_strategy.kind == "thing":
            query = trav.GET_THING_TYPES_QUERY
        else:
            query = trav.GET_ROLE_TYPES_QUERY

        schema_concept_types = \
            traversal_executor.get_schema_concept_types(query,
                                                        include_implicit=schema_strategy.include_implicit,
                                                        include_metatypes=schema_strategy.include_metatypes)

        schema_concept_super_types = trav.get_sups_labels_per_type(schema_concept_types, include_self=True,
                                                                   include_metatypes=schema_strategy.include_metatypes)

        self._lookup_table = tf.contrib.lookup.index_table_from_tensor(mapping=schema_concept_types, num_oov_buckets=0,
                                                                       default_value=-1)
        self._multi_hot_embeddings = _build_adjacency_matrix(schema_concept_super_types)

    def __call__(self, schema_type_features):

        # First go from string features to the indices of those types
        type_indices = self._lookup_table.lookup(schema_type_features)

        # Then look up the row of the multi-hot embeddings to use for each
        embeddings = tf.nn.embedding_lookup(self._multi_hot_embeddings, type_indices)
        embeddings = tf.squeeze(embeddings, axis=-2)
        return embeddings, type_indices
