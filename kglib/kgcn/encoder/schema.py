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

import sys

import numpy as np
import tensorflow as tf


def _build_adjacency_matrix(schema_traversal):
    adj = np.zeros((len(schema_traversal), len(schema_traversal)), dtype=np.float32)
    schema_concepts = list(schema_traversal.keys())
    for schema_concept_index, (schema_concept, super_types) in enumerate(schema_traversal.items()):
        for super_type in super_types:
            super_type_index = schema_concepts.index(super_type)
            adj[schema_concept_index, super_type_index] = 1
    return adj


class MultiHotSchemaTypeEncoder:
    def __init__(self, schema_traversal, default_value=-1, dtype=tf.string, name=None):
        self._name = name
        if self._name is not None:
            name = self._name + '_init'

        with tf.name_scope(name) as scope:
            self._dtype = dtype
            print(list(schema_traversal.keys()))
            self._schema_concept_type_labels = tf.convert_to_tensor(list(schema_traversal.keys()), dtype=self._dtype)
            self._lookup_table = tf.contrib.lookup.index_table_from_tensor(mapping=self._schema_concept_type_labels,
                                                                           num_oov_buckets=0, default_value=default_value,
                                                                           dtype=self._dtype)
            self._multi_hot_embeddings = _build_adjacency_matrix(schema_traversal)

    def __call__(self, schema_type_features):

        with tf.name_scope(self._name) as scope:
            try:
                # First go from string features to the indices of those types
                type_indices = self._lookup_table.lookup(schema_type_features)
            except AttributeError as e:
                raise type(e)(
                    str(e) + "\nExpecting to look up the same value type as stored, check for mismatch").with_traceback(
                    sys.exc_info()[2])

            # Then look up the row of the multi-hot embeddings to use for each
            embeddings = tf.nn.embedding_lookup(self._multi_hot_embeddings, type_indices)
            embeddings = tf.squeeze(embeddings, axis=-2)
            return embeddings
