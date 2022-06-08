#
#  Copyright (C) 2021 Vaticle
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
import sonnet as snt

from kglib.kgcn_tensorflow.models.attribute import CategoricalAttribute, ContinuousAttribute, BlankAttribute
from kglib.kgcn_tensorflow.models.typewise import TypewiseEncoder


class ThingEmbedder(snt.AbstractModule):
    def __init__(self, node_types, type_embedding_dim, attr_embedding_dim, categorical_attributes,
                 continuous_attributes, name="ThingEmbedder"):
        super(ThingEmbedder, self).__init__(name=name)

        self._node_types = node_types
        self._type_embedding_dim = type_embedding_dim
        self._attr_embedding_dim = attr_embedding_dim

        # Create embedders for the different attribute types
        self._attr_embedders = dict()

        if categorical_attributes is not None:
            self._attr_embedders.update(
                construct_categorical_embedders(node_types, attr_embedding_dim, categorical_attributes))

        if continuous_attributes is not None:
            self._attr_embedders.update(
                construct_continuous_embedders(node_types, attr_embedding_dim, continuous_attributes))

        self._attr_embedders.update(
            construct_non_attribute_embedders(node_types, attr_embedding_dim, categorical_attributes,
                                              continuous_attributes))

    def _build(self, features):
        return tf.concat([embed_type(features, len(self._node_types), self._type_embedding_dim),
                          embed_attribute(features, self._attr_embedders, self._attr_embedding_dim)], axis=1)


class RoleEmbedder(snt.AbstractModule):
    def __init__(self, num_edge_types, type_embedding_dim, name="RoleEmbedder"):
        super(RoleEmbedder, self).__init__(name=name)
        self._num_edge_types = num_edge_types
        self._type_embedding_dim = type_embedding_dim

    def _build(self, features):
        return embed_type(features, self._num_edge_types, self._type_embedding_dim)


def embed_type(features, num_types, type_embedding_dim):
    preexistance_feat = tf.expand_dims(tf.cast(features[:, 0], dtype=tf.float32), axis=1)
    type_embedder = snt.Embed(num_types, type_embedding_dim)
    norm = snt.LayerNorm()
    type_embedding = norm(type_embedder(tf.cast(features[:, 1], tf.int32)))
    tf.summary.histogram('type_embedding_histogram', type_embedding)
    return tf.concat([preexistance_feat, type_embedding], axis=1)


def embed_attribute(features, attr_encoders, attr_embedding_dim):
    typewise_attribute_encoder = TypewiseEncoder(attr_encoders, attr_embedding_dim)
    attr_embedding = typewise_attribute_encoder(features[:, 1:])
    tf.summary.histogram('attribute_embedding_histogram', attr_embedding)
    return attr_embedding


def construct_categorical_embedders(node_types, attr_embedding_dim, categorical_attributes):
    attr_embedders = dict()

    # Construct attribute embedders
    for attribute_type, categories in categorical_attributes.items():

        attr_typ_index = node_types.index(attribute_type)

        def make_embedder():
            return CategoricalAttribute(len(categories), attr_embedding_dim,
                                        name=attribute_type + '_cat_embedder')

        # Record the embedder, and the index of the type that it should encode
        attr_embedders[make_embedder] = [attr_typ_index]

    return attr_embedders


def construct_continuous_embedders(node_types, attr_embedding_dim, continuous_attributes):
    attr_embedders = dict()

    # Construct attribute embedders
    for attribute_type in continuous_attributes.keys():

        attr_typ_index = node_types.index(attribute_type)

        def make_embedder():
            return ContinuousAttribute(attr_embedding_dim, name=attribute_type + '_cat_embedder')

        # Record the embedder, and the index of the type that it should encode
        attr_embedders[make_embedder] = [attr_typ_index]

    return attr_embedders


def construct_non_attribute_embedders(node_types, attr_embedding_dim, categorical_attributes, continuous_attributes):

    attribute_names = list(categorical_attributes.keys())
    attribute_names.extend(list(continuous_attributes.keys()))

    non_attribute_nodes = []
    for i, type in enumerate(node_types):
        if type not in attribute_names:
            non_attribute_nodes.append(i)

    # All entities and relations (non-attributes) also need an embedder with matching output dimension, which does
    # nothing. This is provided as a list of their indices
    def make_blank_embedder():
        return BlankAttribute(attr_embedding_dim)

    attr_embedders = dict()

    if len(non_attribute_nodes) > 0:
        attr_embedders[make_blank_embedder] = non_attribute_nodes
    return attr_embedders
