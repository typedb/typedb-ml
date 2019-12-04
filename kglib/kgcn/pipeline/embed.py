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

from kglib.kgcn.models.attribute import CategoricalAttribute, ContinuousAttribute, BlankAttribute


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