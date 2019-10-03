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
from kglib.kgcn.models.typewise import TypewiseEncoder


def common_embedding(features, num_types, type_embedding_dim):
    preexistance_feat = tf.expand_dims(tf.cast(features[:, 0], dtype=tf.float32), axis=1)
    type_embedder = snt.Embed(num_types, type_embedding_dim)
    type_embedding = type_embedder(tf.cast(features[:, 1], tf.int32))
    tf.summary.histogram('type_embedding_histogram', type_embedding)
    return tf.concat([preexistance_feat, type_embedding], axis=1)


def attribute_embedding(features, attr_encoders, attr_embedding_dim):
    typewise_attribute_encoder = TypewiseEncoder(attr_encoders, attr_embedding_dim)
    attr_embedding = typewise_attribute_encoder(features[:, 1:])
    tf.summary.histogram('attribute_embedding_histogram', attr_embedding)
    return attr_embedding


def node_embedding(features, num_types, type_embedding_dim, attr_encoders, attr_embedding_dim):
    return tf.concat([common_embedding(features, num_types, type_embedding_dim),
                      attribute_embedding(features, attr_encoders, attr_embedding_dim)], axis=1)
