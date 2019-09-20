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

import abc

import sonnet as snt
import tensorflow as tf


class Attribute(snt.AbstractModule, abc.ABC):
    """
    Abstract base class for Attribute value embedding models
    """
    def __init__(self, attr_embedding_dim, name='AttributeEmbedder'):
        super(Attribute, self).__init__(name=name)
        self._attr_embedding_dim = attr_embedding_dim


class ContinuousAttribute(Attribute):
    def __init__(self, attr_embedding_dim, name='CategoricalAttributeEmbedder'):
        super(ContinuousAttribute, self).__init__(attr_embedding_dim, name=name)

    def _build(self, attribute_value):
        return snt.Sequential([
            snt.nets.MLP([self._attr_embedding_dim] * 3, activate_final=True),
            snt.LayerNorm()
        ])(attribute_value)


class CategoricalAttribute(Attribute):
    def __init__(self, num_categories, attr_embedding_dim, name='CategoricalAttributeEmbedder'):
        super(CategoricalAttribute, self).__init__(attr_embedding_dim, name=name)

        self._num_categories = num_categories

    def _build(self, attribute_value):
        int_attribute_value = tf.cast(attribute_value, dtype=tf.int32)
        embedding = snt.Embed(self._num_categories, self._attr_embedding_dim)(int_attribute_value)
        return tf.squeeze(embedding, axis=1)


class BlankAttribute(Attribute):

    def __init__(self, attr_embedding_dim, name='BlankAttributeEmbedder'):
        super(BlankAttribute, self).__init__(attr_embedding_dim, name=name)

    def _build(self, attribute_value):
        shape = tf.stack([tf.shape(attribute_value)[0], self._attr_embedding_dim])

        encoded_features = tf.zeros(shape, dtype=tf.float32)
        return encoded_features
