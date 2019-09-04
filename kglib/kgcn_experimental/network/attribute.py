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

import sonnet as snt
import tensorflow as tf


class CategoricalAttribute(snt.AbstractModule):
    def __init__(self, num_categories, attr_embedding_dim, name='CategoricalAttribute'):
        super(CategoricalAttribute, self).__init__(name=name)

        self._attr_embedding_dim = attr_embedding_dim
        self._num_categories = num_categories

    def _build(self, inputs):
        int_inputs = tf.cast(inputs, dtype=tf.int32)
        embedding = snt.Embed(self._num_categories, self._attr_embedding_dim)(int_inputs)
        return tf.squeeze(embedding, axis=1)
