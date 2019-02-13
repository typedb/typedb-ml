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

import tensorflow_hub as hub
import tensorflow as tf


class TensorFlowHubEncoder:

    def __init__(self, module_url, output_feature_size, name='tf_hub_encoder'):
        print(f'Initialising TensorFlow Hub Encoder op\n'
              f'This may take a long time on its first run, as a pre-trained network module ({module_url}) needs to be '
              f'downloaded...')
        self._embed = hub.Module(module_url)
        print('...Encoder op initialised')
        self._name = name
        self._output_feature_size = output_feature_size

    def __call__(self, features: tf.Tensor):
        with tf.name_scope(name=self._name) as scope:
            shape = features.shape.as_list()
            print(shape)
            flattened_features = tf.reshape(features, [-1])
            flat_embeddings = self._embed(flattened_features)
            shape[-1] = self._output_feature_size
            shape = [-1 if s is None else s for s in shape]
            embeddings = tf.reshape(flat_embeddings, shape)
            return embeddings
