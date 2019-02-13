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


def one_hot_boolean_encode(boolean_features_as_integers, name='boolean_1_hot'):
    """
    :param boolean_features_as_integers: A tensor of booleans represented as integers, with final dimension 1.
    Expects 0 to indicate False, 1 to indicate True, -1 to indicate neither True or False
    :return: One-hot boolean encoding tensor of same shape as `boolean_features` but with last dimension 2 (
    one-hot size of boolean)
    """
    with tf.name_scope(name=name) as scope:
        return tf.squeeze(tf.one_hot(boolean_features_as_integers, 2, on_value=1, off_value=0), axis=-2)
