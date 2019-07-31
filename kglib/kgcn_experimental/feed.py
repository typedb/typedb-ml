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

from graph_nets import utils_tf, utils_np


def create_placeholders(input_graphs, target_graphs):
    """
    Creates placeholders for the model training and evaluation.
    Returns:
    input_ph: The input graph's placeholders, as a graph namedtuple.
    target_ph: The target graph's placeholders, as a graph namedtuple.
    """
    input_ph = utils_tf.placeholders_from_networkxs(input_graphs)
    target_ph = utils_tf.placeholders_from_networkxs(target_graphs)
    return input_ph, target_ph


def create_feed_dict(tr_or_ge, tr_ge_split, input_ph, target_ph, inputs, targets, raw_graphs):
    """Creates the feed dict for the placeholders for the model training and evaluation.

    Args:
    input_ph: The input graph's placeholders, as a graph namedtuple.
    target_ph: The target graph's placeholders, as a graph namedtuple.

    Returns:
    feed_dict: The feed `dict` of input and target placeholders and data.
    raw_graphs: The `dict` of raw networkx graphs.
    :param tr_ge_split:
    """
    if tr_or_ge == "tr":
        start = 0
        end = tr_ge_split
    elif tr_or_ge == "ge":
        start = tr_ge_split
        end = None

    input_graphs = utils_np.networkxs_to_graphs_tuple(inputs[start:end])
    target_graphs = utils_np.networkxs_to_graphs_tuple(targets[start:end])
    feed_dict = {input_ph: input_graphs, target_ph: target_graphs}
    return feed_dict, raw_graphs[start:end]
