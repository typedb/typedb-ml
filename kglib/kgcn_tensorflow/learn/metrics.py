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

import numpy as np
from graph_nets import utils_np

from scipy.special import softmax


def compute_accuracy(target, output, use_nodes=True, use_edges=True):
    """Calculate model accuracy.

    Returns the number of elements correctly predicted to exist, and the number of completely correct graphs
    (100% correct predictions).

    Args:
        target: A `graphs.GraphsTuple` that contains the target graph.
        output: A `graphs.GraphsTuple` that contains the output graph.
        use_nodes: A `bool` indicator of whether to compute node accuracy or not.
        use_edges: A `bool` indicator of whether to compute edge accuracy or not.

    Returns:
        correct: A `float` fraction of correctly labeled nodes/edges.
        solved: A `float` fraction of graphs that are completely correctly labeled.

    Raises:
        ValueError: Nodes or edges (or both) must be used
    """
    if not use_nodes and not use_edges:
        raise ValueError("Nodes or edges (or both) must be used")
    tdds = utils_np.graphs_tuple_to_data_dicts(target)
    odds = utils_np.graphs_tuple_to_data_dicts(output)
    cs = []
    ss = []
    for td, od in zip(tdds, odds):
        xn = np.argmax(td["nodes"], axis=-1)
        yn = np.argmax(od["nodes"], axis=-1)
        xe = np.argmax(td["edges"], axis=-1)
        ye = np.argmax(od["edges"], axis=-1)
        c = []
        if use_nodes:
            c.append(xn == yn)
        if use_edges:
            c.append(xe == ye)
        c = np.concatenate(c, axis=0)
        s = np.all(c)
        cs.append(c)
        ss.append(s)
    correct = np.mean(np.concatenate(cs, axis=0))
    solved = np.mean(np.stack(ss))
    return correct, solved


def existence_accuracy(target, output, use_nodes=True, use_edges=True):
    if not use_nodes and not use_edges:
        raise ValueError("Nodes or edges (or both) must be used")
    tdds = utils_np.graphs_tuple_to_data_dicts(target)
    odds = utils_np.graphs_tuple_to_data_dicts(output)
    cs = []
    ss = []
    for td, od in zip(tdds, odds):

        nodes_to_predict = td["nodes"][:, 0] == 0
        xn = np.argmax(td["nodes"][:, 1:], axis=-1)
        xn = xn[nodes_to_predict]
        yn = np.argmax(softmax(od["nodes"][:, 1:], axis=1), axis=-1)
        yn = yn[nodes_to_predict]

        edges_to_predict = td["edges"][:, 0] == 0
        xe = np.argmax(td["edges"][:, 1:], axis=-1)
        xe = xe[edges_to_predict]
        ye = np.argmax(softmax(od["edges"][:, 1:], axis=1), axis=-1)
        ye = ye[edges_to_predict]

        c = []
        if use_nodes:
            c.append(xn == yn)
        if use_edges:
            c.append(xe == ye)
        c = np.concatenate(c, axis=0)
        s = np.all(c)
        cs.append(c)
        ss.append(s)
    correct = np.mean(np.concatenate(cs, axis=0))
    solved = np.mean(np.stack(ss))
    return correct, solved
