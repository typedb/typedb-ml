#
#  Copyright (C) 2022 Vaticle
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

from itertools import chain


def multidigraph_edge_data_iterator(graph):
    for _, _, _, edge_data in graph.edges(data=True, keys=True):
        yield edge_data


def multidigraph_node_data_iterator(graph):
    for _, node_data in graph.nodes(data=True):
        yield node_data


def multidigraph_data_iterator(graph):
    return chain(multidigraph_node_data_iterator(graph), multidigraph_edge_data_iterator(graph))