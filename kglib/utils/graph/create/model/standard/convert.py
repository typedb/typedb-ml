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

from kglib.utils.graph.create.model.common.convert import concept_dict_to_grakn_graph


def concept_dict_to_grakn_standard_graph(concept_dict, variable_graph):
    return concept_dict_to_grakn_graph(concept_dict, variable_graph, add_role_func=add_role_as_direct_edge)


def add_role_as_direct_edge(grakn_graph, relation, roleplayer, data):
    """
    When adding roles to the graph, here we insert the role as a direct edge, with the type of the role stored in the
    edge data as 'type'
    :param grakn_graph: The graph to add roles to
    :param relation: The relation node
    :param roleplayer: The roleplayer node
    :param data: The data dict, containing the type of the role edge
    """
    grakn_graph.add_edge(relation, roleplayer, **data)
