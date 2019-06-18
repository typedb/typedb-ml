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

import networkx as nx

from kglib.kgcn.core.ingest.traverse.data.context.neighbour import Role


def concept_dict_to_grakn_math_graph(concept_dict, variable_graph):
    grakn_math_graph = nx.MultiDiGraph()
    node_to_var = {}

    if set(variable_graph.nodes()) != set(concept_dict.keys()):
        raise ValueError('The variables in the variable_graph must match those in the concept_dict')

    # This assumes that all variables are nodes, which would not be the case for variable roles
    for variable, thing in concept_dict.items():
        grakn_math_graph.add_node(thing)

        # Record the mapping of nodes from one graph to the other
        assert variable not in node_to_var
        node_to_var[variable] = thing

    for sending_var, receiving_var, data in variable_graph.edges(data=True):

        sender = node_to_var[sending_var]
        receiver = node_to_var[receiving_var]

        if sender.base_type_label != 'relation' and not (
                receiver.base_type_label == 'attribute' and data['type'] == 'has'):
            raise ValueError('An edge in the variable_graph originates from a non-relation, check the variable_graph!')

        if data['type'] == 'has':
            grakn_math_graph.add_edge(sender, receiver, type=data['type'])
        else:
            role = Role(sender, receiver, data['type'])
            grakn_math_graph.add_node(role)
            grakn_math_graph.add_edge(sender, role, type='relates')
            grakn_math_graph.add_edge(receiver, role, type='plays')

    return grakn_math_graph


