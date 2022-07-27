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

import networkx as nx


class QueryGraph(nx.MultiDiGraph):
    """
    A custom graph to represent a query. Has additional helper methods specific to adding TypeQL patterns.
    """

    def add_vars(self, vars):
        """
        Add variables, stored as nodes in the graph
        Args:
            vars: String variables

        Returns:
            self
        """
        for var in vars:
            self.add_node(var)
        return self

    def add_has_edge(self, owner_var, attribute_var):
        """
        Add a "has" edge to represent ownership of an attribute
        Args:
            owner_var: The variable of the owner
            attribute_var: The variable of the owned attribute

        Returns:
            self
        """
        self.add_edge(owner_var, attribute_var, type='has')
        return self

    def add_role_edge(self, relation_var, roleplayer_var, role_label):
        """
        Add an edge to represent the role a variable plays in a relation
        Args:
            relation_var: The variable of the relation
            roleplayer_var: The variable of the roleplayer in the relation
            role_label: The role the roleplayer plays in the relation

        Returns:
            self
        """
        self.add_edge(relation_var, roleplayer_var, type=role_label)
        return self


class Query:
    def __init__(self, graph: nx.MultiDiGraph, string: str):
        self.graph = graph
        self.string = string

    def __str__(self):
        return self.string
