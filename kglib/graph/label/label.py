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


"""
Use Cases:

Scenario 1
1. Create a graph from queries
2. Apply ground truth labels to the graph by supplying specific concepts and role edges that should be labelled
positively, with the rest assumed to be negative

Scenario 2
1. Create a graph from queries
2. Create graph2 from the same queries, but now a rule has been added so that extra concepts are inferred
3. Find the difference of the two graphs, and label that difference differently to the rest

Scenario 3 (favoured scenario)
1. Create a graph from queries, with a rule and necessary schema Types added to infer hypothetical concepts to use as
negative examples
2. Label the hypothetical Concepts as negative, and label the positive examples
3. Alter the graph to give all positive and negative examples the same Type, so that the learner can't use the
hypothetical Type to discriminate between the positive and negative examples

"""


def label_concepts(graph, concepts_to_label, labels_to_apply):
    """
    Labels Concepts in a graph
    :param graph: The graph to update
    :param concepts_to_label: The Concepts (nodes) in the graph to label
    :param labels_to_apply: The labels to update with
    """
    for node in concepts_to_label:
        graph.nodes[node].update(labels_to_apply)


def label_direct_roles(graph, roles_to_label, labels_to_apply):
    """
    Labels Role edges as found in a standard Grakn model
    :param graph: The graph to update
    :param roles_to_label: The Role edges in the graph to label
    :param labels_to_apply: The labels to update with
    """

    for role in roles_to_label:
        for sender, receiver, data in graph.edges(data=True):
            if role.sender == sender and role.receiver == receiver \
                    and graph.edges[sender, receiver, 0]['type'] == role.type_label:
                data.update(labels_to_apply)
