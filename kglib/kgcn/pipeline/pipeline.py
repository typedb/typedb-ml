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
import numpy as np
from graph_nets.utils_np import graphs_tuple_to_networkxs

from kglib.kgcn.learn.learn import KGCNLearner
from kglib.kgcn.models.attribute import ContinuousAttribute, CategoricalAttribute, BlankAttribute
from kglib.kgcn.models.core import softmax, KGCN
from kglib.kgcn.pipeline.encode import encode_types, create_input_graph, create_target_graph
from kglib.kgcn.pipeline.utils import apply_logits_to_graphs, duplicate_edges_in_reverse
from kglib.kgcn.plot.plotting import plot_across_training, plot_predictions
from kglib.utils.graph.iterate import multidigraph_node_data_iterator, multidigraph_data_iterator, \
    multidigraph_edge_data_iterator


def pipeline(graphs,
             tr_ge_split,
             node_types,
             edge_types,
             num_processing_steps_tr=10,
             num_processing_steps_ge=10,
             num_training_iterations=10000,
             continuous_attributes=None,
             categorical_attributes=None,
             type_embedding_dim=5,
             attr_embedding_dim=6,
             edge_output_size=3,
             node_output_size=3,
             output_dir=None):

    ############################################################
    # Manipulate the graph data
    ############################################################

    # Encode attribute values
    for graph in graphs:
        for node_data in multidigraph_node_data_iterator(graph):
            typ = node_data['type']

            if categorical_attributes is not None and typ in categorical_attributes.keys():
                # Add the integer value of the category for each categorical attribute instance
                category_values = categorical_attributes[typ]
                node_data['encoded_value'] = category_values.index(node_data['value'])

            elif continuous_attributes is not None and typ in continuous_attributes.keys():
                min_val, max_val = continuous_attributes[typ]
                node_data['encoded_value'] = (node_data['value'] - min_val) / (max_val - min_val)

            else:
                node_data['encoded_value'] = 0

        for edge_data in multidigraph_edge_data_iterator(graph):
            edge_data['encoded_value'] = 0

    indexed_graphs = [nx.convert_node_labels_to_integers(graph, label_attribute='concept') for graph in graphs]
    graphs = [duplicate_edges_in_reverse(graph) for graph in indexed_graphs]

    graphs = [encode_types(graph, node_types, edge_types) for graph in graphs]

    input_graphs = [create_input_graph(graph) for graph in graphs]
    target_graphs = [create_target_graph(graph) for graph in graphs]

    tr_input_graphs = input_graphs[:tr_ge_split]
    tr_target_graphs = target_graphs[:tr_ge_split]
    ge_input_graphs = input_graphs[tr_ge_split:]
    ge_target_graphs = target_graphs[tr_ge_split:]

    ############################################################
    # Build and run the KGCN
    ############################################################

    attr_embedders = configure_embedders(node_types, attr_embedding_dim, categorical_attributes, continuous_attributes)

    kgcn = KGCN(len(node_types),
                len(edge_types),
                type_embedding_dim,
                attr_embedding_dim,
                attr_embedders,
                edge_output_size=edge_output_size,
                node_output_size=node_output_size)

    learner = KGCNLearner(kgcn,
                          num_processing_steps_tr=num_processing_steps_tr,
                          num_processing_steps_ge=num_processing_steps_ge)

    train_values, test_values, tr_info = learner(tr_input_graphs,
                                                 tr_target_graphs,
                                                 ge_input_graphs,
                                                 ge_target_graphs,
                                                 num_training_iterations=num_training_iterations,
                                                 log_dir=output_dir)

    plot_across_training(*tr_info, output_file=f'{output_dir}learning.png')
    plot_predictions(ge_input_graphs, test_values, num_processing_steps_ge, output_file=f'{output_dir}graph.png')

    logit_graphs = graphs_tuple_to_networkxs(test_values["outputs"][-1])

    indexed_ge_graphs = indexed_graphs[tr_ge_split:]
    ge_graphs = [apply_logits_to_graphs(graph, logit_graph) for graph, logit_graph in
                 zip(indexed_ge_graphs, logit_graphs)]

    for ge_graph in ge_graphs:
        for data in multidigraph_data_iterator(ge_graph):
            data['probabilities'] = softmax(data['logits'])
            data['prediction'] = int(np.argmax(data['probabilities']))

    _, _, _, _, _, solveds_tr, solveds_ge = tr_info
    return ge_graphs, solveds_tr, solveds_ge


def configure_embedders(node_types, attr_embedding_dim, categorical_attributes, continuous_attributes):

    def construct_embedder_funcs(node_types, attribute_config, embedder_func):

        attr_embedders = dict()

        # Construct attribute embedders
        for attribute_type, attribute_props in attribute_config.items():

            attr_typ_index = node_types.index(attribute_type)

            # Record the embedder, and the index of the type that it should encode
            attr_embedders[embedder_func(attribute_type, attribute_props)] = [attr_typ_index]

        return attr_embedders

    attr_embedders = dict()

    if categorical_attributes is not None:

        def embedder_func(attribute_type, category_values):
            def make_embedder():
                return CategoricalAttribute(len(category_values), attr_embedding_dim,
                                            name=attribute_type + '_cat_embedder')
            return make_embedder

        attr_embedders.update(construct_embedder_funcs(node_types, categorical_attributes, embedder_func))

    if continuous_attributes is not None:

        def embedder_func(attribute_type, _):
            def make_embedder():
                return ContinuousAttribute(attr_embedding_dim, name=attribute_type + '_cat_embedder')
            return make_embedder

        attr_embedders.update(construct_embedder_funcs(node_types, continuous_attributes, embedder_func))

    attribute_nodes = [l for el in list(attr_embedders.values()) for l in el]

    non_attribute_nodes = []
    for i, _ in enumerate(node_types):
        if i not in attribute_nodes:
            non_attribute_nodes.append(i)

    # All entities and relations (non-attributes) also need an embedder with matching output dimension, which does
    # nothing. This is provided as a list of their indices
    def make_blank_embedder():
        return BlankAttribute(attr_embedding_dim)

    if len(non_attribute_nodes) > 0:
        attr_embedders[make_blank_embedder] = non_attribute_nodes
    return attr_embedders
