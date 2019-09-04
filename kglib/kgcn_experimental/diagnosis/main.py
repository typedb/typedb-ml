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
import tensorflow as tf
from grakn.client import GraknClient
from graph_nets.utils_np import graphs_tuple_to_networkxs

from kglib.kgcn_experimental.diagnosis.data import create_concept_graphs, write_predictions_to_grakn, get_all_types
from kglib.kgcn_experimental.graph_utils.iterate import multidigraph_node_data_iterator, multidigraph_edge_data_iterator
from kglib.kgcn_experimental.graph_utils.prepare import apply_logits_to_graphs, duplicate_edges_in_reverse
from kglib.kgcn_experimental.network.attribute import CategoricalAttribute
from kglib.kgcn_experimental.network.model import KGCN, softmax
from kglib.synthetic_graphs.diagnosis.main import generate_example_graphs


def diagnosis_example(num_graphs=60, num_processing_steps_tr=10, num_processing_steps_ge=10, num_training_iterations=1000):

    # The value at which to split the data into training and evaluation sets
    tr_ge_split = int(num_graphs*0.5)
    keyspace = "diagnosis"
    uri = "localhost:48555"

    generate_example_graphs(num_graphs, keyspace=keyspace, uri=uri)

    # Get the node and edge types, these can simply be passed as a list, but here we query the schema and then remove
    # unwanted types and roles

    client = GraknClient(uri=uri)
    session = client.session(keyspace=keyspace)

    with session.transaction().read() as tx:
        all_node_types, all_edge_types = get_all_types(tx)

    concept_graphs = create_concept_graphs(list(range(num_graphs)), session)

    name_values = 'meningitis', 'flu', 'fever', 'light-sensitivity'
    for graph in concept_graphs:
        for node_data in multidigraph_node_data_iterator(graph):
            typ = node_data['type']
            if typ == 'name':
                node_data['value'] = name_values.index(node_data['value'])
            else:
                node_data['value'] = 0

        for edge_data in multidigraph_edge_data_iterator(graph):
            edge_data['value'] = 0

    tr_graphs = concept_graphs[:tr_ge_split]
    ge_graphs = concept_graphs[tr_ge_split:]

    ind_tr_graphs = [nx.convert_node_labels_to_integers(graph, label_attribute='concept') for graph in tr_graphs]
    ind_ge_graphs = [nx.convert_node_labels_to_integers(graph, label_attribute='concept') for graph in ge_graphs]

    tr_graphs = [duplicate_edges_in_reverse(graph) for graph in ind_tr_graphs]
    ge_graphs = [duplicate_edges_in_reverse(graph) for graph in ind_ge_graphs]

    type_embedding_dim = 5
    attr_embedding_dim = 6

    def make_blank_embedder():

        def attr_encoders(features):
            shape = tf.stack([tf.shape(features)[0], attr_embedding_dim])

            encoded_features = tf.zeros(shape, dtype=tf.float32)
            return encoded_features

        return attr_encoders

    def make_name_embedder():
        return CategoricalAttribute(len(name_values), attr_embedding_dim)

    type_categories_list = list(range(len(all_node_types)))
    non_attribute_nodes = type_categories_list.copy()

    name_index = all_node_types.index('name')

    non_attribute_nodes.pop(name_index)

    attr_encoders = {make_blank_embedder: non_attribute_nodes,
                     make_name_embedder: [name_index]}

    kgcn = KGCN(all_node_types,
                all_edge_types,
                type_embedding_dim,
                attr_embedding_dim,
                attr_encoders,
                latent_size=16, num_layers=2)

    train_values, test_values = kgcn(tr_graphs,
                                     ge_graphs,
                                     num_processing_steps_tr=num_processing_steps_tr,
                                     num_processing_steps_ge=num_processing_steps_ge,
                                     num_training_iterations=num_training_iterations,
                                     log_every_seconds=2)

    logit_graphs = graphs_tuple_to_networkxs(test_values["outputs"][-1])
    ge_graphs = apply_logits_to_graphs(ind_ge_graphs, logit_graphs)

    for ge_graph in ge_graphs:
        for node, data in ge_graph.nodes(data=True):
            data['probabilities'] = softmax(data['logits'])
            data['prediction'] = int(np.argmax(data['probabilities']))

        for sender, receiver, keys, data in ge_graph.edges(keys=True, data=True):
            data['probabilities'] = softmax(data['logits'])
            data['prediction'] = int(np.argmax(data['probabilities']))

    with session.transaction().write() as tx:
        write_predictions_to_grakn(ge_graphs, tx)

    session.close()
    client.close()


if __name__ == "__main__":
    diagnosis_example()
